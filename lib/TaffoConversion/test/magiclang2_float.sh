#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

if [[ -z $PASSLIB ]]; then
  echo -e '\033[31m'"Error"'\033[39m'" please set PASSLIB to the location of LLVMFloatToFixed.so"
fi
if [[ -z $LLVM_DIR ]]; then
  echo -e '\033[33m'"Warning"'\033[39m'" using default llvm/clang"
else
  llvmbin="$LLVM_DIR/bin/"
fi
if [[ -z "$CLANG" ]]; then CLANG=${llvmbin}clang; fi
if [[ -z "$CLANGXX" ]]; then CLANGXX=${CLANG}++; fi
if [[ -z "$OPT" ]]; then OPT=${llvmbin}opt; fi
if [[ -z "$LLC" ]]; then LLC=${llvmbin}llc; fi
if [[ -z "$LLVM_LINK" ]]; then LLVM_LINK=${llvmbin}llvm-link; fi

llvm_debug=$(($("$OPT" --version | grep DEBUG | wc -l)))

parse_state=0
raw_opts="$@"
input_files=()
output_file="a"
float_output_file=
optimization=
opts=
init_flags=
vra_flags=
disable_vra=0
dta_flags=
conversion_flags=
enable_errorprop=0
errorprop_flags=
dontlink=
iscpp=$CLANG
honest_mode=0
feedback=0
pe_model_file=
for opt in $raw_opts; do
  case $parse_state in
  0)
    case $opt in
    -Xinit)
      parse_state=2
      ;;
    -Xvra)
      parse_state=5
      ;;
    -Xdta)
      parse_state=3
      ;;
    -Xconversion)
      parse_state=4
      ;;
    -Xerr)
      enable_errorprop=1
      parse_state=8
      ;;
    -o*)
      if [[ ${#opt} -eq 2 ]]; then
        parse_state=1
      else
        output_file=$(echo "$opt" | cut -b 2)
      fi
      ;;
    -float-output)
      parse_state=6
      ;;
    -O*)
      optimization=$opt
      ;;
    -fixp*)
      conversion_flags="$conversion_flags $opt"
      ;;
    -c)
      dontlink="$opt"
      ;;
    -debug)
      if [[ $llvm_debug -ne 0 ]]; then
        init_flags="$init_flags -debug"
        dta_flags="$dta_flags -debug"
        conversion_flags="$conversion_flags -debug"
        vra_flags="$vra_flags -debug"
        errorprop_flags="$errorprop_flags -debug"
      fi
      ;;
    -debug-taffo)
      if [[ $llvm_debug -ne 0 ]]; then
        init_flags="$init_flags --debug-only=taffo-init"
        dta_flags="$dta_flags --debug-only=taffo-dta"
        conversion_flags="$conversion_flags --debug-only=taffo-conversion"
        vra_flags="$vra_flags --debug-only=taffo-vra"
        errorprop_flags="$errorprop_flags --debug-only=errorprop"
      fi
      ;;
    -disable-vra)
      disable_vra=1
      ;;
    -enable-err)
      enable_errorprop=1
      ;;
    -honest-mode)
      # actually, whether this mode is more honest or not is still
      # subject of active research
      honest_mode=1
      ;;
    -feedback)
      feedback=1
      ;;
    -pe-model)
      parse_state=7
      ;;
    -*)
      opts="$opts $opt"
      ;;
    *.c | *.ll)
      input_files+=("$opt")
      ;;
    *.cpp | *.cc)
      input_files+=("$opt")
      iscpp=$CLANGXX
      ;;
    *)
      opts="$opts $opt"
      ;;
    esac
    ;;
  1)
    output_file="$opt"
    parse_state=0
    ;;
  2)
    init_flags="$init_flags $opt"
    parse_state=0
    ;;
  3)
    dta_flags="$dta_flags $opt"
    parse_state=0
    ;;
  4)
    conversion_flags="$conversion_flags $opt"
    parse_state=0
    ;;
  5)
    vra_flags="$vra_flags $opt"
    parse_state=0
    ;;
  6)
    float_output_file="$opt"
    parse_state=0
    ;;
  7)
    pe_model_file="$opt"
    parse_state=0
    ;;
  8)
    errorprop_flags="$errorprop_flags $opt"
    parse_state=0
    ;;
  esac
done

# enable bash logging
#set -x

###
###  Produce base .ll
###
if [[ ${#input_files[@]} -eq 1 ]]; then
  # one input file
  ${CLANG} \
    $opts -O0 \
    -c -emit-llvm \
    ${input_files} \
    -S -o "${output_file}.1.magiclangtmp.ll" || exit $?
else
  # > 1 input files
  tmp=()
  for input_file in "${input_files[@]}"; do
    thisfn=$(basename "$input_file")
    thisfn=${thisfn%.*}
    thisfn="${output_file}.${thisfn}.0.magiclangtmp.ll"
    tmp+=($thisfn)
    ${CLANG} \
      $opts -O0 \
      -c -emit-llvm \
      ${input_file} \
      -S -o "${thisfn}" || exit $?
  done
  ${LLVM_LINK} \
    ${tmp[@]} \
    -S -o "${output_file}.1.magiclangtmp.ll" || exit $?
fi

# precompute clang invocation for compiling float version
if [[ $honest_mode -ne 0 ]]; then
  build_float="${iscpp} $opts ${optimization} ${input_files[*]}"
else
  build_float="${iscpp} $opts ${optimization} ${output_file}.1.magiclangtmp.ll"
fi

###
###  TAFFO initialization
###
${OPT} \
  -load "$INITLIB" \
  -taffoinit \
  ${init_flags} \
  -S -o "${output_file}.2.magiclangtmp.ll" "${output_file}.1.magiclangtmp.ll" || exit $?

###
###  TAFFO Value Range Analysis
###
if [[ $disable_vra -eq 0 ]]; then
  ${OPT} \
    -debug \
    -load "$VRALIB" \
    -mem2reg -taffoVRA \
    ${vra_flags} \
    -S -o "${output_file}.3.magiclangtmp.ll" "${output_file}.2.magiclangtmp.ll" 2> "VRA.txt" || exit $?
else
  cp "${output_file}.2.magiclangtmp.ll" "${output_file}.3.magiclangtmp.ll"
fi

###
###  TAFFO Precision Analysis
###
#${OPT} -debug \
#  -load "$PRALIB" \
#  -mem2reg -taffoPRA \
#  ${vra_flags} \
#  -S -o "${output_file}.4.magiclangtmp.ll" "${output_file}.3.magiclangtmp.ll" || exit $?

feedback_stop=0
if [[ $feedback -ne 0 ]]; then
  # init the feedback estimator if needed
  base_dta_flags="${dta_flags}"
  dta_flags="${base_dta_flags} "$($TAFFO_FE --init --state "${output_file}.festate.magiclangtmp.bin")
fi
while [[ $feedback_stop -eq 0 ]]; do
  ###
  ###  TAFFO Data Type Allocation
  ###
  ${OPT} \
    -load "$TUNERLIB" \
    -taffodta -globaldce \
    -forcefloat 3 \
    ${dta_flags} \
    -S -o "${output_file}.4.magiclangtmp.ll" "${output_file}.3.magiclangtmp.ll" || exit $?

  ###
  ###  TAFFO Conversion
  ###
  ${OPT} \
    -load "$PASSLIB" \
    -flttofix -globaldce -dce \
    -debug \
    ${conversion_flags} \
    -S -o "${output_file}.5.magiclangtmp.ll" "${output_file}.4.magiclangtmp.ll" || exit $?

  ###
  ###  TAFFO Feedback Estimator
  ###
  if [[ ($enable_errorprop -eq 1) || ($feedback -ne 0) ]]; then
    ${OPT} \
      -load "$ERRORLIB" \
      -errorprop -startonly \
      ${errorprop_flags} \
      -S -o "${output_file}.6.magiclangtmp.ll" "${output_file}.5.magiclangtmp.ll" 2>"${output_file}.errorprop.magiclangtmp.txt" || exit $?
  fi
  if [[ $feedback -eq 0 ]]; then
    break
  fi
  ${build_float} -S -emit-llvm \
    -o "${output_file}.float.magiclangtmp.ll" || exit $?
  ${TAFFO_PE} \
    --fix "${output_file}.5.magiclangtmp.ll" \
    --flt "${output_file}.float.magiclangtmp.ll" \
    --model ${pe_model_file} >"${output_file}.perfest.magiclangtmp.txt" || exit $?
  newflgs=$(${TAFFO_FE} \
    --pe-out "${output_file}.perfest.magiclangtmp.txt" \
    --ep-out "${output_file}.errorprop.magiclangtmp.txt" \
    --state "${output_file}.festate.magiclangtmp.bin" || exit $?)
  if [[ ("$newflgs" == 'STOP') || (-z "$newflgs") ]]; then
    feedback_stop=1
  else
    dta_flags="${base_dta_flags} ${newflgs}"
  fi
done

###
###  Backend
###
${CLANG} \
  $opts ${optimization} \
  -c \
  "${output_file}.5.magiclangtmp.ll" \
  -S -o "$output_file.magiclangtmp.s" || exit $?
${iscpp} \
  $opts ${optimization} \
  ${dontlink} \
  "${output_file}.5.magiclangtmp.ll" \
  -o "$output_file" || exit $?
if [[ ! (-z ${float_output_file}) ]]; then
  ${build_float} \
    ${dontlink} \
    -o "$float_output_file" || exit $?
fi
