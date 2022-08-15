#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

if [[ -z $PASSLIB ]]; then
  echo -e '\033[31m'"Error"'\033[39m'" please set PASSLIB to the location of LLVMFloatToFixed.so";
fi
if [[ -z $LLVM_DIR ]]; then
  echo -e '\033[33m'"Warning"'\033[39m'" using default llvm/clang";
else
  llvmbin="$LLVM_DIR/bin/";
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
feedback=0
pe_model_file=
for opt in $raw_opts; do
  case $parse_state in
    0)
      case $opt in
        -Xinit)
          parse_state=2;
          ;;
        -Xvra)
          parse_state=5;
          ;;
        -Xdta)
          parse_state=3;
          ;;
        -Xconversion)
          parse_state=4;
          ;;
        -Xerr)
          enable_errorprop=1
          parse_state=8;
          ;;
        -o*)
          if [[ ${#opt} -eq 2 ]]; then
            parse_state=1;
          else
            output_file=`echo "$opt" | cut -b 2`;
          fi;
          ;;
        -float-output)
          parse_state=6;
          ;;
        -O*)
          optimization=$opt;
          ;;
        -fixp*)
          conversion_flags="$conversion_flags $opt";
          ;;
        -c)
          dontlink="$opt";
          ;;
        -debug)
          if [[ $llvm_debug -ne 0 ]]; then
            init_flags="$init_flags -debug";
            dta_flags="$dta_flags -debug";
            conversion_flags="$conversion_flags -debug";
            vra_flags="$vra_flags -debug";
            errorprop_flags="$errorprop_flags -debug";
          fi
          ;;
        -debug-taffo)
          if [[ $llvm_debug -ne 0 ]]; then
            init_flags="$init_flags --debug-only=taffo-init";
            dta_flags="$dta_flags --debug-only=taffo-dta";
            conversion_flags="$conversion_flags --debug-only=taffo-conversion";
            vra_flags="$vra_flags --debug-only=taffo-vra";
            errorprop_flags="$errorprop_flags --debug-only=errorprop";
          fi
          ;;
        -disable-vra)
          disable_vra=1
          ;;
        -enable-err)
          enable_errorprop=1
          ;;
        -feedback)
          feedback=1
          ;;
        -pe-model)
          parse_state=7
          ;;
        -*)
          opts="$opts $opt";
          ;;
        *.c | *.ll)
          input_files+=( "$opt" );
          ;;
        *.cpp | *.cc)
          input_files+=( "$opt" );
          iscpp=$CLANGXX
          ;;
        *)
          opts="$opts $opt";
          ;;
      esac;
      ;;
    1)
      output_file="$opt";
      parse_state=0;
      ;;
    2)
      init_flags="$init_flags $opt";
      parse_state=0;
      ;;
    3)
      dta_flags="$dta_flags $opt";
      parse_state=0;
      ;;
    4)
      conversion_flags="$conversion_flags $opt";
      parse_state=0;
      ;;
    5)
      vra_flags="$vra_flags $opt";
      parse_state=0;
      ;;
    6)
      float_output_file="$opt";
      parse_state=0;
      ;;
    7)
      pe_model_file="$opt";
      parse_state=0;
      ;;
    8)
      errorprop_flags="$errorprop_flags $opt";
      parse_state=0;
      ;;
  esac;
done

# enable bash logging
set -x

###
###  Produce base .ll
###
if [[ ${#input_files[@]} -eq 1 ]]; then
  # one input file
  ${CLANG} \
    $opts -O0 -Xclang -disable-O0-optnone \
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
    tmp+=( $thisfn )
    ${CLANG} \
      $opts -O0 -Xclang -disable-O0-optnone \
      -c -emit-llvm \
      ${input_file} \
      -S -o "${thisfn}" || exit $?
  done
  ${LLVM_LINK} \
    ${tmp[@]} \
    -S -o "${output_file}.1.magiclangtmp.ll" || exit $?
fi

# precompute clang invocation for compiling float version
build_float="${iscpp} $opts ${optimization} ${output_file}.1.magiclangtmp.ll"

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
    -load "$VRALIB" \
    -mem2reg -taffoVRA \
    ${vra_flags} \
    -S -o "${output_file}.3.magiclangtmp.ll" "${output_file}.2.magiclangtmp.ll" || exit $?;
else
  cp "${output_file}.2.magiclangtmp.ll" "${output_file}.3.magiclangtmp.ll";
fi

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
    ${dta_flags} \
    -S -o "${output_file}.4.magiclangtmp.ll" "${output_file}.3.magiclangtmp.ll" || exit $?
    
  ###
  ###  TAFFO Conversion
  ###
  ${OPT} \
    -load "$PASSLIB" \
    -flttofix -globaldce -dce \
    ${conversion_flags} \
    -S -o "${output_file}.5.magiclangtmp.ll" "${output_file}.4.magiclangtmp.ll" || exit $?
    
  ###
  ###  TAFFO Feedback Estimator
  ###
  if [[ ( $enable_errorprop -eq 1 ) || ( $feedback -ne 0 ) ]]; then
    ${OPT} \
      -load "$ERRORLIB" \
      -errorprop -startonly \
      ${errorprop_flags} \
      -S -o "${output_file}.6.magiclangtmp.ll" "${output_file}.5.magiclangtmp.ll" 2> "${output_file}.errorprop.magiclangtmp.txt" || exit $?
  fi
  if [[ $feedback -eq 0 ]]; then
    break
  fi
  ${build_float} -S -emit-llvm \
    -o "${output_file}.float.magiclangtmp.ll" || exit $?
  ${TAFFO_PE} \
    --fix "${output_file}.5.magiclangtmp.ll" \
    --flt "${output_file}.float.magiclangtmp.ll" \
    --model ${pe_model_file} > "${output_file}.perfest.magiclangtmp.txt" || exit $?
  newflgs=$(${TAFFO_FE} \
    --pe-out "${output_file}.perfest.magiclangtmp.txt" \
    --ep-out "${output_file}.errorprop.magiclangtmp.txt" \
    --state "${output_file}.festate.magiclangtmp.bin" || exit $?)
  if [[ ( "$newflgs" == 'STOP' ) || ( -z "$newflgs" ) ]]; then
    feedback_stop=1
  else
    dta_flags="${base_dta_flags} ${newflgs}"
  fi
done

###
###  Backend
###


##${CLANG} \
##  $opts ${optimization} \
##  -c \
##  "${output_file}.5.magiclangtmp.ll" \
##  -S -o "$output_file.magiclangtmp.s" || exit $?
##${iscpp} \
##  $opts ${optimization} \
##  ${dontlink} \
##  "${output_file}.5.magiclangtmp.ll" \
##  -o "$output_file" || exit $?
##if [[ ! ( -z ${float_output_file} ) ]]; then
##  ${build_float} \
##    ${dontlink} \
##    -o "$float_output_file" || exit $?
##fi
BENCH=${output_file}.5.magiclangtmp

DIR_BUILD_LLVM13=/home/denisovlev/Data/llvm12-riscv

LLVM13_LLC=${DIR_BUILD_LLVM13}/bin/llc
LLVM13_CLANG=${DIR_BUILD_LLVM13}/bin/clang
LLVM13_STRIP=${DIR_BUILD_LLVM13}/bin/llvm-strip
LLVM13_OBJDUMP=${DIR_BUILD_LLVM13}/bin/llvm-objdump

# Compiler back-end
${LLVM13_LLC} -O3 ${BENCH}.ll --march=riscv32 --mattr=+m,+experimental-zm -fast-isel=false -o ${BENCH}.s > /dev/null
#${LLVM13_LLC} -O3 ${BENCH}.ll --march=riscv32 --mattr=+f,+m -fast-isel=false -o ${BENCH}.s > /dev/null
#${LLVM13_LLC} -O3 ${BENCH}.ll --march=riscv32 --mattr=+m -fast-isel=false -o ${BENCH}.s > /dev/null
#${LLVM13_LLC} -O3 ${BENCH}.ll --march=riscv32 --mattr=+m,+f,+experimental-zm -fast-isel=false -o ${BENCH}.s > /dev/null

# x86-64
#${LLVM13_LLC} -O3 ${BENCH}.ll --march=x86-64 -fast-isel=false -o ${BENCH}.s > /dev/null
#${LLVM13_CLANG} -march=x86-64 ${BENCH}.s ./low_level/crt0.S -o ${BENCH}.elf -T ./low_level/riscv32_mod.ld -nostdlib -L/work/llvm_compiler/lampBench/1_flowScripts/ -lgcc -fno-unwind-tables

# Linking 
# Hard-float
${LLVM13_CLANG} -march=rv32imf -mabi=ilp32 ${BENCH}.s ./low_level/crt0.S -o ${BENCH}.elf -T ./low_level/riscv32_mod.ld -nostdlib -L/home/davide/Documents/tools/riscv32-8.2.0/lib/gcc/riscv32-unknown-elf/8.2.0/ -lgcc -fno-unwind-tables
# Soft-float
#${LLVM13_CLANG} -march=rv32im -mabi=ilp32 ${BENCH}.s ./low_level/crt0.S -o ${BENCH}.elf -T ./low_level/riscv32_mod.ld -nostdlib -L/work/llvm_compiler/lampBench/1_flowScripts/ -lgcc -fno-unwind-tables

# Strip ELF
${LLVM13_STRIP} --strip-all --strip-debug --remove-section .riscv.attributes -o ${BENCH}_strip.elf ${BENCH}.elf

# Objdump strip and non-strip ELF
#riscv32-unknown-elf-objdump -D ${BENCH}_strip.elf > ${BENCH}_strip_riscv32_gcc.objdump
#riscv32-unknown-elf-objdump -D ${BENCH}.elf > ${BENCH}_riscv32_gcc.objdump

${LLVM13_OBJDUMP} -D ${BENCH}_strip.elf > ${BENCH}_strip.objdump
${LLVM13_OBJDUMP} -D ${BENCH}.elf > ${BENCH}.objdump

# Objcopy stripped ELF - NOTE: llvm-objcopy does not support Verilog output
riscv32-unknown-elf-objcopy ${BENCH}_strip.elf ${BENCH}.vmem.tmp -O verilog --remove-section=.comment --remove-section=.sdata --reverse-bytes=4 

# Generate .vmem file
./utils/vmem_formatter ${BENCH}.vmem.tmp ${BENCH}.vmem
