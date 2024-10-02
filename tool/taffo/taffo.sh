#!/usr/bin/env bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

LOG=/dev/null

taffo_setenv_find()
{
  cd $1
  BASEDIR=$(pwd)

  if [[ $2 == 'lib' ]]; then
    SOEXT="so"
    if [ $(uname -s) = "Darwin" ]; then
      SOEXT="dylib";
    fi
    FN="$3.$SOEXT"
  else
    FN="$3"
  fi

  PATH="$BASEDIR/$2/$FN"
  if [[ ! -e $PATH ]] && [[ $2 == 'lib' ]]; then
    PATH="$BASEDIR/${2}64/$FN"
  fi

  if [[ ! -e "$PATH" ]]; then
    echo "Cannot find $FN" >> $LOG
  else
    echo "Found $PATH" >> $LOG
    echo "$PATH"
  fi
}


SCRIPTPATH=$(dirname "$BASH_SOURCE")
TAFFO_PREFIX=${SCRIPTPATH}/..

export TAFFOLIB=$(taffo_setenv_find $TAFFO_PREFIX 'lib' 'Taffo')
export INSTMIX=$(taffo_setenv_find $TAFFO_PREFIX 'bin' 'taffo-instmix')
export TAFFO_MLFEAT=$(taffo_setenv_find $TAFFO_PREFIX 'bin' 'taffo-mlfeat')
export TAFFO_FE=$(taffo_setenv_find $TAFFO_PREFIX 'bin' 'taffo-fe')
export TAFFO_PE=$(taffo_setenv_find $TAFFO_PREFIX 'bin' 'taffo-pe')

if [[ -z "$LLVM_DIR" ]]; then
  LLVM_DIR=$(llvm-config --prefix 2> /dev/null)
  if [[ $? -ne 0 ]]; then
    printf "*** WARNING ***\nCannot set LLVM_DIR using llvm-config\n"
  fi
fi
if [[ ! ( -z "$LLVM_DIR" ) ]]; then
  if [ $(uname -s) = "Darwin" ]; then
    # xcrun patches the command line parameters to clang to add the standard
    # include paths depending on where the currently active platform SDK is
    export CLANG="xcrun $LLVM_DIR/bin/clang"
    export CLANGXX="xcrun $LLVM_DIR/bin/clang++"
  fi
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
llvm_ver=$(${llvmbin}/llvm-config --version)
llvm_ver_maj=${llvm_ver%%.*}

compat_flags_clang=
compat_flags_opt=
if [[ $llvm_ver_maj -ge 15 ]]; then
  compat_flags_clang='-Xclang -no-opaque-pointers'
  compat_flags_opt='-opaque-pointers=0'
fi

sycl=0
linker_command=1;
bufferid_in_file=
bufferid_out_file=
parse_state=0
cuda_kern=0
raw_opts="$@"
input_files=()
output_file="a"
float_output_file=
emit_source=
optimization=
opts=
float_opts=
init_flags=
vra_flags=
disable_vra=0
dta_flags=
dta_inst_set=
conversion_flags=
enable_errorprop=0
errorprop_flags=
errorprop_out=
mem2reg='function(taffomem2reg),'
dontlink=
AUTO_CLANGXX=$CLANG
feedback=0
pe_model_file=
time_profile_file=
temporary_dir=$(mktemp -d)
del_temporary_dir=1
help=0
print_version=0
for opt in $raw_opts; do
  case $parse_state in
    0)
      case $opt in
        -fsycl)
          sycl=1;
          opts="$opts $opt";
          ;;
        -bufferid-import)
          parse_state=15;
          ;;
        -bufferid-export)
          parse_state=16;
          ;;
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
        -Xfloat)
          parse_state=12;
          ;;
        -oclkern)
          init_flags="$init_flags -oclkern"
          ;;
        -cudakern)
          cuda_kern=1
          init_flags="$init_flags -cudakern"
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
            init_flags="$init_flags -debug --stats";
            dta_flags="$dta_flags -debug --stats";
            conversion_flags="$conversion_flags -debug --stats";
            vra_flags="$vra_flags -debug --stats";
            errorprop_flags="$errorprop_flags -debug --stats";
          fi
          LOG=/dev/stderr
          ;;
        -debug-taffo)
          if [[ $llvm_debug -ne 0 ]]; then
            init_flags="$init_flags --debug-only=taffo-init";
            dta_flags="$dta_flags --debug-only=taffo-dta";
            conversion_flags="$conversion_flags --debug-only=taffo-conversion";
            vra_flags="$vra_flags --debug-only=taffo-vra --debug-only=taffo-mem2reg";
            errorprop_flags="$errorprop_flags --debug-only=errorprop";
          fi
          LOG=/dev/stderr
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
        -temp-dir)
          del_temporary_dir=0
          parse_state=9
          ;;
        -err-out)
          enable_errorprop=1
          parse_state=10
          ;;
        -no-mem2reg)
          mem2reg=
          ;;
        -mixedmode)
          dta_flags="$dta_flags -mixedmode=1"
          if [[ -z "$dta_inst_set" ]]; then
            dta_inst_set="-instructionsetfile=$TAFFO_PREFIX/share/ILP/constrain/fix"
          fi
          ;;
        -costmodel)
          parse_state=11
          ;;
        -instructionset)
          parse_state=13
          ;;
        -time-profile-file)
          parse_state=14
          ;;
        -costmodelfilename*)
          dta_flags="$dta_flags $opt"
          ;;
        -instructionsetfile*)
          dta_inst_set="$opt"
          ;;
        -S)
          emit_source="s"
          ;;
        -emit-llvm)
          emit_source="ll"
          ;;
        -print-clang)
          printf '%s\n' "$CLANG"
          exit 0
          ;;
        -print-llvm-bin-dir)
          printf '%s\n' ${llvmbin}
          exit 0
          ;;
        -help | -h | --help)
          help=1
          ;;
        -v | -version | --version)
          print_version=1
          ;;
        -*)
          opts="$opts $opt";
          ;;
        *.cu)
          input_files+=( "$opt" );
          linker_command=0;
          ;;
        *.c | *.ll)
          input_files+=( "$opt" );
          linker_command=0;
          ;;
        *.cpp | *.cc)
          input_files+=( "$opt" );
          linker_command=0;
          AUTO_CLANGXX=$CLANGXX
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
    9)
      temporary_dir="$opt";
      parse_state=0;
      ;;
    10)
      errorprop_out="$opt";
      parse_state=0;
      ;;
    11)
      f="$TAFFO_PREFIX"/share/ILP/cost/"$opt".csv
      if [[ -e "$f" ]]; then
        dta_flags="$dta_flags -costmodelfilename=$f"
      else
        printf 'error: specified cost model "%s" does not exist\n' "$opt"
        exit 1
      fi
      parse_state=0
      ;;
    12)
      float_opts="$float_opts $opt";
      parse_state=0;
      ;;
    13)
      f="$TAFFO_PREFIX"/share/ILP/constrain/"$opt"
      if [[ -e "$f" ]]; then
        dta_inst_set="-instructionsetfile=$f"
      else
        printf 'error: specified instruction set "%s" does not exist\n' "$opt"
        exit 1
      fi
      parse_state=0
      ;;
    14)
      time_profile_file="$opt";
      parse_state=0;
      ;;
    15)
      bufferid_in_file="$opt";
      parse_state=0;
      ;;
    16)
      bufferid_out_file="$opt";
      parse_state=0;
      ;;
  esac;
done

output_basename=$(basename ${output_file})

if [[ $print_version -ne 0 ]]; then
  printf 'clang info:\n-----------\n%s\n\n' "$($CLANG --version)"
  printf 'clang++ info:\n-------------\n%s\n\n' "$($CLANGXX --version)"
  printf 'opt info:\n---------\n%s\n\n' "$($OPT --version)"
  printf 'llc info:\n---------\n%s\n\n' "$($LLC --version)"
  printf 'llvm-link info:\n---------------\n%s\n' "$($LLVM_LINK --version)"
  exit 0
fi

if [[ $help -ne 0 ]]; then
  cat << HELP_END
TAFFO: Tuning Assistant for Floating point and Fixed point Optimization
Usage: taffo [options] file...

The specified files can be any C or C++ source code files.
Apart from the TAFFO-specific options listed below, all CLANG options are
also accepted.

Options:
  -h, --help            Display available options
  -v, --version         Displays information about LLVM components and TAFFO
  -o <file>             Write compilation output to <file>
  -O<level>             Set the optimization level to the specified value.
                        The accepted optimization levels are the same as CLANG.
                        (-O, -O1, -O2, -O3, -Os, -Of)
  -c                    Do not link the output
  -S                    Produce an assembly file in output instead of a binary
                        (overrides -c and -emit-llvm)
  -emit-llvm            Produce a LLVM-IR assembly file in output instead of
                        a binary (overrides -c and -S)
  -mixedmode            Enables experimental data-driven type allocator
    -costmodel <name>          Loads one of the builtin cost models
    -costmodelfilename=<file>  Loads the given the cost model file
                               (produced by taffo-costmodel)
    -instructionset <name>     Loads one of the builtin instruction sets
    -instructionsetfile=<file> Loads the given instruction whitelist file
  -enable-err           Enable the error propagator (disabled by default)
  -err-out <file>       Produce a textual report about the estimates performed
                        by the Error Propagator in the specified file.
  -disable-vra          Disables the VRA analysis pass, and replaces it with
                        a simpler, optimistic, and potentially incorrect greedy
                        algorithm.
  -no-mem2reg           Disable scheduling of the mem2reg pass.
  -float-output <file>  Also compile the files without using TAFFO and store
                        the output to the specified location.
  -time-profile-file <file> Outputs information about the execution time of
                        the various TAFFO passes into the specified file
  -Xinit <option>       Pass the specified option to the Initializer pass of
                        TAFFO
  -Xvra <option>        Pass the specified option to the VRA pass of TAFFO
  -Xdta <option>        Pass the specified option to the DTA pass of TAFFO
  -Xconversion <option> Pass the specified option to the Conversion pass of
                        TAFFO
  -Xerr <option>        Pass the specified option to the Error Propagator pass
                        of TAFFO
  -debug                Enable LLVM and TAFFO debug logging during the
                        compilation.
  -debug-taffo          Enable TAFFO-only debug logging during the compilation.
  -temp-dir <dir>       Store various temporary files related to the execution
                        of TAFFO to the specified directory.

Available builtin cost models:
HELP_END
  for f in "$TAFFO_PREFIX"/share/ILP/cost/*.csv; do
    fn=$(basename "$f")
    printf '  %s\n' ${fn%%.csv}
  done
  echo
  echo 'Available builtin instruction sets:'
  for f in "$TAFFO_PREFIX"/share/ILP/constrain/*; do
    if [[ ! ( $f == *.md ) ]]; then
      fn=$(basename "$f")
      printf '  %s\n' ${fn}
    fi
  done
  exit 0
fi

# enable bash logging
if [[ $LOG != /dev/null ]]; then
  set -x
fi

if [[ $( uname -s ) == 'Darwin' ]]; then
  time_command='date +%s'
else
  time_command='date +%s.%N'
fi
time_string_header=
time_string=
append_time_string () {
  if [[ -z ${time_string_header} ]]; then
    time_string_header+="${1}"
  else
    time_string_header+=",${1}"
  fi
  if [[ -z ${time_string} ]]; then
    time_string+=$($time_command)
  else
    time_string+=,$($time_command)
  fi
}
output_time_string () {
  if [[ ! ( -z ${time_profile_file} ) ]]; then
    printf '%s\n%s\n' ${time_string_header} ${time_string} > ${time_profile_file}
  fi
}

append_time_string "taffo_start"

if [[ ${linker_command} -eq 1 ]]; then
  ${CLANGXX} $optimization $opts -o $output_file #TODO use the proper clang for c and c++ (maybe needs taffo++)
else
  ###
  ###  Produce base .ll
  ###
  if [[ ${#input_files[@]} -eq 1 ]]; then
    # one input file
    if [[ "${input_files##*.}" != "ll" ]]; then
      # not an ll file
      ${CLANG} \
        $opts -D__TAFFO__ -O0 -Xclang -disable-O0-optnone $compat_flags_clang \
        -c -emit-llvm \
        ${input_files} \
        -S -o "${temporary_dir}/${output_basename}.1.taffotmp.ll" || exit $?
    else
      # already an ll file
      cp "${input_files}" "${temporary_dir}/${output_basename}.1.taffotmp.ll"
    fi
  else
    # > 1 input files
    tmp=()
    for input_file in "${input_files[@]}"; do
      thisfn=$(basename "$input_file")
      thisfn=${thisfn%.*}
      thisfn="${temporary_dir}/${output_basename}.${thisfn}.0.taffotmp.ll"
      tmp+=( $thisfn )
      if [[ "${input_file##*.}" != "ll" ]]; then
        # not an ll file
        ${CLANG} \
          $opts -D__TAFFO__ -O0 -Xclang -disable-O0-optnone $compat_flags_clang \
          -c -emit-llvm \
          ${input_file} \
          -S -o "${thisfn}" || exit $?
      else
        # already an ll file
        cp "${input_file}" "${thisfn}"
      fi
    done
    ${LLVM_LINK} \
      ${tmp[@]} \
      -S -o "${temporary_dir}/${output_basename}.1.taffotmp.ll" || exit $?
  fi

  targets=()
  if [[ $sycl -eq 1 ]]; then
    # find target triples
    target_lines=$(sed -n "/__CLANG_OFFLOAD_BUNDLE____START__/p" "${temporary_dir}/${output_basename}.1.taffotmp.ll")
    while IFS= read -r line ; do
      target=${line#*"__CLANG_OFFLOAD_BUNDLE____START__ "}
      output="${temporary_dir}/${output_basename}.${target}.1.taffotmp.ll"
      targets+=( ".${target}" )
      targets_string=${targets_string},${target}
      outputs=${outputs},${output}
    done <<< "$target_lines"
    targets_string=${targets_string#,}
    outputs=${outputs#,}

    # unbundle targets ll files
    clang-offload-bundler -targets="${targets_string}" -type=ll -outputs="${outputs}" -inputs="${temporary_dir}/${output_basename}.1.taffotmp.ll" -unbundle

    # put host target first in the list
    for i in "${!targets[@]}"; do
      if [[ "${targets[$i]}" == *"host"* ]]; then
        host="${targets[$i]}"
        unset 'targets[i]'
        break
      fi
    done
    targets=("$host" "${targets[@]}")

    targets_string=""
    for target in "${targets[@]}"; do
      targets_string=${targets_string},${target#.}
    done
    targets_string=${targets_string#,}

  else
    # single default target
    targets+=( "" )
  fi

  # run taffo passes for each target
  sycl_kernel_variables=()
  sycl_accessor_variables=()
  device_string_definitions=()
  kernel_variables_classes=()
  for target in "${targets[@]}"; do
    bufferid_options=""

    # target is host
    if [[ "$target" == *"host"* ]]; then
      bufferid_options=--bufferid-export=$bufferid_out_file

      # find all host string definitions
      string_definitions=()
      mapfile -t temp < <(grep -E "@\.str(\.([0-9]+))? = " "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll")
      for def in "${temp[@]}"; do
        escaped_def="${def//"\\"/"\\\\"}"
        string_definitions+=("$escaped_def")
      done

      # find sycl accessor annotations in host

      # for each annotation line
      while IFS= read -r line; do
        is_kernel_var=0

        # find all the string definitions used by this annotation
        # and check if it is a kernel variable annotation
        line_copy="$line"
        while [[ "$line_copy" =~ @\.str(\.([0-9]+))? ]]; do
          string_name="${BASH_REMATCH[0]}"

          for def in "${string_definitions[@]}"; do
            if [[ "$def" == *"${string_name} = "* ]]; then
              if [[ "$def" == *"sycl_accessor("* ]] || [[ "$def" == *"sycl_variable("* ]]; then
                is_kernel_var=1
                if [[ "$def" == *"sycl_accessor("* ]]; then
                  sycl_accessor_variables+=(1)
                  kernel_variable=$(echo "$def" | sed -n "s/.*sycl_accessor('\([^']*\)').*/\1/p")
                else
                  sycl_accessor_variables+=(0)
                  kernel_variable=$(echo "$def" | sed -n "s/.*sycl_variable('\([^']*\)').*/\1/p")
                fi
                # save the name of the kernel variable
                sycl_kernel_variables+=("${kernel_variable}")
                # save string definition for device
                device_string_definitions+=("${def}")
              fi
            fi
          done

          line_copy=${line_copy/$string_name/}
        done

        # if it's a kernel variable annotation, save other relevant data for device code
        if [[ $is_kernel_var -eq 1 && "$line" =~ call\ void\ @llvm\.var\.annotation\(i8\*\ %([^,]+), ]]; then
          annotation=$(grep -E "%${BASH_REMATCH[1]} = bitcast .* to i8*" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll")
          variable_class=$(echo "$annotation" | sed -n 's/.*bitcast \([^ ]*\) %.*/\1/p')
          kernel_variables_classes+=("${variable_class}")
        fi
      done < <(grep "call void @llvm.var.annotation" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll")

      # remove duplicate device string definitions
      unique_definitions=()
      declare -A seen
      for def in "${device_string_definitions[@]}"; do
        if [[ -z "${seen[$def]}" ]]; then
          unique_definitions+=("$def")
          seen["$def"]=1
        fi
      done
      device_string_definitions=("${unique_definitions[@]}")

    # target is a device and there are kernel variables to annotate
    elif [[ ${#sycl_kernel_variables[@]} -gt 0 ]]; then
      bufferid_options=--bufferid-import=$bufferid_in_file

      # insert accessor annotations

      # find the index of each variable inside the sycl kernel
      kernel_var_indices=()
      for var in "${sycl_kernel_variables[@]}"; do
        var_gep=$(grep -m 1 -E "${var} = getelementptr .*, .* %__SYCLKernel" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll")
        var_kernel_index=$(echo "$var_gep" | grep -oE '[0-9]+' | tail -n 1)
        kernel_var_indices+=("${var_kernel_index}")
      done

      # extract relevant parts from host annotations
      annotation_parts=()
      parts_tot_len=0
      for def in "${device_string_definitions[@]}"; do
        # extract scalar part only from accessor annotations
        if [[ ${sycl_accessor_variables[j]} -eq 1 ]]; then
          scalar_part=$(extract_scalar_part "$def")
          annotation_parts+=("${scalar_part}")
          parts_tot_len=$((parts_tot_len + ${#scalar_part}))

        # extract whole struct/scalar part from other kernel variable annotations
        else
          # Try to extract struct part if it exists
          annotation_part=$(extract_struct_part "$def")
          # Otherwise extract scalar part
          if [ -z "$annotation_part" ]; then
            annotation_part=$(extract_scalar_part "$def")
          fi
          annotation_parts+=("${annotation_part}")
          parts_tot_len=$((parts_tot_len + ${#annotation_part}))
        fi
      done

      # build sycl kernel annotations

      # count the data elements inside kernel
      sycl_kernel_type=$(grep -m 1 -E "%class.anon = type" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll")
      kernel_content=$(echo "$sycl_kernel_type" | sed -n 's/.*{\(.*\)}.*/\1/p')
      comma_count=$(echo "$kernel_content" | grep -o ',' | wc -l)
      elements_count=$((comma_count + 1))

      sycl_kernel_struct="struct["
      sycl_kernel_struct_len=7
      for ((i=0; i<elements_count; i++)); do
        annotation_part=0
        is_accessor=0
        for ((j=0; j<${#sycl_kernel_variables[@]}; j++)); do
          if [[ ${kernel_var_indices[j]} -eq $i ]]; then
            if [[ ${sycl_accessor_variables[j]} -eq 1 ]]; then
              is_accessor=1
            fi
            annotation_part=${annotation_parts[j]}
            break
          fi
        done
        if [[ "${annotation_part}" != 0 ]]; then
          if [[ $is_accessor -eq 1 ]]; then
            sycl_kernel_struct+="struct[void, struct[${annotation_part}]], "
            sycl_kernel_struct_len=$((sycl_kernel_struct_len + 24))
          else
            sycl_kernel_struct+="${annotation_part}, "
            sycl_kernel_struct_len=$((sycl_kernel_struct_len + 2))
          fi
        else
          sycl_kernel_struct+="void, "
          sycl_kernel_struct_len=$((sycl_kernel_struct_len + 6))
        fi
      done
      sycl_kernel_struct="${sycl_kernel_struct::-2}"
      sycl_kernel_struct+="]"
      sycl_kernel_struct_len=$((sycl_kernel_struct_len - 1))

      # new device string definitions
      device_def_len1=$((22 + sycl_kernel_struct_len + parts_tot_len))
      device_def1="@.taffo.str.1 = private unnamed_addr constant [${device_def_len1} x i8] c\"target('SYCLKernel') ${sycl_kernel_struct}\\\\00\""

      device_def_len2=$((48 + sycl_kernel_struct_len + parts_tot_len))
      device_def2="@.taffo.str.2 = private unnamed_addr constant [${device_def_len2} x i8] c\"target('SYCLRoundedRangeKernel') struct[void, ${sycl_kernel_struct}]\\\\00\""

      device_def3="@.taffo.str.3 = private unnamed_addr constant [9 x i8] c\"main.cpp\\\\00\", section \"llvm.metadata\""

      additional_string_definitions=""
      additional_string_definitions+="$device_def1\n"
      additional_string_definitions+="$device_def2\n"
      additional_string_definitions+="$device_def3\n"

      # kernel annotations
      device_annotation1="  %__SYCLKernel2 = bitcast %\"class.anon\"* %__SYCLKernel to i8*\n"
      device_annotation1+="  call void @llvm.var.annotation(i8* %__SYCLKernel2, i8* getelementptr inbounds ([${device_def_len1} x i8], [${device_def_len1} x i8]* @.taffo.str.1, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.taffo.str.3, i32 0, i32 0), i32 72, i8* null)"
      sed -i "/%__SYCLKernel = alloca/a\\${device_annotation1}" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll"

      device_annotation2="  %RoundedRangeKernel2 = bitcast %\"class.cl::sycl::detail::RoundedRangeKernel\"* %RoundedRangeKernel to i8*\n"
      device_annotation2+="  call void @llvm.var.annotation(i8* %RoundedRangeKernel2, i8* getelementptr inbounds ([${device_def_len2} x i8], [${device_def_len2} x i8]* @.taffo.str.2, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.taffo.str.3, i32 0, i32 0), i32 72, i8* null)"
      sed -i "/%RoundedRangeKernel = alloca/a\\${device_annotation2}" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll"

      # Additional annotations for kernel variables which are not accessors:
      # differently from accessors, which copy the actual data back and forth between host and device,
      # other variables are copied by value and their float value is stored in local variables which need to be converted
      taffo_def_counter=3
      for ((i=0; i<${#sycl_kernel_variables[@]}; i++)); do
        if [[ ${sycl_accessor_variables[i]} -eq 0 ]]; then
          var=${sycl_kernel_variables[i]}
          annotation_part=${annotation_parts[i]}
          ((taffo_def_counter++))
          device_def_len=$((12 + ${#var} + ${#annotation_part}))
          device_def="@.taffo.str.${taffo_def_counter} = private unnamed_addr constant [${device_def_len} x i8] c\"target('${var}') ${annotation_part}\\\\00\""
          additional_string_definitions+="$device_def\n"
          device_annotation="  %_arg_${var}2 = bitcast ${kernel_variables_classes[i]} %_arg_${var}.addr to i8*\n"
          device_annotation+="  call void @llvm.var.annotation(i8* %_arg_${var}2, i8* getelementptr inbounds ([${device_def_len} x i8], [${device_def_len} x i8]* @.taffo.str.${taffo_def_counter}, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.taffo.str.3, i32 0, i32 0), i32 72, i8* null)"
          sed -i "/%_arg_${var}.addr = alloca/a\\${device_annotation}" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll"
        fi
      done

      # find first device string definition and insert new definitions after
      first_def=$(grep -m 1 -E "@\.str(\.([0-9]+))? = " "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll")
      escaped_first_def=$(printf '%s\n' "$first_def" | sed -e 's/[]\/$*.^[]/\\&/g')
      sed -i "/$escaped_first_def/i $additional_string_definitions" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll"

      # insert llvm annotation function declaration and its attributes
      #last_attribute=$(grep -E "attributes #([0-9]+)" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll" | tail -n 1)
      #attribute="${last_attribute#* #}"
      #attribute="${attribute%% = *}"
      #((attribute++))
      #attribute_string="attributes #$attribute = { inaccessiblememonly nocallback nofree nosync nounwind willreturn }"
      #sed -i "/${last_attribute}/a\\${attribute_string}" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll"

      #first_attribute="attributes #0"
      #declaration="; Function Attrs: inaccessiblememonly nocallback nofree nosync nounwind willreturn\ndeclare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #$attribute\n"
      #sed -i "/$first_attribute/i $declaration" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll"
    fi

    # precompute clang invocation for compiling float version
    build_float="${AUTO_CLANGXX} $opts ${optimization} ${float_opts} $compat_flags_clang ${temporary_dir}/${output_basename}${target}.1.taffotmp.ll"

    # Note: in the following we load the plugin both with -load and --load-pass-plugin
    # because the latter does not load the .so until later in the game after command
    # line args are all parsed, and because -load does not register new pass manager
    # passes at all.

    ###
    ###  TAFFO initialization
    ###
    append_time_string "init_start"
    ${OPT} \
      -load "$TAFFOLIB" --load-pass-plugin="$TAFFOLIB" \
      --passes='no-op-module,taffoinit' \
      ${init_flags} \
      -S -o "${temporary_dir}/${output_basename}${target}.2.taffotmp.ll" "${temporary_dir}/${output_basename}${target}.1.taffotmp.ll" || exit $?

    ###
    ###  TAFFO Value Range Analysis
    ###
    append_time_string "vra_start"
    if [[ $disable_vra -eq 0 ]]; then
      ${OPT} \
        -load "$TAFFOLIB" --load-pass-plugin="$TAFFOLIB" \
        --passes="no-op-module,${mem2reg}taffovra" \
        $compat_flags_opt ${vra_flags} \
        -S -o "${temporary_dir}/${output_basename}${target}.3.taffotmp.ll" "${temporary_dir}/${output_basename}${target}.2.taffotmp.ll" || exit $?;
    else
      cp "${temporary_dir}/${output_basename}${target}.2.taffotmp.ll" "${temporary_dir}/${output_basename}${target}.3.taffotmp.ll";
    fi

    feedback_stop=0
    if [[ $feedback -ne 0 ]]; then
      # init the feedback estimator if needed
      base_dta_flags="${dta_flags}"
      dta_flags="${base_dta_flags} "$($TAFFO_FE --init --state "${temporary_dir}/${output_basename}${target}.festate.taffotmp.bin")
    fi
    while [[ $feedback_stop -eq 0 ]]; do
      ###
      ###  TAFFO Data Type Allocation
      ###
      append_time_string "dta_start"
      ${OPT} \
        -load "$TAFFOLIB" --load-pass-plugin="$TAFFOLIB" \
        --passes="no-op-module,taffodta,globaldce" \
        $compat_flags_opt ${dta_flags} ${bufferid_options} ${dta_inst_set} \
        -S -o "${temporary_dir}/${output_basename}${target}.4.taffotmp.ll" "${temporary_dir}/${output_basename}${target}.3.taffotmp.ll" || exit $?

      ###
      ###  TAFFO Conversion
      ###
      append_time_string "conversion_start"
      ${OPT} \
        -load "$TAFFOLIB" --load-pass-plugin="$TAFFOLIB" \
        --passes='no-op-module,taffoconv,globaldce,dce' \
        $compat_flags_opt ${conversion_flags} \
        -S -o "${temporary_dir}/${output_basename}${target}.5.taffotmp.ll" "${temporary_dir}/${output_basename}${target}.4.taffotmp.ll" || exit $?

      ###
      ###  TAFFO Feedback Estimator
      ###
      if [[ ( $enable_errorprop -eq 1 ) || ( $feedback -ne 0 ) ]]; then
        ${OPT} \
          -load "$TAFFOLIB" --load-pass-plugin="$TAFFOLIB" \
          --passes='no-op-module,taffoerr' \
          $compat_flags_opt ${errorprop_flags} \
          -S -o "${temporary_dir}/${output_basename}${target}.6.taffotmp.ll" "${temporary_dir}/${output_basename}${target}.5.taffotmp.ll" 2> "${temporary_dir}/${output_basename}.errorprop.taffotmp.txt" || exit $?
        if [[ ! ( -z "$errorprop_out" ) ]]; then
          cp "${temporary_dir}/${output_basename}${target}.errorprop.taffotmp.txt" "$errorprop_out"
        fi
      fi
      if [[ $feedback -eq 0 ]]; then
        break
      fi
      ${build_float} -S -emit-llvm \
        -o "${temporary_dir}/${output_basename}${target}.float.taffotmp.ll" || exit $?
      ${TAFFO_PE} \
        --fix "${temporary_dir}/${output_basename}${target}.5.taffotmp.ll" \
        --flt "${temporary_dir}/${output_basename}${target}.float.taffotmp.ll" \
        --model ${pe_model_file} > "${temporary_dir}/${output_basename}${target}.perfest.taffotmp.txt" || exit $?
      newflgs=$(${TAFFO_FE} \
        --pe-out "${temporary_dir}/${output_basename}${target}.perfest.taffotmp.txt" \
        --ep-out "${temporary_dir}/${output_basename}${target}.errorprop.taffotmp.txt" \
        --state "${temporary_dir}/${output_basename}${target}.festate.taffotmp.bin" || exit $?)
      if [[ ( "$newflgs" == 'STOP' ) || ( -z "$newflgs" ) ]]; then
        feedback_stop=1
      else
        dta_flags="${base_dta_flags} ${newflgs}"
      fi
    done

    inputs=${inputs},"${temporary_dir}/${output_basename}${target}.5.taffotmp.ll"
  done
  inputs=${inputs#,}

  if [[ $sycl -eq 1 ]]; then
    # bundle targets ll files back together
    clang-offload-bundler -targets="${targets_string}" -type=ll -inputs="${inputs}" -outputs="${temporary_dir}/${output_basename}.5.taffotmp.ll"
  fi

  ###
  ###  Backend
  ###
  append_time_string "backend_start"
  # Produce the requested output file
  if [[ ( $emit_source == "s" ) || ( $del_temporary_dir -eq 0 ) ]]; then
    if [[ $cuda_kern == 1 ]]; then
      ${CLANG} \
      $opts -target nvptx64-nvidia-cuda ${optimization} $compat_flags_clang \
      -c \
      "${temporary_dir}/${output_basename}.5.taffotmp.ll" \
      -S -o "${temporary_dir}/${output_basename}.taffotmp.s" || exit $?
    else
      ${CLANG} \
      $opts ${optimization} $compat_flags_clang \
      -c \
      "${temporary_dir}/${output_basename}.5.taffotmp.ll" \
      -S -o "${temporary_dir}/${output_basename}.taffotmp.s" || exit $?
    fi
  fi
  if [[ $emit_source == "s" ]]; then
    cp "${temporary_dir}/${output_basename}.taffotmp.s" "$output_file"
  elif [[ $emit_source == "ll" ]]; then
    cp "${temporary_dir}/${output_basename}.5.taffotmp.ll" "$output_file"
  else
    ${AUTO_CLANGXX} \
      $opts ${optimization} $compat_flags_clang \
      ${dontlink} \
      "${temporary_dir}/${output_basename}.5.taffotmp.ll" \
      -o "$output_file" || exit $?
  fi

  if [[ ! ( -z ${float_output_file} ) ]]; then
    if [[ $emit_source == 's' ]]; then
      type_opts='-S'
    elif [[ $emit_source == 'll' ]]; then
      type_opts='-S -emit-llvm'
    fi
    ${build_float} \
      ${dontlink} -S \
      -o "${temporary_dir}/${output_basename}.float.taffotmp.s" || exit $?
    ${build_float} \
      ${dontlink} ${type_opts} \
      -o "$float_output_file" || exit $?
  fi
fi

###
###  Cleanup
###
if [[ $del_temporary_dir -ne 0 ]]; then
  rm -rf "${temporary_dir}"
fi

append_time_string "taffo_end"
output_time_string
