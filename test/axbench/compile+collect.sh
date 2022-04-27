#!/bin/bash

if [[ $# -lt 2 ]]; then
  cat <<HELP_END
Usage: $0 BENCHNAME BENCHSRC [STATS_DIR]
If STATS_DIR is specified, instmix and mlfeat stats are copied there.
Set DONT_REBUILD to only copy stats without rebuilding.
Set ENABLE_ERROR to enable error propagation.
Set NO_MEM2REG to disable mem2reg pass before VRA.
LLVM_DIR and TAFFOLIB must be set appropriately.
HELP_END
  exit 0
fi

if [[ -z $TAFFOLIB ]]; then
  printf "Please set TAFFOLIB by running ./setenv.sh."
  exit 1
fi

bench=$1
benchsrc=$2

errorprop=
if [[ -n $ENABLE_ERROR ]]; then
  errorprop="-enable-err -Xerr -startonly -err-out stats/errorprop.log"
fi

no_mem2reg=
if [[ -n $NO_MEM2REG ]]; then
  no_mem2reg=-no-mem2reg
fi

if [[ -z "$LLVM_DIR" ]]; then
  LLVM_DIR=$(llvm-config --prefix 2> /dev/null)
  if [[ $? -ne 0 ]]; then
    printf "*** ERROR ***\nCannot set LLVM_DIR using llvm-config\n"
    exit 1
  fi
fi
OPT=${LLVM_DIR}/bin/opt
CLANG=${LLVM_DIR}/bin/clang
CLANGXX=${LLVM_DIR}/bin/clang++
LLC=${LLVM_DIR}/bin/llc

debug=
if [[ $(${LLVM_DIR}/bin/llvm-config --build-mode) == Debug ]]; then
  debug=-debug-taffo
fi

if [[ -z $DONT_REBUILD ]]; then
  mkdir -p obj
  mkdir -p bin
  mkdir -p stats

  shopt -s extglob
  files='src/*.@(cc|cpp|c)'

  taffo -temp-dir obj \
    -o bin/${bench}.out.fixp \
    -float-output bin/${bench}.out \
    -O3 \
    -I../common/src \
    ${files} \
    ${debug} \
    ${errorprop} \
    ${no_mem2reg} \
    -lm \
    2> stats/taffo.log

  dynamic_analysis() {
    ${OPT} -load=${TAFFOLIB} -S \
    --taffoinit --taffo-name-variables -globaldce -dce -stats \
    obj/${bench}.out.fixp.1.taffotmp.ll \
    -o obj/${bench}.out.named.taffotmp.ll

    ${OPT} -load=${TAFFOLIB} -S \
        --taffo-inject-func-call -stats \
        obj/${bench}.out.named.taffotmp.ll \
        -o obj/${bench}.out.instrumented.taffotmp.ll

    ${LLC} -filetype=obj obj/${bench}.out.instrumented.taffotmp.ll -o obj/${bench}.out.instrumented.taffotmp.o

    ${CLANGXX} obj/${bench}.out.instrumented.taffotmp.o -o bin/${bench}.out.instrumented

    bin/${bench}.out.instrumented \
    /home/denisovlev/Projects/TAFFO/test/axbench/blackscholes/data/input/blackscholesTrain_100K.data \
    /dev/null \
    > obj/${bench}.out.instrumented.trace

    ${OPT} -load=${TAFFOLIB} -S \
            -O0 --taffo-read-trace -stats -trace_file obj/${bench}.out.instrumented.trace \
            obj/${bench}.out.named.taffotmp.ll \
            -o obj/${bench}.out.dynamic.taffotmp.ll

    ${OPT} -load=${TAFFOLIB} -S \
                -stats --taffodta \
                obj/${bench}.out.dynamic.taffotmp.ll \
                -o obj/${bench}.out.dynamic_flttofix.taffotmp.ll

    ${OPT} -load=${TAFFOLIB} -S \
                    -stats --flttofix -globaldce -dce \
                    obj/${bench}.out.dynamic_flttofix.taffotmp.ll \
                    -o obj/${bench}.out.dynamic_final.taffotmp.ll

    ${LLC} -filetype=obj obj/${bench}.out.dynamic_final.taffotmp.ll -o obj/${bench}.out.dynamic_final.taffotmp.o

    ${CLANGXX} -O3 obj/${bench}.out.dynamic_final.taffotmp.o -o bin/${bench}.out.dynamic_final
  }

  # Make stats
  taffo-instmix obj/${bench}.out.fixp.5.taffotmp.ll > stats/${benchsrc}.fixp.mix.txt
  taffo-instmix obj/${bench}.out.fixp.1.taffotmp.ll > stats/${benchsrc}.mix.txt
  taffo-mlfeat obj/${bench}.out.fixp.5.taffotmp.ll > stats/${benchsrc}.fixp.mlfeat.txt
  taffo-mlfeat obj/${bench}.out.fixp.1.taffotmp.ll > stats/${benchsrc}.mlfeat.txt
  taffo-instmix obj/${bench}.out.dynamic_final.taffotmp.ll > stats/${benchsrc}.dynamic_final.mix.txt
  taffo-mlfeat obj/${bench}.out.dynamic_final.taffotmp.ll > stats/${benchsrc}.dynamic_final.mlfeat.txt
  ${OPT} -load=${TAFFOLIB} -S -flttofix -dce -stats obj/${bench}.out.fixp.4.taffotmp.ll -o /dev/null 2> stats/${benchsrc}.llvm.txt
  dynamic_analysis
fi

# $3 stats directory
if [[ $# -ge 3 ]]; then
  cp stats/${benchsrc}.fixp.mix.txt $3/${bench}.imix.txt
  cp stats/${benchsrc}.mix.txt $3/${bench}.float.imix.txt
  cp stats/${benchsrc}.fixp.mlfeat.txt $3/${bench}.mlfeat.txt
  cp stats/${benchsrc}.mlfeat.txt $3/${bench}.float.mlfeat.txt
  cp stats/${benchsrc}.llvm.txt $3/${bench}.llstat.txt
  # cat stats/errorprop.log | grep 'Computed error for target' > $3/${bench}.errprop.txt
fi
