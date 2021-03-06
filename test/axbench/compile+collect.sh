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

  # Make stats
  taffo-instmix obj/${bench}.out.fixp.5.taffotmp.ll > stats/${benchsrc}.fixp.mix.txt
  taffo-instmix obj/${bench}.out.fixp.1.taffotmp.ll > stats/${benchsrc}.mix.txt
  taffo-mlfeat obj/${bench}.out.fixp.5.taffotmp.ll > stats/${benchsrc}.fixp.mlfeat.txt
  taffo-mlfeat obj/${bench}.out.fixp.1.taffotmp.ll > stats/${benchsrc}.mlfeat.txt
  ${OPT} -load=${TAFFOLIB} -S -flttofix -dce -stats obj/${bench}.out.fixp.4.taffotmp.ll -o /dev/null 2> stats/${benchsrc}.llvm.txt
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
