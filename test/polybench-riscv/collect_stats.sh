#!/bin/bash

SCALING_MAX=4

if [[ -z $LLVM_DIR ]]; then
  echo -e '\033[33m'"Warning"'\033[39m'" using default llvm/clang";
else
  llvmbin="$LLVM_DIR/bin/";
  if [[ -z "$CLANG" ]]; then CLANG=${llvmbin}clang; fi
  if [[ -z "$CLANGXX" ]]; then CLANGXX=${CLANG}++; fi
fi
if [[ -z "$OPT" ]]; then OPT=${llvmbin}opt; fi

compile_stats()
{
  benchpath="$1"
  scaling="$2"
  benchname=$(basename $benchpath .c)
  mkdir -p build_stats/"$scaling"/"$benchname"
  "$CLANG" \
    -o build_stats/"$scaling"/"$benchname"/"$benchname".out \
    "$benchpath" \
    -Isources/. \
    -DCOLLECT_STATS \
    -DSCALING_FACTOR=$scaling \
    2> build_stats/"$scaling"/"$benchname"/${benchname}.log || return $?
}

mkdir -p build_stats
rm -f build_stats.log

all_benchs=$(cat ./benchmark_list)
for bench in $all_benchs; do
  benchname=$(basename $bench .c)
  for (( scaling=1; scaling<=SCALING_MAX; scaling=scaling*2 ))
  do
     printf '[....] %s' "$bench"_"$scaling"
     compile_stats "$bench" "$scaling"
     bpid_fc=$?
     if [[ $bpid_fc == 0 ]]; then
       bpid_fc=' ok '
     fi
     printf '\033[1G[%4s] %s\n' "$bpid_fc" "$bench"_"$scaling"
     build_stats/"$scaling"/"$benchname"/"$benchname".out 2> build_stats/"$scaling"/"$benchname"/"$benchname".csv
  done
#  if [ $benchname = "2mm" ]; then
#     break
#  fi
done