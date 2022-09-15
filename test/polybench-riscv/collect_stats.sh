#!/bin/bash

SCALING_MAX=1024

compile_stats()
{
  benchpath="$1"
  scaling="$2"
  benchname=$(basename $benchpath .c)
  mkdir -p build_stats/"$scaling"/"$benchname"
  cc \
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
  for (( scaling=1; scaling<=SCALING_MAX; scaling=scaling*4 ))
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
#  if [ $benchname = "trisolv" ]; then
#     break
#  fi
done