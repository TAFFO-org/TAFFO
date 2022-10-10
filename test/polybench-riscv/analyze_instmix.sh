#!/bin/bash

SCALING_MAX=1024

all_benchs=$(cat ./benchmark_list)
for bench in $all_benchs; do
  benchname=$(basename $bench .c)
  for (( scaling=1; scaling<=SCALING_MAX; scaling=scaling*4 ))
  do
    printf '[....] %s' "$benchname"_"$scaling"
    taffo-instmix build_float/"$scaling"/"$benchname"/${benchname}.float.out.ll \
     1> build_stats_float/"$scaling"/"$benchname"/${benchname}.mix.txt \
     2> build_stats_float/"$scaling"/"$benchname"/${benchname}.mix.log.txt
     printf '\033[1G[%4s] %s\n' "$bpid_fc" "$benchname"_"$scaling"
  done
done
