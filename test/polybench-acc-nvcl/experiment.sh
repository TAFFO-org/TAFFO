#!/usr/bin/env bash

exp_dir=_experiment_$(date +%Y-%m-%d_%H-%M-%S)
mkdir ${exp_dir}

host_dta_set='fixp f32 mixed'
kern_args_set='fixp f32 f16 mixed'
kern_dta_set='fixp f32 f16 mixed'
#host_dta_set='fixp'
#kern_args_set='f16'
#kern_dta_set='f32'

for host_dta in $host_dta_set; do
  for kern_args in $kern_args_set; do
    for kern_dta in $kern_dta_set; do
      make clean || continue
      make TAFFO_HOST_DTA=${host_dta} TAFFO_KERN_ARGS=${kern_args} TAFFO_KERN_DTA=${kern_dta} || continue
      make -s run > ${exp_dir}/${host_dta}_${kern_args}_${kern_dta}_run.txt || continue
      make -s validate > ${exp_dir}/${host_dta}_${kern_args}_${kern_dta}_validate.txt || continue
    done
  done
done

