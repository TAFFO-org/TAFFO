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
      exp_file_prefix="${exp_dir}/${host_dta}_${kern_args}_${kern_dta}"
      echo host_dta=${host_dta} kern_args=${kern_args} kern_dta=${kern_dta}
      make clean &> /dev/null || continue

      echo build
      make TAFFO_HOST_DTA=${host_dta} TAFFO_KERN_ARGS=${kern_args} TAFFO_KERN_DTA=${kern_dta} -j`nproc` &> ${exp_file_prefix}_build.txt
      build_res=$?
      find . -name 'taffo.log' | xargs cat >> ${exp_file_prefix}_build.txt
      if [[ $build_res != 0 ]]; then
        continue
      fi
      
      echo run
      make -s run &> ${exp_file_prefix}_run.txt || continue

      echo validate
      make -s validate &> ${exp_file_prefix}_validate.txt || continue
    done
  done
done

