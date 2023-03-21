#!/usr/bin/env bash

mkdir -p _experiments
exp_dir=_experiments/$(date +%Y-%m-%d_%H-%M-%S)
mkdir ${exp_dir}
raw_data_dir=${exp_dir}/raw_data

#host_dta_set='fixp f32 mixed'
#kern_args_set='fixp f32 f16 mixed'
#kern_dta_set='fixp f32 f16 mixed'
#host_dta_set='f32'
#kern_args_set='mixed'
#kern_dta_set='f16'
host_dta_set='fixp f32'
kern_args_set='fixp f32 f16 mixed'
kern_dta_set='fixp f32 f16'

for host_dta in $host_dta_set; do
  for kern_args in $kern_args_set; do
    for kern_dta in $kern_dta_set; do
      conf="${host_dta}_${kern_args}_${kern_dta}"
      exp_file_prefix="${exp_dir}/${conf}"
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

      raw=$(pwd)/${raw_data_dir}/${conf}
      mkdir -p ${raw}
      find . -regex '\./[^_].*\.exe\.txt' -execdir cp '{}' ${raw} ';'

      echo validate
      make -s validate &> ${exp_file_prefix}_validate.txt || continue
    done
  done
done

