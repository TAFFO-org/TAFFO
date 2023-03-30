#!/bin/bash

src_dir=$(cd "$(dirname "$0")"; pwd)
stats_dir=$1
if [ -z "${stats_dir}" ]; then
    stats_dir="${src_dir}/stats"
fi
input1="${src_dir}/data/input/blackscholesTrain_100K.data"
../compile+collect.sh blackscholes blackscholes $stats_dir $input1
