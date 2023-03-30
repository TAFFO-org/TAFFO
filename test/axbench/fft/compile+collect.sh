#!/bin/bash

src_dir=$(cd "$(dirname "$0")"; pwd)
stats_dir=$1
if [ -z "${stats_dir}" ]; then
    stats_dir="${src_dir}/stats"
fi
../compile+collect.sh fft main $stats_dir 2048 8192 65536
