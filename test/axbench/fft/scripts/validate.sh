#!/usr/bin/env bash

if [[ -z $FORMAT ]]; then
  FORMAT='%40s %12s %12s%12s%12s\n'
fi


match_time()
{
  regex='^kernel[^0-9]*([^ ]+)'
  if [[ ( $1 =~ $regex ) ]]; then
    echo ${BASH_REMATCH[1]}
  else
    echo -1
  fi
}

match_error()
{
  regex="^[^${2:0:1}]*$2[^0-9]*([0-9.]+)"
  if [[ ( $1 =~ $regex ) ]]; then
    echo ${BASH_REMATCH[1]}
  else
    echo -1
  fi
}

benchmark=fft

input=( 2048 8192 65536 4194304)

for f in "${input[@]}"
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  
  mfloat=$(match_time "`cat data/output/${f}_${benchmark}_time.txt`")
  mfix=$(match_time "`cat data/output/${f}_${benchmark}_time-taffo.txt`")
  error=$(./scripts/qos.py data/output/${f}_${benchmark}_out.data data/output/${f}_${benchmark}_out.data.fixp)
  mabs_error=$(match_error "$error" 'Absolute error')
  mrel_error=$(match_error "$error" 'Relative error')
  
  printf "$FORMAT" "${benchmark}_${filename}" $mfix $mfloat "$mrel_error" "$mabs_error"
done
