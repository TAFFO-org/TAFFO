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

benchmark=inversek2j

for f in data/input/*.data
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  
  mfloat=$(match_time "`cat data/output/${filename}_${benchmark}_time.txt`")
  mfix=$(match_time "`cat data/output/${filename}_${benchmark}_time-taffo.txt`")
  error=$(./scripts/qos.py data/output/${filename}_${benchmark}_out.data data/output/${filename}_${benchmark}_out.data.fixp)
  mabs_error=$(match_error "$error" 'Absolute Error')
  mrel_error=$(match_error "$error" 'Relative Error')
  
  printf "$FORMAT" "${benchmark}_${filename}" $mfix $mfloat "$mrel_error" "$mabs_error"
done
