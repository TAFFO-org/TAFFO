#!/bin/bash

if [[ -z $FORMAT ]]; then
  FORMAT='%40s %12s %12s%5s%6s%12s%12s\n'
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

rm -rf data/output
mkdir data/output
benchmark=blackscholes

for f in data/input/*.data
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  
  if [[ -z $NORUN ]]; then
    float=$(./bin/${benchmark}.out ${f} data/output/${filename}_${benchmark}_out.data)
    mfloat=$(match_time "$float")
  
    fix=$(./bin/${benchmark}.out.fixp ${f} data/output/${filename}_${benchmark}_out.data.fixp)
    mfix=$(match_time "$fix")
  else
    mfloat='0'
    mfix='0'
  fi
  
  if [[ -z $NOERROR ]]; then
    error=$(./scripts/qos.py data/output/${filename}_${benchmark}_out.data data/output/${filename}_${benchmark}_out.data.fixp)
    mabs_error=$(match_error "$error" 'Absolute error')
    mrel_error=$(match_error "$error" 'Relative error')
  else
    mabs_error='0'
    mrel_error='0'
  fi
  
  printf "$FORMAT" "${benchmark}_${filename}" $mfix $mfloat 0 0 "$mrel_error" "$mabs_error"
done
