#!/usr/bin/env bash

rm -rf data/output
mkdir -p data/output
benchmark=kmeans

for f in ./../common/img/*.rgb
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"

  ./${benchmark} ${f} data/output/${filename}_${benchmark}.rgb > data/output/${filename}_${benchmark}_time.txt
  ./${benchmark}-taffo ${f} data/output/${filename}_${benchmark}.rgb.fixp > data/output/${filename}_${benchmark}_time-taffo.txt
  
  printf "$FORMAT" "${benchmark}_${filename}" $mfix $mfloat '0' '0' "$mrel_error" '-'
done
