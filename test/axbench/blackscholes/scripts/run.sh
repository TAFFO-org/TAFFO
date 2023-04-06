#!/usr/bin/env bash

rm -rf data/output
mkdir data/output
benchmark=blackscholes

for f in data/input/*.data
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  
  ./${benchmark} ${f} data/output/${filename}_${benchmark}_out.data > data/output/${filename}_${benchmark}_time.txt
  ./${benchmark}-taffo ${f} data/output/${filename}_${benchmark}_out.data.fixp > data/output/${filename}_${benchmark}_time-taffo.txt
done
