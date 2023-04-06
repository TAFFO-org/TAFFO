#!/bin/bash

rm -rf data/output
mkdir data/output
benchmark=inversek2j

for f in data/input/*.data
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  
  ./inversek2j ${f} data/output/${filename}_${benchmark}_out.data > data/output/${filename}_${benchmark}_time.txt
  ./inversek2j-taffo ${f} data/output/${filename}_${benchmark}_out.data.fixp > data/output/${filename}_${benchmark}_time-taffo.txt
done
