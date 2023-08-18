#!/usr/bin/env bash

rm -rf data/output data/sobel
mkdir -p data/output
mkdir -p data/sobel
benchmark=sobel

for f in ./../common/img/*.rgb
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"

  ./${benchmark} ${f} data/output/${filename}_${benchmark}.rgb 2> data/output/${filename}_${benchmark}_scaled.data > data/output/${filename}_${benchmark}_time.txt
  ./${benchmark}-taffo ${f} data/output/${filename}_${benchmark}.rgb.fixp 2> data/output/${filename}_${benchmark}_scaled.data.fixp > data/output/${filename}_${benchmark}_time-taffo.txt
done

