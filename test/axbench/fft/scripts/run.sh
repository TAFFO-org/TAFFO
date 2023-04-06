#!/bin/bash

rm -rf data/output
mkdir -p data/output
benchmark=fft

input=( 2048 8192 65536 4194304)

for f in "${input[@]}"
do
  ./${benchmark} ${f} data/output/${f}_${benchmark}_out.data > data/output/${f}_${benchmark}_time.txt
  ./${benchmark}-taffo ${f} data/output/${f}_${benchmark}_out.data.fixp > data/output/${f}_${benchmark}_time-taffo.txt
done
