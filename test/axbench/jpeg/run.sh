#!/bin/bash

rm -rf data/output
mkdir data/output
benchmark=jpeg

for f in ./../common/img/*.rgb
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	
	echo -e "\e[95m------ ${filename} ------\e[0m"
	
	echo -e "\e[96m*** Float Version ***\e[0m"
	time ./bin/${benchmark}.out ${f} data/output/${filename}_${benchmark}_out.jpg
	
	echo -e "\e[96m*** Fix Version ***\e[0m"
	time ./bin/${benchmark}.out.fixp ${f} data/output/${filename}_${benchmark}_out.jpg.fixp
	
	echo -e "\e[32m### QoS ###\e[0m"
	compare -metric RMSE data/output/${filename}_${benchmark}_out.jpg.fixp data/output/${filename}_${benchmark}_out.jpg null > tmp.log 2> tmp.err
	awk '{ printf("*** Error: %0.2f%\n",substr($2, 2, length($2) - 2) * 100) }' tmp.err
done
