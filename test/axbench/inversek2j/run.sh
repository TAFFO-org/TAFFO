#!/bin/bash

rm -rf data/output
mkdir -p data/output
benchmark=inversek2j

for f in data/input/*.data
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	
	echo -e "\e[95m------ ${filename} ------\e[0m"
	
	echo -e "\e[96m*** Float Version ***\e[0m"
	time ./bin/inversek2j.out ${f} data/output/${filename}_${benchmark}_out.data
	
	echo -e "\e[96m*** Fix Version ***\e[0m"
	time ./bin/inversek2j.out.fixp ${f} data/output/${filename}_${benchmark}_out.data.fixp
	
	echo -e "\e[32m### QoS ###\e[0m"
	python ./scripts/qos.py data/output/${filename}_${benchmark}_out.data data/output/${filename}_${benchmark}_out.data.fixp
done
