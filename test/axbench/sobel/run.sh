#!/bin/bash

function run_image {
	f=$1
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"

	echo -e "\e[95m------ ${filename} ------\e[0m"
	
	echo -e "\e[96m*** Float Version ***\e[0m"
	time ./bin/${benchmark}.out ${f} data/output/${filename}_${benchmark}.rgb 2> data/sobel/${filename}_${benchmark}.data
	
	echo -e "\e[96m*** Fix Version ***\e[0m"
	time ./bin/${benchmark}.out.fixp ${f} data/output/${filename}_${benchmark}.rgb.fixp 2> data/sobel/${filename}_${benchmark}.data.fixp
	
	echo -e "\e[32m### QoS ###\e[0m"
	
	./../common/scripts/png2rgb.py png data/output/${filename}_${benchmark}.rgb data/output/${filename}_${benchmark}.png > out1.tmp
	./../common/scripts/png2rgb.py png data/output/${filename}_${benchmark}.rgb.fixp data/output/${filename}_${benchmark}.fixp.png > out2.tmp
	
	compare -metric RMSE data/output/${filename}_${benchmark}.png data/output/${filename}_${benchmark}.fixp.png /dev/null > tmp.log 2> tmp.err
	awk '{ printf("*** Relative error: %0.2f%%\n*** Absolute error: %f\n",substr($2, 2, length($2) - 2) * 100,substr($1, 1)) }' tmp.err

	# echo -e "\e[32m### QoS (Sobel) ###\e[0m"
	# ./scripts/qos.py data/sobel/${filename}_${benchmark}.data data/sobel/${filename}_${benchmark}.data.fixp
	rm -f data/sobel/${filename}_${benchmark}.data data/sobel/${filename}_${benchmark}.data.fixp
}

# fix awk decimal number parser
export LANG=en_US.UTF-8

rm -rf data/output data/sobel
mkdir -p data/output
mkdir -p data/sobel
benchmark=sobel

echo "***** AxBench pictures *****"
for f in ./../common/img/*.rgb
do
	run_image $f
done
