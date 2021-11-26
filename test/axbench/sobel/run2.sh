#!/bin/bash

# fix awk decimal number parser
export LANG=en_US.UTF-8


if [[ -z $FORMAT ]]; then
  FORMAT='%25s %12s %12s%5s%6s%12s%12s%12s%12s\n'
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

rm -rf data/output data/sobel
mkdir -p data/output
mkdir -p data/sobel
benchmark=sobel

for f in ./../common/img/*.rgb
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"

  if [[ -z $NORUN ]]; then
    float=$(./bin/${benchmark}.out ${f} data/output/${filename}_${benchmark}.rgb 2> data/sobel/${filename}_${benchmark}.data)
    mfloat=$(match_time "$float")
    
    fix=$(./bin/${benchmark}.out.fixp ${f} data/output/${filename}_${benchmark}.rgb.fixp 2> data/sobel/${filename}_${benchmark}.data.fixp)
    mfix=$(match_time "$fix")
  else
    mfloat='0'
    mfix='0'
  fi
  
  if [[ -z $NOERROR ]]; then
    python3 ./../common/scripts/png2rgb.py png data/output/${filename}_${benchmark}.rgb data/output/${filename}_${benchmark}.png > out1.tmp
    python3 ./../common/scripts/png2rgb.py png data/output/${filename}_${benchmark}.rgb.fixp data/output/${filename}_${benchmark}.fixp.png > out2.tmp
    
    compare -metric RMSE data/output/${filename}_${benchmark}.png data/output/${filename}_${benchmark}.fixp.png /dev/null > tmp.log 2> tmp.err
    mrel_error=$(awk '{ printf("%0.6f", substr($2, 2, length($2) - 2) * 100) }' tmp.err)

    if [[ $SCALERROR ]]; then
	error=$(python ./scripts/qos.py data/sobel/${filename}_${benchmark}.data data/sobel/${filename}_${benchmark}.data.fixp)
	msabs_error=$(match_error "$error" 'Absolute error')
	msrel_error=$(match_error "$error" 'Relative error')
    else
	msabs_error=
	msrel_error=
    fi
    rm -f data/sobel/${filename}_${benchmark}.data data/sobel/${filename}_${benchmark}.data.fixp
  else
    mrel_error='0'
  fi
  
  printf "$FORMAT" "${benchmark}_${filename}" $mfix $mfloat '0' '0' "$mrel_error" '-' $msrel_error $msabs_error
done

