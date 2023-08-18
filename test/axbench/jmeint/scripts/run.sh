#!/usr/bin/env bash

rm -rf data/output
mkdir -p data/output
benchmark=jmeint

# decompress data files
pushd ./data/input > /dev/null
for f in *.data.bz2; do
  uncompressed="${f%.bz2}"
  if [[ ! ( -e ${uncompressed} ) ]]; then
    realf=$(readlink ${f})
    if [[ $? -ne 0 ]]; then
      realf=${f}
    fi
    bunzip2 -ck -- ${realf} > ${uncompressed}
  fi
done
popd > /dev/null
# cleanup stale data files
for f in data/input/*.data; do
  compressed="${f}.bz2"
  if [[ ! ( -e ${compressed} ) ]]; then
    rm ${f}
  fi
done

for f in data/input/*.data
do
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  
  ./${benchmark} ${f} data/output/${filename}_${benchmark}_out.data > data/output/${filename}_${benchmark}_time.txt
  ./${benchmark}-taffo ${f} data/output/${filename}_${benchmark}_out.data.fixp > data/output/${filename}_${benchmark}_time-taffo.txt
done
