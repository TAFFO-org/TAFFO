#!/bin/bash

trap 'echo trapped; kill -s KILL $!; exit 1;' INT

if [[ -z $1 ]]; then
  OUT_DIR='fe-stats'
else
  OUT_DIR=$1
fi

if [[ -z $2 ]]; then
  NEXEC=20
else
  NEXEC=$2
fi

mkdir -p $OUT_DIR

for conf in vra; do
  echo conf = $conf
  ./compile.sh metrics & wait
  ./run.sh --times=$NEXEC & wait
  ./validate.py > $OUT_DIR/${conf}.txt & wait
  mv results-out $OUT_DIR/${conf}
done
