#!/bin/bash

OUTFILE="similarbits.txt"

# $1 similarbits
check()
{
  echo "SIMILARBITS="$1 >> $OUTFILE
  ../../../magiclang2.sh decoder.cpp -enable-err -Xerr -abserror -ljpeg -o decoder-opt -O3 -Xdta --similarbits=$1
  ./decoder-opt ../benchmark-dataset/rgb16bit/ >> $OUTFILE
  echo >> $OUTFILE
  grep "Computed error for target" ./decoder-opt.errorprop.magiclangtmp.txt >> $OUTFILE
}

check_bench()
{
  rm $OUTFILE
  for sb in {0..32}
  do
      check $sb
  done
}

# enable bash logging
set -x

check_bench
