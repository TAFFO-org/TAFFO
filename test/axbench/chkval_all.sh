#!/bin/bash

export FORMAT='%40s %12s %12s%11s%11s%14s%20s\n'


# $1 benchmark to check
check()
{
  pushd $1 > /dev/null
  ./run2.sh
  popd > /dev/null
}


printf "$FORMAT" '' 'fix T' 'flo T' '# ofl fix' '# ofl flo' 'avg err %' 'avg abs err'

for arg; do
  case $arg in
    --noerror)
      export NOERROR=1
      ;;
    --norun)
      export NORUN=1
      ;;
    *)
      check $arg
      exit 0
  esac
done

check 'blackscholes'
check 'blackscholes_B00'
check 'blackscholes_B10'
check 'blackscholes_B11'
check 'blackscholes_B12'

check 'fft'
check 'fft_00'
check 'fft_01'
check 'fft_02'

check 'inversek2j_00'
check 'inversek2j_01'
check 'inversek2j_02'
check 'inversek2j_03'
check 'inversek2j_04'
check 'inversek2j_05'
check 'inversek2j_06'
check 'inversek2j_base'

check 'jmeint_00'
check 'jmeint_01'
check 'jmeint_base'

check 'kmeans'
check 'kmeans_00'
check 'kmeans_01'
check 'kmeans_base_00'
check 'kmeans_base_01'

check 'sobel'
check 'sobel_16bit'



