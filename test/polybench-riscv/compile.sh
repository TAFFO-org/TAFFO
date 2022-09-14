#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM SIGKILL

SCRIPTPATH=$(dirname "$BASH_SOURCE")
cd "$SCRIPTPATH"

TIMEOUT='timeout'
if [[ -z $(which $TIMEOUT) ]]; then
  TIMEOUT='gtimeout'
fi
if [[ ! ( -z $(which $TIMEOUT) ) ]]; then
  TIMEOUT="$TIMEOUT 120"
else
  printf 'warning: timeout command not found\n'
  TIMEOUT=''
fi

if [[ -z $LLVM_DIR ]]; then
  echo -e '\033[33m'"Warning"'\033[39m'" using default llvm/clang";
else
  llvmbin="$LLVM_DIR/bin/";
fi
if [[ -z "$OPT" ]]; then OPT=${llvmbin}opt; fi

if [[ -z $(which taffo) ]]; then
  echo -e '\031[33m'"Error"'\033[39m'" taffo command not found. Install taffo and make sure the place where you installed it is in your PATH!";
fi


compile_one()
{
  benchpath="$1"
  scaling="$2"
  xparams="$3"
  benchname=$(basename $benchpath .c)
  mkdir -p build/"$scaling"/"$benchname"
  $TIMEOUT taffo \
    -o build/"$scaling"/"$benchname"/"$benchname".out \
    -temp-dir build/"$scaling"/"$benchname" \
    "$benchpath" \
    -Isources/. \
    $xparams \
    -debug-taffo \
    -lm \
    -DSCALING_FACTOR=$scaling \
    2> build/"$scaling"/"$benchname"/${benchname}.log || return $?
}

read_opts()
{
  benchpath=$1
  optspath=$(dirname ${benchpath})/$(basename ${benchpath} .c).opts
  if [ -a ${optspath} ]; then
    opts=$(tr '\n' ' ' < ${optspath})
  else
    opts=
  fi
  if [[ "$opts" != *-Xerr* ]]; then
      opts="$opts -Xerr -nounroll -Xerr -startonly"
  fi
  if [[ "$opts" != *-Xvra* ]]; then
      opts="$opts -Xvra -max-unroll=0"
  fi
  
  # filter opts if errorprop is disabled
  if [[ -z $errorprop ]]; then
    skip=0
    for opt in $opts; do
      if [[ ( $opt == '-Xerr' ) && ( $skip -eq 0 ) ]]; then
        skip=$((skip + 2))
      fi
      if [[ $skip -eq 0 ]]; then
        printf '%s ' "$opt"
      else
        skip=$((skip - 1))
      fi
    done
  else
    echo "$opts"
  fi
}

read_stats()
{
  benchname=$1
  scaling=$2
  stats=$(python3 ./stats_to_opts.py ./build_stats/"$scaling"/"$benchname"/"$benchname".csv)
  echo "$stats"
}


if [[ -z $(which taffo) ]]; then
  printf 'error: no taffo in the path\n' > /dev/stderr
  exit 1
fi

TAFFO_PREFIX=$(dirname $(which taffo))/..

SCALING_MAX=64

if [[ -z $mixedmode ]];  then export mixedmode=0; fi
if [[ -z $CFLAGS ]];     then export CFLAGS='-g -O3'; fi
if [[ -z $errorprop ]];  then export errorprop=''; fi # -enable-err
if [[ -z $costmodel ]];  then export costmodel=soc_im_zm; fi
if [[ -z $instrset ]];   then export instrset=embedded; fi
if [[ -z $enobweight ]]; then export enobweight=1; fi
if [[ -z $timeweight ]]; then export timeweight=500; fi
if [[ -z $castweight ]]; then export castweight=200; fi
export mixedmodeopts=""

printf 'Configuration:\n'
printf '  CFLAGS           = %s\n' "$CFLAGS"
printf '  errorprop        = %s\n' "$errorprop"

if [ "$mixedmode" -ne "0" ]; then
  printf '  costmodel        = %s\n' "$costmodel"
  printf '  instrset         = %s\n' "$instrset"
  printf '  enobweight       = %s\n' "$enobweight"
  printf '  timeweight       = %s\n' "$timeweight"
  printf '  castweight       = %s\n' "$castweight"

  mixedmodeopts=" -mixedmode \
  -costmodel "$costmodel" \
  -instructionsetfile="$TAFFO_PREFIX/share/ILP/constrain/$instrset" \
  -Xdta -mixedtuningenob -Xdta "$enobweight" \
  -Xdta -mixedtuningtime -Xdta "$timeweight" \
  -Xdta -mixedtuningcastingtime -Xdta "$castweight" "
fi

mkdir -p build
rm -f build.log

all_benchs=$(cat ./benchmark_list)
for bench in $all_benchs; do
  benchname=$(basename $bench .c)
  opts=$(read_opts ${bench})
  for (( scaling=1; scaling<=SCALING_MAX; scaling=scaling*2 ))
  do
    printf '[....] %s' "$benchname"_"$scaling"
    stats=$(read_stats ${benchname} ${scaling})
    compile_one "$bench" \
      "$scaling" \
      "-m32 \
      ${CFLAGS} \
      ${errorprop} \
      ${mixedmodeopts} \
      ${stats}"
    bpid_fc=$?
    if [[ $bpid_fc == 0 ]]; then
      bpid_fc=' ok '
    fi
    printf '\033[1G[%4s] %s\n' "$bpid_fc" "$benchname"_"$scaling"
    taffo-instmix build/"$scaling"/"$benchname"/${benchname}.out.5.taffotmp.ll \
     1> build_stats/"$scaling"/"$benchname"/${benchname}.mix.txt \
     2> build_stats/"$scaling"/"$benchname"/${benchname}.mix.log.txt
  done
  if [ $benchname = "syrk" ]; then
     break
  fi
done


