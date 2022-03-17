#!/usr/bin/env bash

export SCRIPTPATH=$(cd $(dirname "$BASH_SOURCE") && pwd)
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

build_one()
{
  # $1: bench name

  bench="$1"
  pushd "$bench" > /dev/null
  out="${bench}_${costmodel}_${enobweight}_${timeweight}_${castweight}"
  float_out="${bench}_unmodified"
  taffo \
    *.c -I.. -o "$out" $CFLAGS -D$pb_dataset -lm \
    -float-output "$float_out" \
    -mixedmode \
    -costmodel "$costmodel" \
    -instructionsetfile="$TAFFO_PREFIX/share/ILP/constrain/$instrset" \
    -Xdta -mixedtuningenob -Xdta "$enobweight" \
    -Xdta -mixedtuningtime -Xdta "$timeweight" \
    -Xdta -mixedtuningcastingtime -Xdta "$castweight" \
    -Xvra -unroll -Xvra 0 \
    -debug-taffo \
    -temp-dir . \
      &> "$out.log"
  err=$?

  popd > /dev/null
  return $err
}

build_one_embedded()
{
  # $1: bench name

  bench="$1"
  pushd "$bench" > /dev/null
  mkdir -p "../../embedded_src/bench_obj"
  main="bench_${bench}_${costmodel}_${enobweight}_${timeweight}_${castweight}"
  main=${main/-/_}
  out="../../embedded_src/bench_obj/${main}.o"
  taffo \
    *.c -I../../embedded_src -I../../embedded_src/stm32f207 -c -o "$out" $CFLAGS -D$pb_dataset \
    -DBENCH_MAIN="$main" \
    -mixedmode \
    -costmodel "$costmodel" \
    -instructionsetfile="$TAFFO_PREFIX/share/ILP/constrain/$instrset" \
    -Xdta -mixedtuningenob -Xdta "$enobweight" \
    -Xdta -mixedtuningtime -Xdta "$timeweight" \
    -Xdta -mixedtuningcastingtime -Xdta "$castweight" \
    -Xvra -unroll -Xvra 0 \
    -debug-taffo \
    -temp-dir . \
    --target="$embedded_sysroot" -mcpu="$embedded_cpu" --sysroot="$embedded_sysroot" -fshort-enums \
      &> "$out.log"
  err=$?
  if [[ err -eq 0 ]]; then
    main_flt="bench_${bench}_orig"
    main_flt=${main_flt/-/_}
    asm_flt="../../embedded_src/bench_obj/${main_flt}.s"
    out_flt="../../embedded_src/bench_obj/${main_flt}.o"
    CLANG=$(taffo -print-clang)
    echo 'FLOAT VERSION' >> "$out.log"
    cp "${main}.o.taffotmp.s" "../../embedded_src/bench_obj/${main}.s"
    ${CLANG} \
      *.c -I../../embedded_src -I../../embedded_src/stm32f207 -S -o "$asm_flt" $CFLAGS -D$pb_dataset \
      -DBENCH_MAIN="$main_flt" \
      --target="$embedded_sysroot" -mcpu="$embedded_cpu" --sysroot="$embedded_sysroot" -fshort-enums \
        &>> "$out.log"
    ${CLANG} \
      "$asm_flt" -c -o "$out_flt" \
      --target="$embedded_sysroot" -mcpu="$embedded_cpu" --sysroot="$embedded_sysroot" -fshort-enums \
        &>> "$out.log"
    printf '%s();\n' "$main" >> ../../embedded_src/bench_main.c.in
    printf '%s();\n' "$main_flt" >> ../../embedded_src/bench_main.c.in
    printf 'void %s();\n' "$main" >> ../../embedded_src/bench_main.h
    printf 'void %s();\n' "$main_flt" >> ../../embedded_src/bench_main.h
  fi

  popd > /dev/null
  return $err
}

run_one()
{
  executable="${bench}_${costmodel}_${enobweight}_${timeweight}_${castweight}"
  float_executable="${bench}_unmodified"

  pushd "$bench" > /dev/null
  ./$executable > taffo_out.txt 2> taffo_time.txt
  ./$float_executable > float_out.txt 2> float_time.txt

  nfo=$($SCRIPTPATH/error.py float_out.txt taffo_out.txt)
  err=$?
  printf '%-19s' "$nfo"

  popd > /dev/null
  return $err
}

clean_one()
{
  # $1: bench name
  bench="$1"
  pushd "$bench" > /dev/null
  rm -rf *.dSYM
  rm -f *.ll *.log ${bench}_* *.txt *.s
  popd > /dev/null
}

if [[ -z $(which taffo) ]]; then
  printf 'error: no taffo in the path\n' > /dev/stderr
  exit 1
fi

TAFFO_PREFIX=$(dirname $(which taffo))/..

if [[ -z $costmodel ]];  then export costmodel=stm32; fi
if [[ -z $instrset ]];   then export instrset=fix; fi
if [[ -z $enobweight ]]; then export enobweight=1; fi
if [[ -z $timeweight ]]; then export timeweight=100; fi
if [[ -z $castweight ]]; then export castweight=100; fi
if [[ -z $pb_dataset ]]; then export pb_dataset=MINI_DATASET; fi
if [[ -z $CFLAGS ]];     then export CFLAGS="-g -O3"; fi

printf 'Configuration:\n'
printf '  costmodel        = %s\n' "$costmodel"
printf '  instrset         = %s\n' "$instrset"
printf '  enobweight       = %s\n' "$enobweight"
printf '  timeweight       = %s\n' "$timeweight"
printf '  castweight       = %s\n' "$castweight"
printf '  pb_dataset       = %s\n' "$pb_dataset"
printf '  CFLAGS           = %s\n' "$CFLAGS"

if [[ ( $# -gt 0 ) && ( $1 == build ) ]]; then
  action=build
  shift
fi
if [[ ( $# -gt 0 ) && ( $1 == clean ) ]]; then
  action=clean
  shift
fi
if [[ ( $# -gt 0 ) && ( $1 == run ) ]]; then
  action=run
  shift
fi
if [[ ( $# -gt 0 ) && ( $1 == build_embedded ) ]]; then
  action=build_embedded
  echo > "$SCRIPTPATH"/embedded_src/bench_main.c.in
  echo > "$SCRIPTPATH"/embedded_src/bench_main.h
  rm -rf "$SCRIPTPATH"/embedded_src/bench_obj
  if [[ -z $embedded_sysroot ]]; then export embedded_sysroot=/usr/local/arm-none-eabi; fi
  if [[ -z $embedded_triple ]];  then export embedded_triple=arm-none-eabi; fi
  if [[ -z $embedded_cpu ]];     then export embedded_cpu=cortex-m3; fi
  printf '  embedded_sysroot = %s\n' "$embedded_sysroot"
  printf '  embedded_triple  = %s\n' "$embedded_triple"
  printf '  embedded_cpu     = %s\n' "$embedded_cpu"
  shift
fi
if [[ -z $action ]]; then export action=build; fi
printf '  action           = %s\n' "$action"

if [[ $# -eq 0 ]]; then
  benchs=*/
else
  benchs=$@
fi

cd "$SCRIPTPATH/src"
for benchdir in $benchs; do
  bench=${benchdir%/}
  printf '%-5s %-16s' "$action" "$bench"
  case $action in
    build)
      build_one $bench
      ;;
    build_embedded)
      build_one_embedded $bench
      ;;
    run)
      run_one $bench
      ;;
    clean)
      clean_one $bench
      ;;
  esac
  out=$?
  if [[ $out -eq 0 ]]; then
    printf ' OK!\n'
  else
    printf ' fail %d\n' $out
  fi
done
