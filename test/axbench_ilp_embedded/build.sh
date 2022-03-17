#!/usr/bin/env bash

export SCRIPTPATH=$(cd $(dirname "$BASH_SOURCE") && pwd)
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

build_one_embedded()
{
  # $1: bench name
  bench="$1"

  pushd "$bench" > /dev/null
  logdir="../../log/$bench"
  mkdir -p "../../embedded_src/bench_obj"
  mkdir -p "$logdir"
  main="bench_${bench}_${costmodel}_${enobweight}_${timeweight}_${castweight}"
  main=${main/-/_}
  out="../../embedded_src/bench_obj/${main}.o"
  log="$logdir/$main.log"
  extra_cxx_includes="$embedded_sysroot"/include/c++/7.3.1/arm-none-eabi
  echo $bench > "$log"
  taffo \
    *.cc -I../../embedded_src -I../../embedded_src/stm32f207 -c -o "$out" $CFLAGS \
    -DBENCH_MAIN="$main" \
    -mixedmode \
    -costmodel "$costmodel" \
    -instructionsetfile="$TAFFO_PREFIX/share/ILP/constrain/$instrset" \
    -Xdta -mixedtuningenob -Xdta "$enobweight" \
    -Xdta -mixedtuningtime -Xdta "$timeweight" \
    -Xdta -mixedtuningcastingtime -Xdta "$castweight" \
    -Xvra -unroll -Xvra 0 \
    -debug-taffo \
    -temp-dir "$logdir" \
    -stdlib=libstdc++ -I$extra_cxx_includes \
    --target="$embedded_sysroot" -mcpu="$embedded_cpu" --sysroot="$embedded_sysroot" -fshort-enums \
      &>> "$log"
  err=$?
  if [[ err -eq 0 ]]; then
    printf "%s" "$main"
  fi

  popd > /dev/null
  return $err
}
export -f build_one_embedded

build_one_embedded_float()
{
  # $1: bench name
  bench="$1"
  
  pushd "$bench" > /dev/null
  logdir="../../log/$bench"
  mkdir -p "../../embedded_src/bench_obj"
  mkdir -p "$logdir"
  main_flt="bench_${bench}_orig"
  main=${main_flt/-/_}
  log="$logdir/${main_flt}.log"
  extra_cxx_includes="$embedded_sysroot"/include/c++/7.3.1/arm-none-eabi
  echo $bench > "$log"
  CLANG=$(taffo -print-clang)
  for f in *.cc; do
    out_flt="../../embedded_src/bench_obj/${f}_orig.o"
    ${CLANG} \
      "$f" -I../../embedded_src -I../../embedded_src/stm32f207 -c -o "$out_flt" $CFLAGS \
      -DBENCH_MAIN="$main_flt" \
      -stdlib=libstdc++ -I${extra_cxx_includes} \
      --target="$embedded_sysroot" -mcpu="$embedded_cpu" --sysroot="$embedded_sysroot" -fshort-enums \
        &>> "$log"
    err=$?
    if [[ err -ne 0 ]]; then
      return $err
    fi
  done
  printf "%s" "$main"
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

if [[ -z $X_IS_CHILD ]]; then
  echo > "$SCRIPTPATH"/embedded_src/bench_main.c.in
  echo > "$SCRIPTPATH"/embedded_src/bench_main.h
  rm -rf "$SCRIPTPATH"/embedded_src/bench_obj

  if [[ -z $CFLAGS ]];     then export CFLAGS="-g -O3"; fi
  if [[ -z $embedded_sysroot ]]; then export embedded_sysroot=/usr/local/arm-none-eabi; fi
  if [[ -z $embedded_triple ]];  then export embedded_triple=arm-none-eabi; fi
  if [[ -z $embedded_cpu ]];     then export embedded_cpu=cortex-m3; fi
  if [[ -z $costmodel ]];  then export costmodel=stm32; fi
  if [[ -z $instrset ]];   then export instrset=embedded; fi
  if [[ -z $enobweight ]]; then export enobweight=1; fi
  if [[ -z $timeweight ]]; then export timeweight=100; fi
  if [[ -z $castweight ]]; then export castweight=100; fi

  printf 'Configuration:\n'
  printf '  CFLAGS           = %s\n' "$CFLAGS"
  printf '  embedded_sysroot = %s\n' "$embedded_sysroot"
  printf '  embedded_triple  = %s\n' "$embedded_triple"
  printf '  embedded_cpu     = %s\n' "$embedded_cpu"
  printf '  costmodel        = %s\n' "$costmodel"
  printf '  instrset         = %s\n' "$instrset"
fi

if [[ ( $# -gt 0 ) && ( $1 == clean ) ]]; then
  action=clean
  shift
fi
if [[ ( $# -gt 0 ) && ( $1 == build_experiment ) ]]; then
  action=build_experiment
  shift
fi
if [[ -z $action ]]; then
  action=build
fi

printf '  action           = %s\n' "$action"

if [[ $# -eq 0 ]]; then
  benchs=*/
else
  benchs=$@
fi

cd "$SCRIPTPATH/src"
for benchdir in $benchs; do
  bench=${benchdir%/}
  case $action in
    build)
      printf '  enobweight       = %s\n' "$enobweight"
      printf '  timeweight       = %s\n' "$timeweight"
      printf '  castweight       = %s\n' "$castweight"
      printf '%-5s %-16s' "$action" "$bench"
      main=$(build_one_embedded $bench)
      out=$?
      if [[ $out -eq 0 ]]; then
        printf ' OK!\n'
        printf 'printf("%%s\n", "%s");\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
        printf '%s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
        printf 'void %s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.h
      else
        printf ' fail %d\n' $out 
      fi
      ;;
    build_float)
      printf '%-5s %-16s' "$action" "$bench"
      main=$(build_one_embedded_float $bench)
      out=$?
      if [[ $out -eq 0 ]]; then
        printf ' OK!\n'
        printf 'printf("%%s\\n", "%s");\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
        printf '%s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
        printf 'void %s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.h
      else
        printf ' fail %d\n' $out 
      fi
      ;;
    build_experiment)
      printf '%-5s %-16s' "$action" "$bench"
      main=$(build_one_embedded_float $bench)
      out=$?
      if [[ $out -eq 0 ]]; then
        printf ' OK!\n'
        printf 'printf("%%s\\n", "%s");\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
        printf '%s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
        printf 'void %s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.h
      else
        printf ' fail %d\n' $out 
      fi
      if [[ -z $wstart ]]; then export wstart=0; fi
      if [[ -z $wend   ]]; then export wend=100000; fi
      if [[ -z $wstep  ]]; then export wstep=50000; fi
      printf '  wstart           = %s\n' "$wstart"
      printf '  wend             = %s\n' "$wend"
      printf '  wstep            = %s\n' "$wstep"
      for (( i = wstart; i <= wend; i = i + wstep )); do
        export enobweight=$i
        export timeweight=$(( wend - i ))
        export castweight=$(( wend - i ))
        printf '  enobweight       = %s\n' "$enobweight" > /dev/stderr
        printf '  timeweight       = %s\n' "$timeweight" > /dev/stderr
        printf '  castweight       = %s\n' "$castweight" > /dev/stderr
        printf '%-5s %-16s' "$action" "$bench" > /dev/stderr
        #printf 'enobweight=%d timeweight=%d castweight=%d build_one_embedded %s\n' $enobweight $timeweight $castweight $bench
        main=$(build_one_embedded $bench)
        out=$?
        if [[ $out -eq 0 ]]; then
          printf ' OK!\n' > /dev/stderr
          printf 'printf("%%s\\n", "%s");\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
          printf '%s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.c.in
          printf 'void %s();\n' "$main" >> "$SCRIPTPATH"/embedded_src/bench_main.h
        else
          printf ' fail %d\n' $out > /dev/stderr
        fi
      done
      ;;
    clean)
      printf '%-5s %-16s' "$action" "$bench"
      clean_one $bench
      out=$?
      if [[ $out -eq 0 ]]; then
        printf ' OK!\n'
      else
        printf ' fail %d\n' $out
      fi
      ;;
  esac
done
