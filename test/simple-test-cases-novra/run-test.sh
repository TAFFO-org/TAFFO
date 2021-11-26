#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM SIGKILL

SCRIPTPATH=$(dirname "$BASH_SOURCE")

TIMEOUT='timeout'
if [[ -z $(which $TIMEOUT) ]]; then
  TIMEOUT='gtimeout'
fi
if [[ ! ( -z $(which $TIMEOUT) ) ]]; then
  TIMEOUT="$TIMEOUT 30"
else
  printf 'warning: timeout command not found\n'
  TIMEOUT=''
fi

recompile_one() {
  args=
  input=$1
  ext=${1##*.}
  if [[ ( $ext = 'll' ) || ( $(basename $input) = test* ) ]]; then
    args="$args -c"
  fi
  extraargs=
  extraargspatt=
  if [[ ( $ext = 'c' ) || ( $ext = 'cpp' ) ]]; then
    extraargspatt='///TAFFO_TEST_ARGS'
  elif [[ $ext = 'll' ]]; then
    extraargspatt=';;;TAFFO_TEST_ARGS'
  fi
  if [[ ! ( -z "$extraargspatt" ) ]]; then
    argstmp=$(grep -E -m 1 "$extraargspatt" $input)
    if [[ ! ( -z "$argstmp" ) ]]; then
      extraargs=${argstmp/$extraargspatt/}
    fi
  fi
  out=${1%.*}.out
  mkdir -p build
  printf '[BUILD] [....] %s' "$input"
  if [[ $FLOAT -eq 1 ]]; then
    args="$args -float-output ${1%.*}.float.out"
  fi
  $TIMEOUT taffo "$args" -o "$out" "$input" $extraargs -debug -temp-dir ./build 2> "$input".log
  bpid_fc=$?
  if [[ $bpid_fc -ne 0 ]]; then
    code='FAIL'
    if [[ $bpid_fc -eq 124 ]]; then
      code='TIME'
    fi
    printf '\033[1G[BUILD] [%s] %s\n' "$code" "$input"
    return 0
  else
    printf '\033[1G[BUILD] [ ok ] %s\n' "$input"
    if [[ $FLOAT -ne 1 ]]; then
      for testin in "$SCRIPTPATH"/input/${1%.*}.*; do
        [ -f "$testin" ] || continue
        printf '[TEST ] [....] %s' $(basename "$testin")
        testout=$(mktemp -t $(basename "$testin")XXX)
        "$out" < "$testin" > "$testout"
        correctout=${SCRIPTPATH}/output/$(basename "$testin")
        logf="$SCRIPTPATH"/$(basename "$testin").log
        diff ${correctout} ${testout} > "$logf"
        if [[ $? -eq 0 ]]; then
          printf '\033[1G[TEST ] [ ok ] %s\n' $(basename "$testin")
          rm "$logf"
        else
          printf '\033[1G[TEST ] [FAIL] %s\n' $(basename "$testin")
          cat "$logf"
        fi
        rm "$testout"
      done
    fi
  fi
  return 0
}

if [[ "$1" == "clean" ]]; then
  rm -f "$SCRIPTPATH"/build/*.taffotmp.*
  rm -f "$SCRIPTPATH"/*.out
  rm -f "$SCRIPTPATH"/*.log
  exit 0
fi

FLOAT=0
if [[ "$1" == "--float" ]]; then
  FLOAT=1
  shift
fi


if [[ "$1" == '--only' ]]; then
  files="$SCRIPTPATH/$2"
else
  files="$SCRIPTPATH/*.c $SCRIPTPATH/*.cpp $SCRIPTPATH/*.ll"
fi

for fn in $files; do
  if [[ ( "$fn" != *.taffotmp.ll ) && ( "$fn" != *NOT-WORKING-YET* ) ]]; then
    recompile_one "$fn" || exit $?
  fi
done

