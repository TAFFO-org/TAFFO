#!/bin/bash


# $1 benchmark to check
collect()
{
  pushd $1
  ./compile+collect.sh "$RESULTS_DIR/axbench"
  popd > /dev/null
}


main()
{
  if [[ -z $1 ]]; then
    RESULTS_DIR="./results"
  else
    RESULTS_DIR="$1"
  fi
  
  if [[ -e "$RESULTS_DIR" ]]; then
    echo 'results dir already exists; archive or remove first please'
    return;
  fi

  rm -r ./raw-times
  mkdir -p "$RESULTS_DIR"
  mkdir -p "$RESULTS_DIR/axbench"
  
  RESULTS_DIR=$(cd ${RESULTS_DIR} 2> /dev/null && pwd -P)

  for d in *; do
    if [[ ( -d $d ) && ( $d != 'common' ) ]]; then
      collect $d &
    fi
  done
  wait

  #./chkval_all_better.py 1 > "$RESULTS_DIR/axbench.txt"
  hostname > "$RESULTS_DIR/MACHINE.txt"
}


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
main "$1" & wait

