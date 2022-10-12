export LLVM_DIR=export LLVM_DIR=/home/denisovlev/Data/llvm-12-debug/bin/
TAFFO_PREFIX=$(dirname $(which taffo))/..

"$LLVM_DIR"opt \
  -S \
  -load "$TAFFO_PREFIX"/lib/Taffo.so \
  --taffo-float-size-analysis \
  -stats_output_file stats.csv\
  /home/denisovlev/Projects/TAFFO/test/polybench-riscv/build_mixed/4/cholesky/cholesky.out.ll.5.taffotmp.ll \
  -o /dev/null
