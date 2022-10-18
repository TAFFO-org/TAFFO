export LLVM_DIR=/home/denisovlev/Data/llvm-12-debug/bin/
TAFFO_PREFIX=$(dirname $(which taffo))/..
LAMP_SIMULATOR=/home/denisovlev/Projects/LAMPSimulator/build/LAMPSimulator/LAMPSimulator.so

export MANTISSA=''

for arg; do
  case $arg in
    -mantissa=* | -cvt-mant=* | -add-mant=* | -sub-mant=* | -mul-mant=* | -div-mant=*)
      export MANTISSA="$MANTISSA $arg"
      ;;
    *)
      echo Unrecognized option $arg
      exit 1
  esac
done

"$LLVM_DIR"opt \
  -S \
  -debug \
  -load "$LAMP_SIMULATOR" \
  -lampsim \
  $MANTISSA \
  /home/denisovlev/Projects/TAFFO/test/polybench-riscv/build/4/cholesky/cholesky.out.ll.5.taffotmp.ll \
  -o /home/denisovlev/Projects/TAFFO/test/polybench-riscv/build/4/cholesky/cholesky.out.lamp.ll
