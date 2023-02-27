### Usage

```shell
export LLVM_DIR=/.../llvm-15-debug/bin/
export TAFFOLIB=/.../lib/Taffo.so
"$LLVM_DIR"opt -S -load "$TAFFOLIB" --load-pass-plugin="$TAFFOLIB" \
                    --debug \
                    --passes='no-op-module,tafforangeexport' \
                    -ranges_file ranges.csv \
                    /.../blackscholes.out.fixp.2.taffotmp.ll \
                    -o /dev/null
```