dynamic_analysis() {
    taffo -temp-dir obj \
      -o bin/${bench}.out.dynamic_instrumented \
      -O0 -disable-O0-optnone \
      -dynamic-instrument \
      -I../common/src \
      ${files} \
      ${debug} \
      ${errorprop} \
      ${no_mem2reg} \
      -lm \
      2> stats/taffo1.log

    "bin/${bench}.out.dynamic_instrumented" \
    /home/denisovlev/Projects/TAFFO/test/axbench/blackscholes/data/input/blackscholesTrain_100K.data \
    /dev/null \
     > "obj/${bench}.out.instrumented.trace"

    taffo -temp-dir obj \
      -o bin/${bench}.out.dynamic_final \
      -O3 \
      -dynamic-trace "obj/${bench}.out.instrumented.trace" \
      -I../common/src \
      ${files} \
      ${debug} \
      ${errorprop} \
      ${no_mem2reg} \
      -lm \
      2> stats/taffo2.log
  }

  dynamic_analysis