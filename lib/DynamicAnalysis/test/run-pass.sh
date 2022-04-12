export LLVM_DIR=/usr/lib/llvm-12/bin/
# Generate an LLVM file to analyze
"$LLVM_DIR"clang -O0 -emit-llvm -S ./input.c -o input.ll
# Run the pass through opt - New PM
"$LLVM_DIR"opt -S -load ../../../build/lib/Taffo.so --taffoinit input.ll -o taffoinit.ll
#"$LLVM_DIR"opt -S -load-pass-plugin ../../../build/lib/Taffo.so --passes="name-variables" taffoinit.ll -o input_named.ll
#"$LLVM_DIR"opt -S -load-pass-plugin ../../../build/lib/Taffo.so --passes="inject-func-call" input_named.ll -o instrumented.ll
"$LLVM_DIR"opt -S -load ../../../build/lib/Taffo.so --taffo-name-variables taffoinit.ll -o input_named.ll
"$LLVM_DIR"opt -S -load ../../../build/lib/Taffo.so --taffo-inject-func-call input_named.ll -o instrumented.ll

"$LLVM_DIR"llc -filetype=obj instrumented.ll -o instrumented.o

"$LLVM_DIR"clang instrumented.o -o instrumented

./instrumented > trace.log
./instrumented > trace2.log

#"$LLVM_DIR"opt -S -load-pass-plugin ../../../build/lib/Taffo.so --passes="read-trace" instrumented.ll -o annotated.ll
#"$LLVM_DIR"opt -S -load ../../../build/lib/Taffo.so -load-pass-plugin ../../../build/lib/Taffo.so --passes="read-trace" -trace_file trace.log -trace_file trace2.log input_named.ll -o annotated.ll
"$LLVM_DIR"opt -S -load ../../../build/lib/Taffo.so --taffo-read-trace \
  -trace_file trace.log -trace_file trace2.log input_named.ll -o annotated.ll

"$LLVM_DIR"opt -S -load ../../../build/lib/Taffo.so --taffodta annotated.ll -o taffodta.ll
