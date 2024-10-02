seed=0
m=10000
n=10000
warmup_iterations=5
clang_results=results_expected.txt
taffo_results=results_taffo.txt
comparison_results=results.txt

make clang
make taffo
./gemv_clang.exe ${seed} ${m} ${n} ${warmup_iterations} ${clang_results}
./gemv_taffo.exe ${seed} ${m} ${n} ${warmup_iterations} ${taffo_results}
../common/check_performance.exe ../gemv/${taffo_results} ../gemv/${clang_results} ../gemv/${comparison_results}
