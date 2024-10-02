seed=0
m=1000
k=1000
n=1000
warmup_iterations=5
clang_results=results_expected.txt
taffo_results=results_taffo.txt
comparison_results=results.txt

make clang
make taffo
./gemm_clang.exe ${seed} ${m} ${k} ${n} ${warmup_iterations} ${clang_results}
./gemm_taffo.exe ${seed} ${m} ${k} ${n} ${warmup_iterations} ${taffo_results}
../common/check_performance.exe ../gemm/${taffo_results} ../gemm/${clang_results} ../gemm/${comparison_results}
