seed=0
vecLen=10000000
warmup_iterations=5
clang_results=results_expected.txt
taffo_results=results_taffo.txt
comparison_results=results.txt

make clang
make taffo
./axpy_clang.exe ${seed} ${vecLen} ${warmup_iterations} ${clang_results}
./axpy_taffo.exe ${seed} ${vecLen} ${warmup_iterations} ${taffo_results}
../common/check_performance.exe ../axpy/${taffo_results} ../axpy/${clang_results} ../axpy/${comparison_results}
