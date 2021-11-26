# OpenMP Benchmarks
This document outlines the results of benchmarks concerning the OpenMP x TAFFO integration.

## Polybench-ACC
To evaluate the integration's performance, a properly modified version of [Polybench 3.2 with OpenMP pragmas included](https://github.com/cavazos-lab/PolyBench-ACC/tree/master/OpenMP) is used. 
Note that most benchmarks in the original repository have been modified, not only by adding annotations but by fixing syntactical and logical errors in the usage of OpenMP.

Moreover, a not-negligible part of the benchmarks has not only syntactical but also logical errors, when compiling with plain clang. Therefore, they have been excluded.
More information about the modifications and the disabled benchmarks can be found in the specific [benchmark README.md](https://github.com/HEAPLab/TAFFO-test/blob/openmp/polybench-c-openmp/README.md).

### Quantitative results
System specifications:
- CPU: Intel(R) Core(TM) i5-4440 CPU @ 3.10GHz
- Mitigations enabled: itlb_multihit l1tf mds meltdown spec_store_bypass spectre_v1 spectre_v2 srbds
- Memory: 16 GB DDR3 RAM @ 1600Mhz
- OS: Manjaro Linux with kernel 5.10.18-1

The following times have been obtained as the median of 10 executions.

|                 | error_percentage     | error_absolute       | fixed_time | floating_time | speedup           |
|-----------------|----------------------|----------------------|------------|---------------|-------------------|
| covariance      | 9.51816227834747E-05 | 41047797.5653472     | 5.396324   | 5.4954205     | 1.01836370462559  |
| 2mm             | 8.85E-09             | 89916908.3609044     | 12.2622075 | 17.5256945    | 1.42924465272668  |
| 3mm             | 7.45080473244552E-05 | 1092134343786.54     | 4.1437205  | 4.9877275     | 1.20368338067203  |
| doitgen         | 0.00146711757481     | 1.90287227630615     | 0.6280955  | 0.758921      | 1.20828918532293  |
| gemm            | 8.85E-09             | 255.753626909256     | 6.622233   | 9.1119535     | 1.37596389314601  |
| symm            | 5.79050262189691     | 40478069464911.5     | 10.609001  | 12.363222     | 1.16535213824563  |
| syr2k           | 8.85E-09             | 511.513935213089     | 4.4426915  | 6.0616655     | 1.36441287899464  |
| syrk            | 8.85E-09             | 255.753072862625     | 2.1982825  | 3.028159      | 1.37751130712272  |
| floyd-warshall  | 0                    | 0.00E+00             | 35.0197655 | 34.0332085    | 0.971828566356334 |
| reg_detect      | 0.061994358513375    | 0.555555555555556    | 0.0617945  | 0.0627135     | 1.01487187371044  |
| fdtd-2d         | 6.65355579516224E-05 | 0.000154133333333    | 1.8677     | 1.722786      | 0.922410451357284 |
| jacobi-1d-imper | 0.000404369938798    | 2.02043675458368E-06 | 0.0022675  | 0.001722      | 0.759426681367144 |
| jacobi-2d-imper | 0.000156162867696    | 0.00039119           | 0.101391   | 0.091854      | 0.905938396899133 |
| seidel-2d       | 7.85984582155418E-05 | 0.00019689           | 0.3680305  | 0.313082      | 0.850695798310194 |


The benchmarks were executed with `STANDARD_DATASET`, since most benchmarks are currently failing using bigger datasets, without applying TAFFO optimizations. 

In particular, `LARGE_DATASET` and `EXTRALARGE_DATASET` have been tested on the working benchmarks. 
In both cases, every benchmark but the ones specified below failed with Segmentation Fault error. 
The following benchmarks did not fail with `LARGE_DATASET` and their speedup (computed using the same system specifications 
and methodologies outlined above) is coherent with `STANDARD_DATASET`:
- `covariance`
- `syrk`
- `jacobi-1d-imper`
- `jacobi-2d-imper`
- `seidel-2d`

Note that only `jacobi-1d-imper` worked with `EXTRALARGE_DATASET`.

Possible explanation for the lower-than-expected speed-ups are:
- the benchmarks used are based on an outdated version of Polybench with respect to the other one currently used in TAFFO.
- the OpenMP directives in the adapted OpenMP-Polybench are not always correctly used, leading to far different results compared to the serial version.
   In particular, the modified parallel version of Polybench generate numbers with larger order-of-magnitude with respect to the original Polybench suite, 
   requiring more precision in fixed-points and often reaching overflows.

## Other benchmarks
A few other OpenMP benchmarks with floating-points that could be adapted are:
- http://rodinia.cs.virginia.edu/doku.php
- https://gitlab.pnnl.gov/perf-lab-hub/perfect/perfect-suite/-/tree/master/suite
