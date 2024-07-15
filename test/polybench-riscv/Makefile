LLVM_BIN := $(LLVM_DIR)/bin/
OPT := $(LLVM_BIN)opt
CLANG := $(LLVM_BIN)clang
CLANGXX := $(CLANG)++
#LAMP_SIMULATOR_DIR := /home/denisovlev/Projects/LAMPSimulator/build/
LAMP_SIMULATOR := $(LAMP_SIMULATOR_DIR)/LAMPSimulator/LAMPSimulator.so
TAFFO := $(shell eval which taffo)
TAFFO_PREFIX := $(dir $(TAFFO))/..
TAFFO_INSTMIX := $(shell eval which taffo-instmix)
DEBUG := 0
ifeq ($(DEBUG), 1)
	DEBUG_TAFFO := -debug-taffo
else
	DEBUG_TAFFO :=
endif

BUILD_DIR := ./build
SRC_DIR := ./sources
HEADERS_DIR := ./sources/.

CFLAGS := -g0 -Xclang -disable-O0-optnone -fno-unroll-loops -fno-slp-vectorize -fno-vectorize -ffp-contract=off -m32 -march=i386
CFLAGS_DOUBLE := -g0 -Xclang -disable-O0-optnone -fno-unroll-loops -fno-slp-vectorize -fno-vectorize -ffp-contract=off
POLYBENCH_PRINT_FLAGS := -D_PRINT_OUTPUT -DPOLYBENCH_DUMP_ARRAYS
POLYBENCH_DATA_FLAGS := -DDATA_TYPE_IS_FLOAT
POLYBENCH_DOUBLE_DATA_FLAGS := -DDATA_TYPE_IS_DOUBLE

errorprop :=
costmodel := soc_im_zm
instrset := soc_zoni
instrset_file := $(TAFFO_PREFIX)/share/ILP/constrain/$(instrset)
enobweight := 100
timeweight := 1000
castweight := 100

#modes := float fixed mixed
modes := float fixed
#benchmarks := 2mm 3mm atax bicg doitgen mvt gemm gemver gesummv symm syr2k syrk trmm lu ludcmp
benchmarks := adi correlation covariance heat-3d jacobi-1d jacobi-2d seidel-2d
#benchmarks := 2mm 3mm atax bicg doitgen mvt gemm gemver gesummv symm syr2k syrk trmm cholesky durbin gramschmidt lu ludcmp trisolv
#benchmarks := 2mm 3mm atax bicg gemm gemver gesummv symm syr2k syrk trmm lu ludcmp
#benchmarks := 2mm
mantissa_sizes := 8 11 16 24
input_sizes := standard
scaling_factors := 1 2 4 8 16 32 64 128 256 512 1024

double_jobs := $(foreach bench, $(benchmarks), \
				$(foreach input_size, $(input_sizes), \
					$(foreach scaling, $(scaling_factors), \
						double-job-$(bench)--$(input_size)--$(scaling) )))

stats_jobs := $(foreach bench, $(benchmarks), \
				$(foreach input_size, $(input_sizes), \
					$(foreach scaling, $(scaling_factors), \
						stats-job-$(bench)--$(input_size)--$(scaling) )))

jobs := $(foreach bench, $(benchmarks), \
			$(foreach mode, $(modes), \
				$(foreach mantissa, $(mantissa_sizes), \
					$(foreach input_size, $(input_sizes), \
						$(foreach scaling, $(scaling_factors), \
							job-$(bench)--$(input_size)--$(scaling)--$(mode)--$(mantissa) )))))

bench = $(word 1, $(subst --, ,$*))
input_size = $(word 2, $(subst --, ,$*))
scaling = $(word 3, $(subst --, ,$*))
mode = $(word 4, $(subst --, ,$*))
mantissa = $(word 5, $(subst --, ,$*))
bench_src = $(SRC_DIR)/$(bench).c
job_dir = $(BUILD_DIR)/bin/$(mode)/$(bench)/$(mantissa)
job_file_base = $(job_dir)/$(bench)-$(input_size)-$(scaling)-$(mode)-$(mantissa)
stats_job_dir = $(BUILD_DIR)/stats/$(bench)/$(scaling)
double_job_dir = $(BUILD_DIR)/bin/double/$(bench)/$(scaling)
stats_job_file_base = $(stats_job_dir)/$(bench)-$(input_size)-$(scaling)
double_job_file_base = $(double_job_dir)/$(bench)-$(input_size)-$(scaling)
summary_dir = $(BUILD_DIR)/summary
configurations_file = $(BUILD_DIR)/configurations.csv
double_configurations_file = $(BUILD_DIR)/double_configurations.csv

.PHONY: all
all: $(clean_configurations) ${double_jobs} ${stats_jobs} ${jobs} summary ; echo $@ Success

.PHONY: stats
stats: ${stats_jobs} ; echo $@ Success

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

.PHONY: clean_configurations
clean_configurations:
	rm -r $(configurations_file)
	rm -r $(double_job_file_base)

.PHONY: ${double_jobs}
${double_jobs}: double-job-%:
	@mkdir -p $(double_job_dir)
	@echo $(bench),$(input_size),$(scaling),$(double_job_file_base) \
			>> $(double_configurations_file)
	$(call double_build)
	$(call double_build_run)

.PHONY: ${stats_jobs}
${stats_jobs}: stats-job-%:
	@mkdir -p $(stats_job_dir)
	$(call stats_build)
	$(call stats_build_run)
	$(call stats_build_tracing)
	$(call stats_build_tracing_run)

.PHONY: ${jobs}
${jobs}: job-%:
	@echo $(bench),$(input_size),$(scaling),$(mode),$(mantissa),$(job_file_base),$(stats_job_file_base) \
 		>> $(configurations_file)
	@mkdir -p $(job_dir)
	$(call build_$(mode))
	$(call truncated_precision)
	$(call compile_binary)
	$(call compile_truncated_binary)
	$(call call_binary)
	$(call call_truncated_binary)
	$(call run_instmix)

.PHONY: summary
summary:
	@mkdir -p $(summary_dir)
	@python3 ./analyze.py $(BUILD_DIR)

define double_build
    @echo double $(bench)
    $(CLANG) \
        -o $(double_job_file_base).out \
        -lm \
        $(bench_src) \
        -I$(HEADERS_DIR) \
        $(CFLAGS_DOUBLE) \
        $(POLYBENCH_PRINT_FLAGS) \
        $(POLYBENCH_DOUBLE_DATA_FLAGS) \
        -DSCALING_FACTOR=$(scaling) \
        2> $(double_job_file_base).log
endef

define build_float
    @echo float $(bench) $(mantissa)
    export VAR_RANGES="$$($(call read_stats))"; \
    $(CLANG) \
        -o $(job_file_base).out.ll \
        -S \
        -emit-llvm \
        -lm \
        $(bench_src) \
        -I$(HEADERS_DIR) \
        $$VAR_RANGES \
        $(CFLAGS) \
        $(POLYBENCH_PRINT_FLAGS) \
        $(POLYBENCH_DATA_FLAGS) \
        -DSCALING_FACTOR=$(scaling) \
        2> $(job_file_base).log
endef

define build_fixed
    @echo fixed $(bench) $(mantissa)
    export VAR_RANGES="$$($(call read_stats))"; \
    $(TAFFO) \
		-o $(job_file_base).out.ll \
		-emit-llvm \
		-temp-dir $(job_dir) \
		-lm \
		-time-profile-file $(job_file_base).compile_time.csv \
		$(DEBUG_TAFFO) \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		$$VAR_RANGES \
		$(CFLAGS) \
		$(POLYBENCH_PRINT_FLAGS) \
		$(POLYBENCH_DATA_FLAGS) \
		-DSCALING_FACTOR=$(scaling) \
		2> $(job_file_base).log
endef

define build_mixed
    @echo mixed $(bench) $(mantissa)
    export VAR_RANGES="$$($(call read_stats))"; \
	$(TAFFO) \
		-o $(job_file_base).out.ll \
		-emit-llvm \
		-temp-dir $(job_dir) \
		-lm \
		-time-profile-file $(job_file_base).compile_time.csv \
		$(DEBUG_TAFFO) \
		-mixedmode \
		-costmodel $(costmodel) \
		-instructionsetfile=$(instrset_file) \
		-Xdta -mixedtuningenob -Xdta $(enobweight) \
		-Xdta -mixedtuningtime -Xdta $(timeweight) \
		-Xdta -mixedtuningcastingtime -Xdta $(castweight) \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		$$VAR_RANGES \
		$(CFLAGS) \
		$(POLYBENCH_PRINT_FLAGS) \
		$(POLYBENCH_DATA_FLAGS) \
		-DSCALING_FACTOR=$(scaling) \
		2> $(job_file_base).log
endef

define stats_build
	$(CLANG) \
		-o $(stats_job_file_base).out \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		$(CFLAGS_DOUBLE) \
		$(POLYBENCH_DATA_FLAGS) \
		-DCOLLECT_STATS \
		-DSCALING_FACTOR=$(scaling) \
		2> $(stats_job_file_base).log
endef

define double_build_run
	@$(double_job_file_base).out 2> $(double_job_file_base).csv
endef

define stats_build_run
	@$(stats_job_file_base).out 2> $(stats_job_file_base).csv
endef

define stats_build_tracing
	$(TAFFO) \
	 	-temp-dir $(stats_job_dir) \
		-o $(stats_job_file_base).out.dynamic_instrumented \
		-O0 -disable-O0-optnone \
		-lm \
		$(DEBUG_TAFFO) \
		-dynamic-instrument \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		$(CFLAGS_DOUBLE) \
		$(POLYBENCH_DATA_FLAGS) \
		-DSCALING_FACTOR=$(scaling) \
		2> $(stats_job_file_base).dynamic_instrumented.log
endef

define stats_build_tracing_run
	@$(stats_job_file_base).out.dynamic_instrumented > $(stats_job_file_base).instrumented.trace
endef

define truncated_precision
	$(OPT) \
	-S \
	-debug \
	-load $(LAMP_SIMULATOR) \
	--load-pass-plugin=$(LAMP_SIMULATOR) \
	--passes="no-op-function,lampsim" \
	-mantissa=$(mantissa) \
	$(job_file_base).out.ll \
	-o $(job_file_base).lamp.out.ll
endef

define compile_binary
	$(CLANG) \
		$(job_file_base).out.ll \
		-m32 \
		-o $(job_file_base).out
endef

define compile_truncated_binary
	$(CLANG) \
		$(job_file_base).lamp.out.ll \
		-m32 \
		-o $(job_file_base).lamp.out
endef

define run_instmix
	-$(TAFFO_INSTMIX) $(job_file_base).out.ll \
		1> $(job_file_base).mix.txt \
		2> $(job_file_base).mix.log.txt
endef

define call_binary
	-$(job_file_base).out 2> $(job_file_base).csv
endef

define call_truncated_binary
	-$(job_file_base).lamp.out 2> $(job_file_base).lamp.csv
endef

define read_stats
  python3 ./stats_to_opts.py $(stats_job_file_base).csv
endef