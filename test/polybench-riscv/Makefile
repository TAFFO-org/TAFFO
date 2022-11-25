LLVM_BIN := $(LLVM_DIR)/bin/
OPT := $(LLVM_BIN)opt
CLANG := $(LLVM_BIN)clang
CLANGXX := $(CLANG)++
LAMP_SIMULATOR_DIR := /home/denisovlev/Projects/LAMPSimulator/build/
LAMP_SIMULATOR := $(LAMP_SIMULATOR_DIR)/LAMPSimulator/LAMPSimulator.so
TAFFO := $(shell eval which taffo)
TAFFO_PREFIX := $(dir $(TAFFO))/..
TAFFO_INSTMIX := $(shell eval which taffo-instmix)

BUILD_DIR := ./build
SRC_DIR := ./sources
HEADERS_DIR := ./sources/.

CFLAGS := -g -Xclang -disable-O0-optnone -fno-unroll-loops -fno-slp-vectorize -fno-vectorize \
			-D_PRINT_OUTPUT -DPOLYBENCH_DUMP_ARRAYS
errorprop :=
costmodel := soc_im_zm
instrset := soc_zoni
enobweight := 1000
timeweight := 1000
castweight := 1

modes := float fixed mixed
#benchmarks := 2mm 3mm atax bicg doitgen mvt gemm gemver gesummv symm syr2k syrk trmm cholesky durbin gramschmidt lu ludcmp trisolv
benchmarks := 2mm
mantissa_sizes := 8 11 16 24
input_sizes := standard
scaling_factors := 1

stats_jobs := $(foreach bench, $(benchmarks), \
				$(foreach input_size, $(input_sizes), \
					$(foreach scaling, $(scaling_factors), \
						stats-job-$(bench)-$(input_size)-$(scaling) )))

jobs := $(foreach bench, $(benchmarks), \
			$(foreach mode, $(modes), \
				$(foreach mantissa, $(mantissa_sizes), \
					$(foreach input_size, $(input_sizes), \
						$(foreach scaling, $(scaling_factors), \
							job-$(bench)-$(input_size)-$(scaling)-$(mode)-$(mantissa) )))))

.PHONY: all
all: ${stats_jobs} ${jobs} ; echo $@ Success

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

bench = $(word 1, $(subst -, ,$*))
input_size = $(word 2, $(subst -, ,$*))
scaling = $(word 3, $(subst -, ,$*))
mode = $(word 4, $(subst -, ,$*))
mantissa = $(word 5, $(subst -, ,$*))
bench_src = $(SRC_DIR)/$(bench).c
job_dir = $(BUILD_DIR)/bin/$(mode)/$(bench)/$(mantissa)
job_file_base = $(job_dir)/$(bench)-$(input_size)-$(scaling)-$(mode)-$(mantissa)

stats_job_dir = $(BUILD_DIR)/stats/$(bench)/$(scaling)
stats_job_file_base = $(stats_job_dir)/$(bench)-$(input_size)-$(scaling)
.PHONY: ${stats_jobs}
${stats_jobs}: stats-job-%:
	@mkdir -p $(stats_job_dir)
	$(call stats_build)
	$(call stats_build_run)
	$(call stats_build_tracing)
	$(call stats_build_tracing_run)

.PHONY: ${jobs}
${jobs}: job-%:
	@#echo mode is $(mode) bench is $(bench) mantissa is $(mantissa) input size is $(input_size) scaling is $(scaling)
	@mkdir -p $(job_dir)
	@#$(call read_stats)
	$(call build_$(mode))
	$(call truncated_precision)
	$(call compile_binary)
	$(call compile_truncated_binary)
	$(call call_binary)
	$(call call_truncated_binary)
	$(call run_instmix)

define build_float
    @echo float $(bench) $(mantissa)
    export VAR_RANGES=$$($(call read_stats)); \
    $(CLANG) \
        -o $(job_file_base).out.ll \
        -S \
        -emit-llvm \
        -lm \
        $(bench_src) \
        -I$(HEADERS_DIR) \
        $$VAR_RANGES \
        $(CFLAGS) \
        -DSCALING_FACTOR=$(scaling) \
        2> $(job_file_base).log
endef

define build_fixed
    @echo fixed $(bench) $(mantissa)
    export VAR_RANGES=$$($(call read_stats)); \
    $(TAFFO) \
		-o $(job_file_base).out.ll \
		-emit-llvm \
		-temp-dir $(job_dir) \
		-lm \
		-debug-taffo \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		$$VAR_RANGES \
		$(CFLAGS) \
		-DSCALING_FACTOR=$(scaling) \
		2> $(job_file_base).log
endef

define build_mixed
    @echo mixed $(bench) $(mantissa)
    export VAR_RANGES=$$($(call read_stats)); \
	$(TAFFO) \
		-o $(job_file_base).out.ll \
		-emit-llvm \
		-temp-dir $(job_dir) \
		-lm \
		-debug-taffo \
		-mixedmode \
		-costmodel $(costmodel) \
		-instructionsetfile=$(TAFFO_PREFIX)/share/ILP/constrain/$(instrset) \
		-Xdta -mixedtuningenob -Xdta $(enobweight) \
		-Xdta -mixedtuningtime -Xdta $(timeweight) \
		-Xdta -mixedtuningcastingtime -Xdta $(castweight) \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		$$VAR_RANGES \
		$(CFLAGS) \
		-DSCALING_FACTOR=$(scaling) \
		2> $(job_file_base).log
endef

define stats_build
	$(CLANG) \
		-o $(stats_job_file_base).out \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		-DCOLLECT_STATS \
		-DSCALING_FACTOR=$(scaling) \
		2> $(stats_job_file_base).log
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
		-dynamic-instrument \
		$(bench_src) \
		-I$(HEADERS_DIR) \
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
	$(TAFFO_INSTMIX) $(job_file_base).out.ll \
		1> $(job_file_base).mix.txt \
		2> $(job_file_base).mix.log.txt
endef

define call_binary
	$(job_file_base).out 2> $(job_file_base).csv
endef

define call_truncated_binary
	$(job_file_base).lamp.out 2> $(job_file_base).lamp.csv
endef

define read_stats
  python3 ./stats_to_opts.py $(stats_job_file_base).csv
endef