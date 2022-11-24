LLVM_BIN := $(LLVM_DIR)/bin/
OPT := $(LLVM_BIN)opt
CLANG := $(LLVM_BIN)clang
CLANGXX := $(CLANG)++
LAMP_SIMULATOR := /home/denisovlev/Projects/LAMPSimulator/build/LAMPSimulator/LAMPSimulator.so
TAFFO := $(shell eval which taffo)
TAFFO_PREFIX := $(dir $(TAFFO))/..

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
benchmarks := 2mm 3mm atax bicg doitgen mvt gemm gemver gesummv symm syr2k syrk trmm cholesky durbin gramschmidt lu ludcmp trisolv
#benchmarks := 2mm 3mm
mantissa_sizes := 8 11 16 24
input_sizes := standard
scaling_factors := 1

jobs := $(foreach bench, $(benchmarks), \
			$(foreach mode, $(modes), \
				$(foreach mantissa, $(mantissa_sizes), \
					$(foreach input_size, $(input_sizes), \
						$(foreach scaling, $(scaling_factors), \
							job-$(mode)-$(bench)-$(mantissa)-$(input_size)-$(scaling) )))))

.PHONY: all
all: ${jobs} ; echo $@ Success

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

mode = $(word 1, $(subst -, ,$*))
bench = $(word 2, $(subst -, ,$*))
mantissa = $(word 3, $(subst -, ,$*))
input_size = $(word 4, $(subst -, ,$*))
scaling = $(word 5, $(subst -, ,$*))
bench_src = $(SRC_DIR)/$(bench).c
job_dir = $(BUILD_DIR)/bin/$(mode)/$(bench)/$(mantissa)
job_file_base = $(job_dir)/$(bench)-$(mantissa)-$(input_size)-$(scaling)

.PHONY: ${jobs}
${jobs}: job-%:
	@echo $(TAFFO_PREFIX)
	@#echo mode is $(mode) bench is $(bench) mantissa is $(mantissa) input size is $(input_size) scaling is $(scaling)
	@mkdir -p $(job_dir)
	@#$(call read_stats)
	$(call build_$(mode))
	$(call compile_binary)
	$(call call_binary)

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
    taffo \
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
	taffo \
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

define compile_binary
	$(CLANG) \
		$(job_file_base).out.ll \
		-m32 \
		-o $(job_file_base).out
endef

define call_binary
	$(job_file_base).out 2> $(job_file_base).csv
endef

define read_stats
  python3 ./stats_to_opts.py ./build_stats/$(scaling)/$(bench)/$(bench).csv
endef