LLVM_BIN := $(LLVM_DIR)/bin/
OPT := $(LLVM_BIN)opt
LLC := $(LLVM_BIN)llc
CLANG := $(LLVM_BIN)clang
CLANGXX := $(CLANG)++
CC :=
TAFFO := $(shell eval which taffo)
TAFFO_PREFIX := $(dir $(TAFFO))/..
TAFFO_INSTMIX := $(shell eval which taffo-instmix)
DEBUG := 0
ifeq ($(DEBUG), 1)
	DEBUG_TAFFO := -debug-taffo
else
	DEBUG_TAFFO :=
endif

BUILD_DIR := $(abspath ./build)
SRC_DIR := $(abspath ./src)
EMBEDDED_SRC_DIR := $(abspath ./embedded_src)
HEADERS_DIR := $(abspath ./utilities/.)

TARGET='stm32l101'
embedded_sysroot=/lib/arm-none-eabi
embedded_triple=arm-none-eabi
embedded_cpu=cortex-m0plus+nofp

CFLAGS := -g0 -Xclang -disable-O0-optnone -lm -fno-unroll-loops -fno-slp-vectorize -fno-vectorize
CFLAGS_DOUBLE := -g0 -O3 -lm
POLYBENCH_SHARED_FLAGS :=  -DPOLYBENCH_STACK_ARRAYS
POLYBENCH_PC_FLAGS :=
POLYBENCH_EMBEDDED_FLAGS :=
POLYBENCH_BENCH_FLAGS := -DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_TIME -DBENCH_NUM_ITERATIONS=10
POLYBENCH_DATASET_FLAGS := -DMINI_DATASET
#POLYBENCH_DATASET_FLAGS := -DSMALL_DATASET
POLYBENCH_DATA_FLAGS :=  -DDATA_TYPE_IS_FLOAT
POLYBENCH_DOUBLE_DATA_FLAGS := -DDATA_TYPE_IS_DOUBLE
PC_OPT_FLAGS :=
PC_FLAGS :=
EMBEDDED_OPT_FLAGS := -m32 \
						--target=$(embedded_triple) \
						-mcpu=$(embedded_cpu) \
						--sysroot=$(embedded_sysroot) \
						-fshort-enums

modes := float fixed dynamic
archs := PC EMBEDDED
benchmarks := 2mm 3mm atax bicg doitgen mvt gemm gemver gesummv symm syr2k syrk trmm cholesky durbin gramschmidt lu ludcmp trisolv
#benchmarks := 2mm 3mm atax bicg gemm gemver gesummv symm syr2k syrk trmm lu ludcmp
#benchmarks := 2mm

double_jobs := $(foreach bench, $(benchmarks), \
				double-job-$(bench) )

stats_jobs := $(foreach bench, $(benchmarks), \
				stats-job-$(bench) )

jobs := $(foreach bench, $(benchmarks), \
			$(foreach mode, $(modes), \
				$(foreach arch, $(archs), \
					job-$(bench)-$(mode)-$(arch) )))

bench = $(word 1, $(subst -, ,$*))
mode = $(word 2, $(subst -, ,$*))
arch = $(word 3, $(subst -, ,$*))

bench_src = $(SRC_DIR)/$(bench)/$(bench).c
bench_src_h = $(SRC_DIR)/$(bench)/$(bench).h
job_dir = $(BUILD_DIR)/bin/$(arch)/$(mode)/$(bench)
job_file_base = $(job_dir)/$(bench)-$(mode)
stats_job_dir = $(BUILD_DIR)/stats/$(bench)
double_job_dir = $(BUILD_DIR)/bin/double/$(bench)
stats_job_file_base = $(stats_job_dir)/$(bench)
double_job_file_base = $(double_job_dir)/$(bench)
summary_dir = $(BUILD_DIR)/summary
deps_dir = $(BUILD_DIR)/dependencies
deps_embedded_dir = $(deps_dir)/embedded
deps_pc_dir = $(deps_dir)/pc
configurations_file = $(BUILD_DIR)/configurations.csv
double_configurations_file = $(BUILD_DIR)/double_configurations.csv

.PHONY: all
#all: $(clean_configurations) ${double_jobs} ${stats_jobs} ${jobs} summary ; echo $@ Success
all: $(clean_configurations) build_deps ${double_jobs} ${stats_jobs} ${jobs} ; echo $@ Success

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

.PHONY: clean_configurations
clean_configurations:
	rm -r $(configurations_file)
	rm -r $(double_job_file_base)

.PHONY: build_deps
build_deps:
	@mkdir -p $(deps_dir)
	@mkdir -p $(deps_embedded_dir)
	@mkdir -p $(deps_pc_dir)
	$(call build_pc_timer)

.PHONY: ${double_jobs}
${double_jobs}: double-job-%:
	@mkdir -p $(double_job_dir)
	@echo $(bench),$(double_job_file_base) \
			>> $(double_configurations_file)
	$(call double_build)
	$(call call_binary_double)

.PHONY: ${stats_jobs}
${stats_jobs}: stats-job-%:
	@mkdir -p $(stats_job_dir)
	$(call stats_build_tracing)
	$(call stats_build_tracing_run)
	$(call stats_build_tracing_compress)

.PHONY: ${jobs}
${jobs}: job-%:
	@echo $(bench),$(arch),$(mode),$(job_file_base),$(stats_job_file_base) \
 		>> $(configurations_file)
	@mkdir -p $(job_dir)
	$(call build_$(mode))
	$(call build_object)
	$(call compile_binary_$(arch))
	$(call call_binary_$(arch))
	$(call run_instmix)

.PHONY: summary
summary:
	@mkdir -p $(summary_dir)
	@python3 ./analyze.py $(BUILD_DIR)

define build_pc_timer
    $(CLANGXX) \
		-o $(deps_pc_dir)/timer.so \
		-c \
		$(HEADERS_DIR)/timer.cpp  \
		-I$(HEADERS_DIR) \
		$(CFLAGS_DOUBLE) \
		$(POLYBENCH_SHARED_FLAGS) \
		$(POLYBENCH_PC_FLAGS) \
		$(POLYBENCH_BENCH_FLAGS) \
		$(POLYBENCH_DOUBLE_DATA_FLAGS) \
		$(POLYBENCH_DATASET_FLAGS) \
		2> $(deps_pc_dir)/timer.log
endef

define double_build
    @echo double $(bench)
    $(CLANG) \
        -o $(double_job_file_base).out \
        $(bench_src)  \
        $(deps_pc_dir)/timer.so \
        -I$(HEADERS_DIR) \
        -I$(bench_src_h) \
        $(CFLAGS_DOUBLE) \
        $(POLYBENCH_SHARED_FLAGS) \
		$(POLYBENCH_$(arch)_FLAGS) \
        $(POLYBENCH_BENCH_FLAGS) \
        $(POLYBENCH_DOUBLE_DATA_FLAGS) \
        $(POLYBENCH_DATASET_FLAGS) \
        2> $(double_job_file_base).log
endef

define build_float
    @echo float $(bench) $(arch)
    $(CLANG) \
        -o $(job_file_base).out.ll \
        -S -emit-llvm \
        $(bench_src) \
        -I$(HEADERS_DIR) \
        -I$(bench_src_h) \
        $($(arch)_OPT_FLAGS) \
        $(CFLAGS) \
        $(POLYBENCH_SHARED_FLAGS) \
		$(POLYBENCH_$(arch)_FLAGS) \
        $(POLYBENCH_BENCH_FLAGS) \
        $(POLYBENCH_DATA_FLAGS) \
        $(POLYBENCH_DATASET_FLAGS) \
        2> $(job_file_base).log
endef

define build_fixed
    @echo fixed $(bench) $(arch)
    $(TAFFO) \
		-o $(job_file_base).out.ll \
		-emit-llvm \
		-temp-dir $(job_dir) \
		$(DEBUG_TAFFO) \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		-I$(bench_src_h) \
		$($(arch)_OPT_FLAGS) \
		$(CFLAGS) \
		$(POLYBENCH_SHARED_FLAGS) \
		$(POLYBENCH_$(arch)_FLAGS) \
		$(POLYBENCH_BENCH_FLAGS) \
		$(POLYBENCH_DATA_FLAGS) \
		$(POLYBENCH_DATASET_FLAGS) \
		2> $(job_file_base).log
endef

define build_dynamic
    @echo mixed $(bench) $(arch)
	$(TAFFO) \
		-o $(job_file_base).out.ll \
		-emit-llvm \
		-temp-dir $(job_dir) \
		-dynamic-trace $(stats_job_file_base).instrumented.trace \
		$(DEBUG_TAFFO) \
		$(bench_src) \
		-I$(HEADERS_DIR) \
		-I$(bench_src_h) \
		$($(arch)_OPT_FLAGS) \
		$(CFLAGS) \
		$(POLYBENCH_SHARED_FLAGS) \
		$(POLYBENCH_$(arch)_FLAGS) \
		$(POLYBENCH_BENCH_FLAGS) \
		$(POLYBENCH_DATA_FLAGS) \
		$(POLYBENCH_DATASET_FLAGS) \
		2> $(job_file_base).log
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
		-I$(bench_src_h) \
		$(POLYBENCH_SHARED_FLAGS) \
		$(POLYBENCH_DATA_FLAGS) \
		$(POLYBENCH_DATASET_FLAGS) \
		2> $(stats_job_file_base).dynamic_instrumented.log
endef

define stats_build_tracing_run
	-@$(stats_job_file_base).out.dynamic_instrumented > $(stats_job_file_base).instrumented.trace 2> /dev/null
endef

define stats_build_tracing_compress
	@python3 ./compress_trace.py $(stats_job_file_base).instrumented.trace $(stats_job_file_base).compressed.trace
	@mv -f $(stats_job_file_base).compressed.trace $(stats_job_file_base).instrumented.trace
endef

define build_object
	$(LLC) \
		$(job_file_base).out.ll \
		-filetype=obj \
		-o $(job_file_base).o
endef

define compile_binary_PC
	$(CLANG) \
		-no-pie -lm \
		$(job_file_base).o \
		$(deps_pc_dir)/timer.so \
		-o $(job_file_base).out.bin
endef

define compile_binary_EMBEDDED
	cd $(EMBEDDED_SRC_DIR) && \
	$(MAKE) \
		OBJ_DIR=$(deps_embedded_dir) \
		BENCH_OBJ=$(job_file_base).o \
		OUTPUT=$(job_file_base).out \
		TARGET=$(TARGET)
endef

define run_instmix
	-$(TAFFO_INSTMIX) $(job_file_base).out.ll \
		1> $(job_file_base).mix.txt \
		2> $(job_file_base).mix.log.txt
endef

define call_binary_double
	-@$(double_job_file_base).out > $(double_job_file_base).time.txt 2> $(double_job_file_base).csv
endef

define call_binary_PC
	-$(job_file_base).out.bin > $(job_file_base).time.txt 2> $(job_file_base).csv
endef

define call_binary_EMBEDDED
#	-$(job_file_base).out.bin > $(job_file_base).time.txt 2> $(job_file_base).csv
endef

define read_stats
  python3 ./stats_to_opts.py $(stats_job_file_base).csv
endef