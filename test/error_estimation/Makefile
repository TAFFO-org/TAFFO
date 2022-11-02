LLVM_BIN := $(LLVM_DIR)/bin/
OPT := $(LLVM_BIN)opt
CLANG := $(LLVM_BIN)clang
CLANGXX := $(CLANG)++
LAMP_SIMULATOR := /home/denisovlev/Projects/LAMPSimulator/build/LAMPSimulator/LAMPSimulator.so
CFLAGS := -g -Xclang -disable-O0-optnone -fno-unroll-loops -fno-slp-vectorize -fno-vectorize

BUILD_DIR := ./build
SRC_DIRS := ./src
BENCHMARK := average
BENCHMARK_LL := $(BUILD_DIR)/$(BENCHMARK).0.ll

MANTISSA := 8 11 16 24
INPUT_SIZE := 1 10 100 1000 10000 100000 1000000
MAX_INPUT_SIZE := 1000000
PRECISION_OUT_LL = $(BUILD_DIR)/$(BENCHMARK)-m$@bit.ll
PRECISION_OUT = $(BUILD_DIR)/$(BENCHMARK)-m$@bit
RESULT = $(BUILD_DIR)/$(BENCHMARK)-m$@bit-input$(input_size)rows.txt


all: $(BUILD_DIR) $(BUILD_DIR)/data.txt $(BENCHMARK_LL) $(MANTISSA) aggregate-all
.PHONY: all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/data.txt: $(SRC_DIRS)/generate_data.py
	python3 $(SRC_DIRS)/generate_data.py $(MAX_INPUT_SIZE) > $(BUILD_DIR)/data.txt

$(BENCHMARK_LL): $(SRC_DIRS)/$(BENCHMARK).c
	$(CLANG) \
		-o $@ \
		-S \
		-emit-llvm \
		--debug \
		-m32 \
		$(CFLAGS) \
		-I$(SRC_DIRS)/. \
		$(SRC_DIRS)/$(BENCHMARK).c \
		2> $@.log

define execute-command
$(1)

endef

.PHONY: $(MANTISSA)
$(MANTISSA): $(BENCHMARK_LL)
	$(OPT) \
		-S \
		--debug \
		-load $(LAMP_SIMULATOR) \
		-lampsim \
		-mantissa=$@ \
		$(BENCHMARK_LL) \
		-o $(PRECISION_OUT_LL)

	$(CLANG) \
      $(PRECISION_OUT_LL) \
      -m32 \
      -o $(PRECISION_OUT)

	$(foreach input_size, $(INPUT_SIZE), $(PRECISION_OUT) $(input_size) $(BUILD_DIR)/data.txt > $(RESULT);)

.PHONY: aggregate-all
aggregate-all: $(wildcard $(BUILD_DIR)/$(BENCHMARK)-m*.txt)
	rm -f $(BUILD_DIR)/summary.csv
	@$(foreach file, $?, printf "%s,%f\n" $(file) $$(cat $(file)) >> $(BUILD_DIR)/summary.csv;)

.PHONY: analyze
analyze: aggregate-all
	python3 $(SRC_DIRS)/analyze.py

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)


#"$OPT" \
#  -S \
#  -debug \
#  -load "$LAMP_SIMULATOR" \
#  -lampsim \
#  "$mantissaopts" \
#  build/"$scaling"/"$benchname"/${benchname}.out.ll \
#  -o build/"$scaling"/"$benchname"/${benchname}.lamp.out.ll
#
#$CLANG \
#  build/"$scaling"/"$benchname"/${benchname}.lamp.out.ll \
#  -m32 \
#  -o build/"$scaling"/"$benchname"/${benchname}.lamp


#$CLANG \
#    -o build/"$scaling"/"$benchname"/"$benchname".out.ll \
#    -S -emit-llvm \
#    "$benchpath" \
#    -Isources/. \
#    $xparams \
#    -lm \
#    -DSCALING_FACTOR=$scaling \
#    2> build/"$scaling"/"$benchname"/${benchname}.log || return $?
#  $CLANG \
#    build/"$scaling"/"$benchname"/"$benchname".out.ll \
#    -m32 \
#    -o build/"$scaling"/"$benchname"/"$benchname".out || return $?