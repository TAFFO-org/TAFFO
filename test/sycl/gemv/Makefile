# Compiler and flags
TAFFO=taffo
CLANG=/home/nico/CLionProjects/llvm-dpc++/build/bin/clang++
SRC=gemv.cpp
OUTPUT_TAFFO=gemv_taffo.exe
OUTPUT_CLANG=gemv_clang.exe
LOG_TAFFO=taffo_log.txt
TEMP_DIR=taffo-temp
BUFFERID=bufferid.txt
TAFFOFLAGS=-bufferid-export $(BUFFERID) -bufferid-import $(BUFFERID) -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3
CLANGFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3

# TAFFO target
taffo: $(OUTPUT_TAFFO)

# TAFFO_DEBUG build target
$(OUTPUT_TAFFO): $(SRC) $(TEMP_DIR)
	@$(TAFFO) -debug -temp-dir $(TEMP_DIR) $(TAFFOFLAGS) $(SRC) -o $(OUTPUT_TAFFO) > $(LOG_TAFFO) 2>&1

# CLANG build target
clang: $(OUTPUT_CLANG)

$(OUTPUT_CLANG): $(SRC)
	@$(CLANG) $(CLANGFLAGS) $(SRC) -o $(OUTPUT_CLANG)

# Create temp directory if it doesn't exist
$(TEMP_DIR):
	@mkdir -p $(TEMP_DIR)

# Clean target
clean:
	rm -f $(OUTPUT_TAFFO) $(OUTPUT_CLANG) $(LOG_TAFFO) $(BUFFERID)
	rm -rf $(TEMP_DIR)

# Phony targets
.PHONY: taffo clang clean
