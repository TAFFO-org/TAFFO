# Compiler and flags
CLANG=/home/nico/CLionProjects/llvm-dpc++/build/bin/clang++
SRC=check_performance.cpp
OUTPUT=check_performance.exe
CLANGFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3

check: $(OUTPUT)

$(OUTPUT): $(SRC)
	@$(CLANG) $(CLANGFLAGS) $(SRC) -o $(OUTPUT)

# Clean target
clean:
	rm -f $(OUTPUT)

# Phony targets
.PHONY: check clean
