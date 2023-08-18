EXECUTABLE_TAFFO  =${EXECUTABLE}-taffo
CLANG             :=$(shell taffo -print-llvm-bin-dir)/clang
CLANGXX           :=$(shell taffo -print-llvm-bin-dir)/clang++
LLVM_LINK         :=$(shell taffo -print-llvm-bin-dir)/llvm-link
LIB               =-lm

TAFFO_DTA    ?=fixp

ifeq ($(TAFFO_DTA),fixp)
endif
ifeq ($(TAFFO_DTA),fixp16)
TAFFO_EXEC_OPTS   += -Xdta -totalbits -Xdta 16
endif
ifeq ($(TAFFO_DTA),f16)
TAFFO_EXEC_OPTS   += -Xdta -usefloat -Xdta f16
endif
ifeq ($(TAFFO_DTA),f32)
TAFFO_EXEC_OPTS   += -Xdta -usefloat -Xdta f32
endif
ifeq ($(TAFFO_DTA),mixed) #ilp
TAFFO_EXEC_OPTS   += -mixedmode -costmodel i7-4 -instructionset fix
endif

ifeq ($(MAKECMDGOALS),)
$(info DTA config type = $(TAFFO_DTA))
endif

.PHONY: all
all: ${EXECUTABLE} ${PTXFILE} ${EXECUTABLE_TAFFO} ${PTXFILE_TAFFO}

${EXECUTABLE}:
	$(CLANGXX) -O3 ${INCLUDE} ${LIB} ${CFLAGS} ${CFILES} -o ${EXECUTABLE}

${EXECUTABLE_TAFFO}:
	mkdir -p taffo_logs
	taffo -O3 \
	  -Xvra -max-unroll=0 \
	  $(TAFFO_EXEC_OPTS) \
	  ${INCLUDE} ${LIB} ${CFLAGS} ${CFILES} -o ${EXECUTABLE_TAFFO} -temp-dir taffo_logs -debug \
	      2> taffo_logs/taffo.log

.PHONY: run
run: ${EXECUTABLE} ${EXECUTABLE_TAFFO}
	@$(RUN_COMMAND)

.PHONY: validate
validate:
	@$(VALIDATE_COMMAND)

.PHONY: clean
clean:
	rm -f *~ ${EXECUTABLE} ${EXECUTABLE_TAFFO}
	rm -rf data/output taffo_logs
