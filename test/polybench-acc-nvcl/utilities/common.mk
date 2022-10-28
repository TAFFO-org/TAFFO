INCLUDE           =-I${PATH_TO_UTILS}
EXECUTABLE_TAFFO  =$(patsubst %.exe,%.taffo.exe,${EXECUTABLE})
PTXFILE           =$(patsubst %.cl,%.ptx,${CLFILE})
PTXFILE_TAFFO     =$(patsubst %.cl,%.taffo.ptx,${CLFILE})
CLANG             :=$(shell taffo -print-llvm-bin-dir)/clang
LLVM_LINK         :=$(shell taffo -print-llvm-bin-dir)/llvm-link
LIB               =-lOpenCL -lm

TAFFO_HOST_DTA    ?=fixp
TAFFO_KERN_ARGS   ?=fixp
TAFFO_KERN_DTA    ?=fixp

ifeq ($(TAFFO_HOST_DTA),fixp)
endif
ifeq ($(TAFFO_HOST_DTA),f16)
TAFFO_EXEC_OPTS   += -Xdta -usefloat -Xdta fp16
endif
ifeq ($(TAFFO_HOST_DTA),f32)
TAFFO_EXEC_OPTS   += -Xdta -usefloat -Xdta fp32
endif
ifeq ($(TAFFO_HOST_DTA),mixed) #ilp
TAFFO_EXEC_OPTS   += -mixedmode -costmodel i7-4 -instructionset fix
endif

ifeq ($(TAFFO_KERN_ARGS),fixp)
TAFFO_EXEC_OPTS   += -Xdta -bufferid-import -Xdta bufferid_conf_fixp32.yaml
TAFFO_KERN_OPTS   += -Xdta -bufferid-import -Xdta bufferid_conf_fixp32.yaml
endif
ifeq ($(TAFFO_KERN_ARGS),f16)
TAFFO_EXEC_OPTS   += -Xdta -bufferid-import -Xdta bufferid_conf_fp16.yaml
TAFFO_KERN_OPTS   += -Xdta -bufferid-import -Xdta bufferid_conf_fp16.yaml
endif
ifeq ($(TAFFO_KERN_ARGS),f32)
TAFFO_EXEC_OPTS   += -Xdta -bufferid-import -Xdta bufferid_conf_fp32.yaml
TAFFO_KERN_OPTS   += -Xdta -bufferid-import -Xdta bufferid_conf_fp32.yaml
endif
ifeq ($(TAFFO_KERN_ARGS),mixed)
TAFFO_EXEC_OPTS   += -Xdta -bufferid-import -Xdta taffo_kern_logs/bufferid.yaml
TAFFO_KERN_OPTS   += -Xdta -bufferid-export -Xdta taffo_kern_logs/bufferid.yaml
endif

ifeq ($(TAFFO_KERN_DTA),fixp)
endif
ifeq ($(TAFFO_KERN_DTA),f16)
TAFFO_KERN_OPTS   += -Xdta -usefloat -Xdta fp16
endif
ifeq ($(TAFFO_KERN_DTA),f32)
TAFFO_KERN_OPTS   += -Xdta -usefloat -Xdta fp32
endif
ifeq ($(TAFFO_KERN_DTA),mixed) #ilp
TAFFO_KERN_OPTS   += -mixedmode -costmodel nv_sm86 -instructionset gpu -Xdta -mixedtuningenob -Xdta 1 -Xdta -mixedtuningtime -Xdta 10000 -Xdta -mixedtuningcastingtime -Xdta 10000
endif

ifeq ($(MAKECMDGOALS),)
$(info Host DTA config type    = $(TAFFO_HOST_DTA))
$(info Kernel args config type = $(TAFFO_KERN_ARGS))
$(info Kernel DTA config type  = $(TAFFO_KERN_DTA))
endif

.PHONY: all
all: ${EXECUTABLE} ${PTXFILE} ${EXECUTABLE_TAFFO} ${PTXFILE_TAFFO}

${EXECUTABLE}:
	$(CLANG) -O3 -DPOLYBENCH_STACK_ARRAYS -I${PATH_TO_UTILS} ${LIB} ${CFLAGS} ${CFILES} -o ${EXECUTABLE}

${EXECUTABLE_TAFFO}: ${PTXFILE_TAFFO}
	mkdir -p taffo_drvr_logs
	taffo -O3 -DPOLYBENCH_STACK_ARRAYS \
    -Xvra -max-unroll=0 \
    $(TAFFO_EXEC_OPTS) \
    -I${PATH_TO_UTILS} ${LIB} ${CFLAGS} ${CFILES} -o ${EXECUTABLE_TAFFO} -temp-dir taffo_drvr_logs -debug \
        2> taffo_drvr_logs/taffo.log

${PTXFILE}:
	$(CLANG) \
		-Xclang -finclude-default-header -Xclang -fdeclare-opencl-builtins -Xclang -disable-O0-optnone -Xclang -no-opaque-pointers\
		-D__cl_clang_storage_class_specifiers \
		-target nvptx64-unknown-nvcl \
		-O0 -march=sm_86 -S -emit-llvm \
		-o ${CLFILE}.ll \
		${CLFILE}
	$(LLVM_LINK) \
		-S -o ${CLFILE}.linked.ll \
		${CLFILE}.ll \
		/usr/lib/clc/nvptx64--nvidiacl.bc
	$(CLANG) \
		-O3 -target nvptx64-unknown-nvcl -march=sm_86 -S \
		-o ${PTXFILE} \
		${CLFILE}.linked.ll

${PTXFILE_TAFFO}: ${PTXFILE}
	mkdir -p taffo_kern_logs
	taffo ${CLFILE}.ll \
		-S -emit-llvm -oclkern -target nvptx64-unknown-nvcl -temp-dir taffo_kern_logs -debug \
		$(TAFFO_KERN_OPTS) \
		-o ${CLFILE}.taffo.ll \
			2> taffo_kern_logs/taffo.log
	$(LLVM_LINK) \
		-S -o ${CLFILE}.taffo.linked.ll \
		${CLFILE}.taffo.ll \
		/usr/lib/clc/nvptx64--nvidiacl.bc
	$(CLANG) \
		-O3 -target nvptx64-unknown-nvcl -march=sm_86 -S \
		-o ${PTXFILE_TAFFO} \
		${CLFILE}.taffo.linked.ll

.PHONY: run
run: ${EXECUTABLE} ${EXECUTABLE_TAFFO}
	echo BENCHMARK = $$(basename $$(pwd))
	ulimit -s unlimited; ./${EXECUTABLE} 2> ${EXECUTABLE}.txt
	ulimit -s unlimited; ./${EXECUTABLE_TAFFO} 2> ${EXECUTABLE_TAFFO}.txt

.PHONY: validate
validate:
	@../utilities/validate.py --ref=${EXECUTABLE}.txt --check=${EXECUTABLE_TAFFO}.txt

.PHONY: clean
clean:
	rm -f *~ *.exe *.txt *.ptx *.ll
	rm -rf taffo_kern_logs taffo_drvr_logs
