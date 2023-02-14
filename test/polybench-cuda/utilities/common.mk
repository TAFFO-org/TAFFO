INCLUDE=-I${PATH_TO_UTILS}
EXECUTABLE_TAFFO=$(patsubst %.exe,%.taffo.exe,${EXECUTABLE})
PTXFILE=$(patsubst %.cu,%.ptx,${CUFILE})
PTXFILE_TAFFO=$(patsubst %.cu,%.taffo.ptx,${CUFILE})
CLANG:=$(shell taffo -print-llvm-bin-dir)/clang
CLANGX:=$(shell taffo -print-llvm-bin-dir)/clang++
LLVM_NVPTX:=$(shell taffo -print-llvm-bin-dir)/llc

TAFFO_HOST_DTA    ?=fixp
TAFFO_KERN_ARGS   ?=fixp
TAFFO_KERN_DTA    ?=fixp

ifeq ($(TAFFO_HOST_DTA),fixp)
endif
ifeq ($(TAFFO_HOST_DTA),f16)
TAFFO_EXEC_OPTS   += -Xdta -usefloat -Xdta f16
endif
ifeq ($(TAFFO_HOST_DTA),f32)
TAFFO_EXEC_OPTS   += -Xdta -usefloat -Xdta f32
endif
ifeq ($(TAFFO_HOST_DTA),mixed) #ilp
TAFFO_EXEC_OPTS   += -mixedmode -costmodel 5600x -instructionset fix
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
TAFFO_KERN_OPTS   += -Xdta -usefloat -Xdta f16
endif
ifeq ($(TAFFO_KERN_DTA),f32)
TAFFO_KERN_OPTS   += -Xdta -usefloat -Xdta f32
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

${EXECUTABLE}:  ${PTXFILE}
	${CLANGX} -O3 -DPOLYBENCH_STACK_ARRAYS --cuda-host-only -L/usr/local/cuda/lib64 -I/usr/local/cuda/include     \
    -lcudart_static -ldl -lrt -pthread -I${PATH_TO_UTILS} ${CFLAGS} ${CFILES} -o ${EXECUTABLE} -lcuda

${EXECUTABLE_TAFFO}: ${PTXFILE_TAFFO}
	mkdir -p taffo_drvr_logs
	taffo -O3 -DPOLYBENCH_STACK_ARRAYS --cuda-host-only -S --rtlib=compiler-rt\
    -Xvra -max-unroll=0 $(TAFFO_EXEC_OPTS)\
    -I/usr/local/cuda/include -I${PATH_TO_UTILS} ${CFLAGS} ${CFILES} -o ${EXECUTABLE_TAFFO}.s -temp-dir taffo_drvr_logs -debug \
        2> taffo_drvr_logs/taffo.log
	${CLANG} -O3 --cuda-host-only -L/usr/local/cuda/lib64 -lstdc++ -lm --rtlib=compiler-rt \
    -lcudart_static -ldl -lrt -pthread ${CFLAGS} ${EXECUTABLE_TAFFO}.s -o ${EXECUTABLE_TAFFO} -lcuda

${PTXFILE}:
	${CLANG} -O3 ${CUFILE} -I${PATH_TO_UTILS} --cuda-device-only --cuda-gpu-arch=sm_86 -S -o ${PTXFILE}

${PTXFILE_TAFFO}:
	mkdir -p taffo_kern_logs
	taffo ${CUFILE} -I${PATH_TO_UTILS} \
	-S -emit-llvm -cudakern -temp-dir taffo_kern_logs -debug --cuda-device-only --cuda-gpu-arch=sm_86 \
	$(TAFFO_KERN_OPTS) -o ${CUFILE}.taffo.ll \
			2> taffo_kern_logs/taffo.log
	$(LLVM_NVPTX) \
		-mcpu=sm_86 -o ${PTXFILE_TAFFO} \
		${CUFILE}.taffo.ll \
	

.PHONY: run
run: ${EXECUTABLE} ${EXECUTABLE_TAFFO}
	echo BENCHMARK = $$(basename $$(pwd))
	ulimit -s unlimited; ./${EXECUTABLE} 2> ${EXECUTABLE}.txt
	ulimit -s unlimited; ./${EXECUTABLE_TAFFO} 2> ${EXECUTABLE_TAFFO}.txt

.PHONY: validate
validate:
	@../../utilities/validate.py --ref=${EXECUTABLE}.txt --check=${EXECUTABLE_TAFFO}.txt	

.PHONY: clean
clean:
	rm -f *~ *.exe *.txt *.ptx *.ll *.exe.s
	rm -rf taffo_kern_logs taffo_drvr_logs
	