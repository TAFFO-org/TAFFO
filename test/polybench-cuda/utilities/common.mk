INCLUDE=-I${PATH_TO_UTILS}
EXECUTABLE_TAFFO=$(patsubst %.exe,%.taffo.exe,${EXECUTABLE})
PTXFILE=$(patsubst %.cu,%.ptx,${CUFILE})
PTXFILE_TAFFO=$(patsubst %.cu,%.taffo.ptx,${CUFILE})
CLANG:=$(shell taffo -print-llvm-bin-dir)/clang
CLANGX:=$(shell taffo -print-llvm-bin-dir)/clang++
LLVM_NVPTX:=$(shell taffo -print-llvm-bin-dir)/llc

.PHONY: all
all: ${EXECUTABLE} ${PTXFILE} ${EXECUTABLE_TAFFO} ${PTXFILE_TAFFO}

${EXECUTABLE}:  ${PTXFILE}
	${CLANGX} -O3 -DPOLYBENCH_STACK_ARRAYS --cuda-host-only -L/usr/local/cuda/lib64 -I/usr/local/cuda/include     \
    -lcudart_static -ldl -lrt -pthread -I${PATH_TO_UTILS} ${CFLAGS} ${CFILES} -o ${EXECUTABLE} -lcuda

${EXECUTABLE_TAFFO}: ${PTXFILE_TAFFO}
	mkdir -p taffo_drvr_logs
	taffo -O3 -DPOLYBENCH_STACK_ARRAYS --cuda-host-only -S\
    -Xvra -max-unroll=0 \
    -Xdta -notypemerge -Xclang -no-opaque-pointers\
    -Xdta -bufferid-import -Xdta taffo_kern_logs/bufferid.yaml -I/usr/local/cuda/include \
    -I${PATH_TO_UTILS} ${CFLAGS} ${CFILES} -o ${EXECUTABLE_TAFFO}.s -temp-dir taffo_drvr_logs -debug \
        2> taffo_drvr_logs/taffo.log
	${CLANG} -O3 --cuda-host-only -L/usr/local/cuda/lib64 -lstdc++ -lm \
    -lcudart_static -ldl -lrt -pthread ${CFLAGS} ${EXECUTABLE_TAFFO}.s -o ${EXECUTABLE_TAFFO} -lcuda

${PTXFILE}:
	${CLANG} -O3 ${CUFILE} -I${PATH_TO_UTILS} --cuda-device-only --cuda-gpu-arch=sm_86 -S -o ${PTXFILE}

${PTXFILE_TAFFO}:
	mkdir -p taffo_kern_logs
	taffo ${CUFILE}  -I${PATH_TO_UTILS}\
		-S -emit-llvm -cudakern -Xdta -notypemerge -temp-dir taffo_kern_logs -debug -Xclang -no-opaque-pointers --cuda-device-only --cuda-gpu-arch=sm_86\
		-Xdta -bufferid-export -Xdta taffo_kern_logs/bufferid.yaml \
		-o ${CUFILE}.taffo.ll \
			2> taffo_kern_logs/taffo.log
	$(LLVM_NVPTX) \
		-mcpu=sm_86 -o ${PTXFILE_TAFFO} \
		${CUFILE}.taffo.ll \
	

.PHONY: run
run: ${EXECUTABLE} ${EXECUTABLE_TAFFO}
	./${EXECUTABLE} 2> ${EXECUTABLE}.txt
	./${EXECUTABLE_TAFFO} 2> ${EXECUTABLE_TAFFO}.txt

.PHONY: validate
validate:
	@../../utilities/validate.py --ref=${EXECUTABLE}.txt --check=${EXECUTABLE_TAFFO}.txt	

.PHONY: clean
clean:
	rm -f *~ *.exe *.txt *.ptx *.ll *.exe.s
	rm -rf taffo_kern_logs taffo_drvr_logs
