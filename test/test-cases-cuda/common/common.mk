INCLUDE=-I${PATH_TO_UTILS}
EXECUTABLE_TAFFO=$(patsubst %.exe,%.taffo.exe,${EXECUTABLE})
PTXFILE=$(patsubst %.cu,%.ptx,${CUFILE})
PTXFILE_TAFFO=$(patsubst %.cu,%.taffo.ptx,${CUFILE})
CLANG:=$(shell taffo -print-llvm-bin-dir)/clang
CLANGX:=$(shell taffo -print-llvm-bin-dir)/clang++
LLVM_NVPTX:=$(shell taffo -print-llvm-bin-dir)/llc

.PHONY: all
all: ${EXECUTABLE} ${PTXFILE} ${EXECUTABLE_TAFFO} ${PTXFILE_TAFFO}

${EXECUTABLE}:
	${CLANGX} -O3 --cuda-host-only -L/usr/local/cuda/lib64 -I/usr/local/cuda/include        \
    -lcudart_static -ldl -lrt -pthread -I${PATH_TO_UTILS} ${CFLAGS} ${CFILES} -o ${EXECUTABLE} -lcuda

${EXECUTABLE_TAFFO}: ${PTXFILE_TAFFO}
	mkdir -p taffo_drvr_logs
	taffo -O3 --cuda-host-only -S\
    -Xvra -max-unroll=0 \
    -Xdta -notypemerge -Xclang -no-opaque-pointers\
    -Xdta -bufferid-import -Xdta taffo_kern_logs/bufferid.yaml -I/usr/local/cuda/include \
    -I${PATH_TO_UTILS} ${CFLAGS} ${CFILES} -o ${EXECUTABLE_TAFFO}.s -temp-dir taffo_drvr_logs -debug \
        2> taffo_drvr_logs/taffo.log
	${CLANG} -O3 --cuda-host-only -L/usr/local/cuda/lib64 -lstdc++        \
    -lcudart_static -ldl -lrt -pthread ${CFLAGS} ${EXECUTABLE_TAFFO}.s -o ${EXECUTABLE_TAFFO} -lcuda

${PTXFILE}:
	${CLANG} ${CUFILE} --cuda-device-only -S -o ${PTXFILE}

${PTXFILE_TAFFO}:
	mkdir -p taffo_kern_logs
	taffo ${CUFILE} \
		-S -emit-llvm -cudakern -Xdta -notypemerge -temp-dir taffo_kern_logs -debug -Xclang -no-opaque-pointers --cuda-device-only --cuda-gpu-arch=sm_60\
		-Xdta -bufferid-export -Xdta taffo_kern_logs/bufferid.yaml -mixedmode -costmodel core2 \
		-o ${CUFILE}.taffo.ll \
			2> taffo_kern_logs/taffo.log
	$(LLVM_NVPTX) \
		-mcpu=sm_60 -o ${PTXFILE_TAFFO} \
		${CUFILE}.taffo.ll \
	

.PHONY: run
run: ${EXECUTABLE} ${EXECUTABLE_TAFFO}
	./${EXECUTABLE} 2> ${EXECUTABLE}.txt
	./${EXECUTABLE_TAFFO} 2> ${EXECUTABLE_TAFFO}.txt

.PHONY: clean
clean:
	rm -f *~ *.exe *.txt *.ptx *.ll
	rm -rf taffo_kern_logs taffo_drvr_logs
