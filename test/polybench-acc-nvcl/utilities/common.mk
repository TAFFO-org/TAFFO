INCLUDE=-I${PATH_TO_UTILS}
EXECUTABLE_TAFFO=$(patsubst %.exe,%.taffo.exe,${EXECUTABLE})
PTXFILE=$(patsubst %.cl,%.ptx,${CLFILE})
PTXFILE_TAFFO=$(patsubst %.cl,%.taffo.ptx,${CLFILE})
CLANG=$(shell taffo -print-clang)
LIB=-lOpenCL -lm

.PHONY: all
all: ${EXECUTABLE} ${PTXFILE} ${EXECUTABLE_TAFFO} ${PTXFILE_TAFFO}

${EXECUTABLE}:
	clang-14 -O3 -DPOLYBENCH_STACK_ARRAYS -I${PATH_TO_UTILS} ${LIB} ${CFLAGS} ${CFILES} -o ${EXECUTABLE}

${EXECUTABLE_TAFFO}:
	mkdir -p taffo_drvr_logs
	taffo -O3 -DPOLYBENCH_STACK_ARRAYS -Xvra -max-unroll=0 -Xdta -notypemerge -I${PATH_TO_UTILS} ${LIB} ${CFLAGS} ${CFILES} -o ${EXECUTABLE_TAFFO} -temp-dir taffo_drvr_logs -debug \
        2> taffo_drvr_logs/taffo.log

${PTXFILE}:
	$(CLANG) \
		-Xclang -finclude-default-header -Xclang -fdeclare-opencl-builtins -Xclang -disable-O0-optnone\
		-D__cl_clang_storage_class_specifiers \
		-target nvptx64-unknown-nvcl \
		-O0 -march=sm_86 -S -emit-llvm \
		-o ${CLFILE}.ll \
		${CLFILE}
	llvm-link-14 \
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
		-S -emit-llvm -oclkern -Xdta -notypemerge -target nvptx64-unknown-nvcl -temp-dir taffo_kern_logs -debug \
		-o ${CLFILE}.taffo.ll \
			2> taffo_kern_logs/taffo.log
	llvm-link-14 \
		-S -o ${CLFILE}.taffo.linked.ll \
		${CLFILE}.taffo.ll \
		/usr/lib/clc/nvptx64--nvidiacl.bc
	$(CLANG) \
		-O3 -target nvptx64-unknown-nvcl -march=sm_86 -S \
		-o ${PTXFILE_TAFFO} \
		${CLFILE}.taffo.linked.ll

.PHONY: run
run: ${EXECUTABLE} ${EXECUTABLE_TAFFO}
	./${EXECUTABLE}
	./${EXECUTABLE_TAFFO}

.PHONY: clean
clean:
	rm -f *~ *.exe *.txt *.ptx *.ll
	rm -rf taffo_kern_logs taffo_drvr_logs
