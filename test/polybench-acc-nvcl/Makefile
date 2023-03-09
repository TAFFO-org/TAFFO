BENCHS := \
	correlation \
	covariance \
	lu \
	fdtd-2d \
	jacobi-1d-imper \
	jacobi-2d-imper \
	gesummv \
	mvt \
	syr2k \
	syrk \
	gramschmidt \
	convolution-2d \
	convolution-3d \
	2mm \
	atax \
	bicg \
	doitgen \
	gemm \
	gemver

.PHONY: all
all:
	for b in $(BENCHS); do $(MAKE) -j -C $$b; done

.PHONY: clean
clean:
	for b in $(BENCHS); do $(MAKE) -C $$b clean; done

.PHONY: run
run:
	for b in $(BENCHS); do $(MAKE) -C $$b run; done

.PHONY: run-taffo
run-taffo:
	for b in $(BENCHS); do $(MAKE) -C $$b run-taffo; done

.PHONY: validate
validate:
	for b in $(BENCHS); do echo "$$b"; $(MAKE) -s -C $$b validate; done
