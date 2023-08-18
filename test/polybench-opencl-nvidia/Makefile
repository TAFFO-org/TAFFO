BENCHS := 2mm 3mm adi atax bicg convolution-2d convolution-3d correlation \
	covariance doitgen fdtd-2d gemm gemver gesummv gramschmidt jacobi-1d-imper \
	jacobi-2d-imper lu mvt syr2k syrk

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
