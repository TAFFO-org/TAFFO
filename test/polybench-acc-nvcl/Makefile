BENCHS := \
	correlation \
	covariance \
	lu \
	fdtd-2d

.PHONY: all
all:
	for b in $(BENCHS); do $(MAKE) -C $$b; done

.PHONY: clean
clean:
	for b in $(BENCHS); do $(MAKE) -C $$b clean; done

.PHONY: run
run:
	for b in $(BENCHS); do $(MAKE) -C $$b run; done

