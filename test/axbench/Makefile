BENCHS := blackscholes fft inversek2j jmeint kmeans sobel

VALIDATE_FMT :='%40s %12s %12s%14s%20s\n'

.PHONY: all
all:
	for b in $(BENCHS); do $(MAKE) -j -C $$b; done

.PHONY: clean
clean:
	for b in $(BENCHS); do $(MAKE) -C $$b clean; done

.PHONY: run
run:
	for b in $(BENCHS); do $(MAKE) -C $$b run; done

.PHONY: validate
validate:
	@printf $(VALIDATE_FMT) '' 'fix T' 'flo T' 'avg err %' 'avg abs err'
	@export FORMAT=$(VALIDATE_FMT); \
	for b in $(BENCHS); do $(MAKE) -s -C $$b validate; done
