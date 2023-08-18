BENCHS := \
	vec_add \


.PHONY: all
all:
	for b in $(BENCHS); do $(MAKE) -C $$b; done

.PHONY: clean
clean:
	for b in $(BENCHS); do $(MAKE) -C $$b clean; done

.PHONY: run
run:
	for b in $(BENCHS); do $(MAKE) -C $$b run; done

.PHONY: validate
validate:
	for b in $(BENCHS); do echo "$$b"; $(MAKE) -s -C $$b validate; done
