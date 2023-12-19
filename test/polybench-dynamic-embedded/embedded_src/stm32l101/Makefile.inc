CPUFLAGS +=-mthumb -mcpu=cortex-m0plus -DSTM32L010xB
CFLAGS  +=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER --specs=nano.specs -Os
ASMFLAGS+=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER
LDFLAGS +=$(CPUFLAGS) -T$(TARGET)/STM32L010RBTx_FLASH.ld -Wl,--gc-sections --specs=nano.specs -u _printf_float -Os
C_SRC   +=$(wildcard $(TARGET)/*.c)
ASM_SRC +=$(wildcard $(TARGET)/*.s)

STM32_Programmer_CLI ?= STM32_Programmer_CLI
SERIAL_DEVICE?=$(wildcard /dev/ttyACM*)
EMBEDDED_OUTPUT?=monitor.txt
TIMEOUT_SECONDS=300

.PHONY: flash monitor cpuflags

flash:
	$(STM32_Programmer_CLI) -c port=SWD -w $(OUTPUT_BIN) 0x08000000 -v

monitor:
	$(STM32_Programmer_CLI) -c port=SWD --start 0x08000000
	grabserial -d $(SERIAL_DEVICE) -e $(TIMEOUT_SECONDS) -q "^END BENCHMARK" -o $(EMBEDDED_OUTPUT)

cpuflags:
	@echo '$(CPUFLAGS)'

