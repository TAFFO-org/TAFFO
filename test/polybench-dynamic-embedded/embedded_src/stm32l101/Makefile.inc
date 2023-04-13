CPUFLAGS +=-mthumb -mcpu=cortex-m0plus -DSTM32L010xB
CFLAGS  +=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER --specs=nano.specs -Os
ASMFLAGS+=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER
LDFLAGS +=$(CPUFLAGS) -T$(TARGET)/STM32L010RBTx_FLASH.ld -Wl,--gc-sections --specs=nano.specs -u _printf_float -Os
C_SRC   +=$(wildcard $(TARGET)/*.c)
ASM_SRC +=$(wildcard $(TARGET)/*.s)

SERIAL_DEVICE?=$(wildcard /dev/cu.usbmodem*)

.PHONY: flash monitor cpuflags costmodel

flash:
	st-flash --reset write $(OUTPUT_BIN) 0x08000000

monitor:
	cat < $(SERIAL_DEVICE) | tee monitor.log

cpuflags:
	@echo '$(CPUFLAGS)'

costmodel:
	@echo stm32f4-float
