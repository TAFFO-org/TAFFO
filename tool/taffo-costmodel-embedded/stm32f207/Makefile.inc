CPUFLAGS =-mthumb -mcpu=cortex-m3 -DSTM32F207xx
CFLAGS  +=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER
ASMFLAGS+=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER
LDFLAGS +=$(CPUFLAGS) -T$(TARGET)/STM32F207IGHx_FLASH.ld -Wl,--gc-sections
C_SRC   +=$(wildcard $(TARGET)/*.c)
ASM_SRC +=$(wildcard $(TARGET)/*.s)

SERIAL_DEVICE?=$(wildcard /dev/cu.usbserial*)

.PHONY: flash monitor cpuflags costmodel

flash:
	st-flash --reset write $(OUTPUT_BIN) 0x08000000

monitor:
	(stty speed 115200 >/dev/null && cat) < $(SERIAL_DEVICE) | tee monitor.log

cpuflags:
	@echo '$(CPUFLAGS)'

costmodel:
	@echo stm32f4-float
