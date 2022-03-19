CFLAGS  +=-mthumb -mcpu=cortex-m3 -I$(TARGET) -DUSE_HAL_DRIVER -DSTM32F207xx
ASMFLAGS+=-mthumb -mcpu=cortex-m3 -I$(TARGET) -DUSE_HAL_DRIVER -DSTM32F207xx
LDFLAGS +=-mthumb -mcpu=cortex-m3 -T$(TARGET)/STM32F207IGHx_FLASH.ld -Wl,--gc-sections
C_SRC   +=$(wildcard $(TARGET)/*.c)
ASM_SRC +=$(wildcard $(TARGET)/*.s)

SERIAL_DEVICE?=$(wildcard /dev/cu.usbserial*)

.PHONY: flash monitor

flash:
	st-flash --reset write $(OUTPUT_BIN) 0x08000000

monitor:
	(stty speed 115200 >/dev/null && cat) < $(SERIAL_DEVICE) | tee monitor.log
