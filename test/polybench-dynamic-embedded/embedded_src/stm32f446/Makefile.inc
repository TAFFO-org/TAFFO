CPUFLAGS +=-mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -DSTM32F446xx
CFLAGS  +=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER --specs=nano.specs -Os
ASMFLAGS+=$(CPUFLAGS) -I$(TARGET) -DUSE_HAL_DRIVER  -Wall -fdata-sections -ffunction-sections
LDFLAGS +=$(CPUFLAGS) -T$(TARGET)/STM32F446ZETx_FLASH.ld -Wl,--gc-sections --specs=nano.specs -u _printf_float -Os  -Wall -fdata-sections -ffunction-sections
C_SRC   +=$(wildcard $(TARGET)/*.c)
ASM_SRC +=$(wildcard $(TARGET)/*.s)

STM32_Programmer_CLI ?= STM32_Programmer_CLI
SERIAL_DEVICE?=$(wildcard /dev/ttyACM1*)
EMBEDDED_OUTPUT?=monitor.txt
TIMEOUT_SECONDS=300

.PHONY: flash monitor cpuflags

new_flash:
	st-flash --reset write $(OUTPUT_BIN) 0x08000000

new_monitor:
	st-flash --serial /dev/ttyACM0 reset
	sleep 2
	python3 ~/Projects/grabserial/grabserial -b 115200 -d /dev/ttyACM1 -e $(TIMEOUT_SECONDS) -q "^END BENCHMARK" -o $(EMBEDDED_OUTPUT)

flash:
	$(STM32_Programmer_CLI) -c port=SWD -w $(OUTPUT_BIN) 0x08000000 -v

monitor:
	$(STM32_Programmer_CLI) -c port=SWD --start 0x08000000
	python3 ~/Projects/grabserial/grabserial -d $(SERIAL_DEVICE) -e $(TIMEOUT_SECONDS) -q "^END BENCHMARK" -o $(EMBEDDED_OUTPUT)

cpuflags:
	@echo '$(CPUFLAGS)'
