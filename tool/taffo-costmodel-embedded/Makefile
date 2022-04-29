.SUFFIXES: # builtin rules are EVIL
CC = arm-none-eabi-gcc
CXX = arm-none-eabi-g++
AS = arm-none-eabi-as
OBJCOPY = arm-none-eabi-objcopy

OUTPUT = taffo-costmodel
TARGET ?= stm32f407

OBJ_DIR = ./obj_$(TARGET)
OUTPUT_ELF = $(OUTPUT).elf
OUTPUT_BIN = $(OUTPUT).bin
OUTPUT_MAP = $(OUTPUT).map

.PHONY: all clean target

all: $(OUTPUT_BIN)

C_SRC += $(wildcard *.c)

CFLAGS = -Os -ffreestanding -g
ASMFLAGS = -ffreestanding -g
LDFLAGS = -ffreestanding -lm -lc -Wl,-Map=$(OUTPUT_MAP)
-include $(TARGET)/Makefile.inc

C_OBJ = $(patsubst %, $(OBJ_DIR)/%, $(notdir $(C_SRC:.c=.o)))
ASM_OBJ = $(patsubst %, $(OBJ_DIR)/%, $(notdir $(ASM_SRC:.s=.o)))
OBJ = $(C_OBJ) $(ASM_OBJ)
DEPS = $(C_OBJ:.o=.d)

-include $(DEPS)

$(OBJ_DIR)/%.o: %.c
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/%.o: $(TARGET)/%.c
	$(CC) $(CFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/%.o: $(TARGET)/%.s
	$(CC) $(ASMFLAGS) -MMD -c $< -o $@

$(OUTPUT_ELF): $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

$(OBJ): | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR)
	rm -f *.elf *.bin *.map *.log *.s

target:
	@echo '$(TARGET)'

