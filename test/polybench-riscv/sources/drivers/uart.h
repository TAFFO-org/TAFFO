#ifndef UART_H
#define UART_H

#include "lamp_core.h"


/**
 * UART peripheral.
 * FIXME: please note that the following 8-bit registers
 * are declared as 32-bit ones because of an intrinsic
 * issue of the MEM stage when coupled with the UART.
 * The address is always aligned to a 32-bit word
 * (2LSB are zeroed) while the byte selector indicates
 * which bytes have to be written or read. Thus, we
 * cannot simple propagate the complete address to the
 * slave because the UART always expects the input data
 * in the lowest 8-bit of the 32-bit wishbone data input
 * while the MEM aligned is compliant with 32-bit alignment
 * and byte enabling mechanism.
 */
struct UART_peripheral {
    volatile uint32_t THR_DLL; // 00
    volatile uint32_t IER_DLR; // 01
    volatile uint32_t IIR_FCR; // 02
    volatile uint32_t LCR;     // 03
    volatile uint32_t MCR;     // 04
    volatile uint32_t LSR;     // 05
    volatile uint32_t MSR;     // 06
    volatile uint32_t SCR;     // 07
};

#define UART_0 ((struct UART_peripheral *)(0x00100000))


void uart_init(void);

void uart_putc(const char c);

void uart_print(const char *str, unsigned int len);

unsigned int str_len(const char *str);

#endif
