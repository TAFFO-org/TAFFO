#include "uart.h"


/**
 * Configure the UART before use. Settings are as follows:
 * 8-bit data, 1 stop bit, no parity.
 * \return nothing.
 */
void uart_init(void) {
    UART_0->LCR = 0x83;
    UART_0->THR_DLL = SYS_CLK_FREQ_HZ / (16 * UART_BAUD_RATE);
    UART_0->LCR = 0x03;
}

/**
 * Print the given character. Check out the TEMT (transmitter empty)
 * bit in the LS register and transmit only when both the transmitter
 * shift register and the transmitter holding register are empty.
 * \param c the character to print.
 * \return nothing.
 */
void uart_putc(const char c) {
    while((UART_0->LSR & 0x40) == 0);
    UART_0->THR_DLL = c;
}

/**
 * Print the given string.
 * \param str the pointer to a sequence of characters to print.
 * \param len the length of the string to print.
 * \return nothing.
 */
void uart_print(const char *str, unsigned int len) {
    while(len > 0) {
        uart_putc(*str++);
        len--;
    }
}

/**
 * Compute the number of characters of a given string.
 * \param str the pointer to a string.
 * \return the number of characters of the passed string.
 */
unsigned int str_len(const char *str) {
    unsigned int cnt = 0;
    
    while(*str != '\0') {
        str++;
        cnt++;
    }

    return cnt;
}
