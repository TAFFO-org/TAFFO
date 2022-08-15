#ifndef TIMER_H
#define TIMER_H

#include "lamp_core.h"


/**
 * Local timer.
 */
struct TIMER_peripheral {
    volatile unsigned int MTIME_L;  // 00
    volatile unsigned int MTIME_H;  // 04
    volatile unsigned int MTCMP_L;  // 08
    volatile unsigned int MTCMP_H;  // 0C
};

#define TIMER_0 ((struct TIMER_peripheral *)(0x00200000))

/**
 * Time constants.
 */
#define TS_1S  SYS_CLK_FREQ_HZ
#define TS_1MS SYS_CLK_FREQ_HZ/1000


/**
 * Reset the MTIME special purpose register.
 * \return nothing.
 */
static inline
void timer_resetMtime(void) {
    TIMER_0->MTIME_H = 0;
    TIMER_0->MTIME_L = 0;
}

/**
 * Set the lower 32-bit of the MTIMECMP register to the
 * passed value.
 * \param value the value to be stored into MTCMP_L.
 * \return nothing.
 */
static inline
void timer_setMtimecmpL(uint32_t value) {
    TIMER_0->MTCMP_L = value;
}

/**
 * Set the higher 32-bit of the MTIMECMP register to the
 * passed value.
 * \param value the value to be stored into MTCMP_H.
 * \return nothing.
 */
static inline
void timer_setMtimecmpH(uint32_t value) {
    TIMER_0->MTCMP_H = value;
}

/**
 * Get the lower 32-bit of MTIME special purpose register.
 * \return the lower 32-bit of the MTIME register.
 */
static inline
uint32_t timer_getMtimeL(void) {
    return TIMER_0->MTIME_L;
}

/**
 * Get the higher 32-bit of the MTIME special purpose register.
 * \return the higher 32-bit of the MTIME register.
 */
static inline
uint32_t timer_getMtimeH(void) {
    return TIMER_0->MTIME_H;
}

#endif
