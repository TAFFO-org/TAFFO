#ifndef INTERRUPT_H
#define INTERRUPT_H

#include "lamp_core.h"


/**
 * Globally disable interrupts.
 * \return nothing.
 */
static inline
void int_disable(void) {
    // Set bit 3 (MIE) to 0
    uint32_t mstatus = 8;
    asm volatile ("csrc mstatus, %0" : /* no output */ : "r" (mstatus));
}

/**
 * Globally enable interrupts.
 * \return nothing.
 */
static inline
void int_enable(void) {
    // Set mstatus.MIE = 1
    uint32_t mstatus = 8;
    asm volatile ("csrs mstatus, %0" : /* no output */ : "r" (mstatus));
}

/**
 * Disable M mode timer interrupts.
 * \return nothing.
 */
static inline
void int_mtimer_disable(void) {
    // Set mie_MTIE = 0
    uint32_t mie = 128;
    asm volatile ("csrc mie, %0" : /* no output */ : "r" (mie));
}

/**
 * Enable M mode timer interrupts.
 * \return nothing.
 */
static inline
void int_mtimer_enable(void) {
    // Set mie.MTIE = 1
    uint32_t mie = 128;
    asm volatile ("csrs mie, %0" : /* no output */ : "r" (mie));
}

/**
 * Overwritable interrupt handler.
 * \return nothing.
 */
void int_default_handler(void);

#endif