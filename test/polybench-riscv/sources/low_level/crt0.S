/**
 * \author Giovanni Scotti
 * \author Davide Zoni
 *
 * Description:
 *  crt0.S: entry point for RISC-V user programs.
 */

    /* BSS INITIALIZATION MACRO */
    .macro memset dst_beg, dst_end, val_reg
    j       memset_2
memset_1:
    sw      \val_reg, (\dst_beg)
    add     \dst_beg, \dst_beg, 4
memset_2:
    bltu    \dst_beg, \dst_end, memset_1
    .endm


    .section .init
    .global _start
    .type   _start, @function
_start:
    /* Setup the global pointer, ABI assumes points to __global_pointer$ symbol. */
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    /* Setup the stack pointer. */  
    la      sp, __stack_pointer$
	
	/* clear bss and sbss. They are adjacent sections. */
    la      t0, __bss_start
    la      t1, __bss_end
    memset  t0, t1, zero
    
    /* Setup machine trap vector */
    la      t0, default_handler
    csrrw   zero, mtvec, t0

    /* Call main, leaving the arguments undefined. */
    call    main

    /* Spin after main() returns. */
1:
    j 1b

default_handler:
	j default_handler
    /*call    int_default_handler*/


    .size  _start, .-_start

    .global _init
    .type   _init, @function
    .global _fini
    .type   _fini, @function
_init:
_fini:
    # These don't have to do anything since we use init_array/fini_array.
    ret
    .size  _init, .-_init
    .size  _fini, .-_fini
