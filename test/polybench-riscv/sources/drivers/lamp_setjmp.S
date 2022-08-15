/**
 * \author Giovanni Scotti
 * \author Davide Zoni
 *
 * Description:
 *  The setjmp.S provides the 'setjmp' and 'longjmp' functions to be used with
 *  the context switch capabilities of the lamp RISC-V core.
 *
 *  Suppose 2 threads and a manager:
 *	tc0, tc1 and tc2, where the tc0 is the manager thread environment.
 *
 *  The manager has to prepare the other two contexts by filling in the tc1
 *  and tc2. Then it invokes the following code:
 *  
 *  int res = setjmp(&tc0); // It is the pointer where to save its context
 *  if(res == 1)
 *      longjmp(&tc2, 2);
 *	else if(res == 2)
 *	    longjmp(&tc1, 1);
 *
 *  Note that each thread calls the longjmp passing its id to the function to
 *  allow the manager to select the next thread to run. 
 *  No interrupts are involved.
 */
 

.align  4
.global lamp_setjmp
.type   lamp_setjmp, @function

lamp_setjmp:
    // a0 contains the address of the env to be saved.

    // Store all the callee-saved registers.
    sw s0,   (8*4)(a0)  // fp, frame pointer
    sw s1,   (9*4)(a0)
    sw s2,  (18*4)(a0)
    sw s3,  (19*4)(a0)
    sw s4,  (20*4)(a0)
    sw s5,  (21*4)(a0)
    sw s6,  (22*4)(a0)
    sw s7,  (23*4)(a0)
    sw s8,  (24*4)(a0)
    sw s9,  (25*4)(a0)
    sw s10, (26*4)(a0)
    sw s11, (27*4)(a0)
    sw ra,   (1*4)(a0)  // ra, return address
    sw sp,   (2*4)(a0)  // sp, stack pointer

    li a0, 0
    ret


.align  4
.global lamp_longjmp
.type   lamp_longjmp, @function

lamp_longjmp:
	// a0 contains the address of the env to be restored.
	// a1 contains the return value to be used later.

    // Load all the callee-saved registers.
    lw s0,   (8*4)(a0)  // fp, frame pointer
    lw s1,   (9*4)(a0)
    lw s2,  (18*4)(a0)
    lw s3,  (19*4)(a0)
    lw s4,  (20*4)(a0)
    lw s5,  (21*4)(a0)
    lw s6,  (22*4)(a0)
    lw s7,  (23*4)(a0)
    lw s8,  (24*4)(a0)
    lw s9,  (25*4)(a0)
    lw s10, (26*4)(a0)
    lw s11, (27*4)(a0)
    lw ra,   (1*4)(a0)  // ra, return address
    lw sp,   (2*4)(a0)  // sp, stack pointer

    // Move the return value into the right register (a0).
    // This is performed at the end of the longjmp
    // otherwise we lose the environment pointer.
    // If a1 contains 0, return 1. Otherwise, return
    // the passed value held by a1.
    li   a0, 1
    beqz a1, longjmp_ret
    mv   a0, a1

longjmp_ret:
    ret
