/**
 * lamp_setjmp.h provides non-local jumps. The control flow deviates from
 * the usual subroutine call and return sequence. The complementary fuctions
 * lamp_setjmp and lamp_longjmp provide this functionality.
 */


#ifndef LAMP_SETJMP_H
#define LAMP_SETJMP_H

#include "lamp_core.h"


typedef struct {
	uint32_t env[32];       /**< Register file status. */  
	uint32_t stack[512];    /**< Stack of the thread. */
} tc_t;


/**
 * Set up a tc_t structure and initialize it for the jump. This routine
 * saves the program's calling environment in the environment buffer
 * structure specified by the given pointer 'tc'.
 * \param tc the pointer to an environment buffer structure that stores
 * the program's calling environment.
 * \return 0, if the return is from a direct invocation. Nonzero value if
 * the return is from a call to 'lamp_longjmp'.
 */
int lamp_setjmp(tc_t *tc);

/**
 * Restore the context of the environment buffer 'tc' that was previously
 * saved by invoking the 'lamp_setjmp' routine. After the longjmp is
 * completed, the program execution continues as if the corresponding
 * invocation of setjmp had just returned. If the 'ret_val' passed to longjmp
 * is 0, setjmp will behave as if it has returned 1; otherwise, it behaves
 * as if it has returned 'ret_val'.
 * \param tc the pointer to an environment buffer structure holding the
 * context to restore.
 * \param ret_val a value passed from the longjmp to the setjmp.
 * \return nothing.
 */
void lamp_longjmp(tc_t *tc, int ret_val);

#endif