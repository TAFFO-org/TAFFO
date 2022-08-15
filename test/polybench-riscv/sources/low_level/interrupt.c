#include "interrupt.h"


__attribute__((weak))
void int_default_handler(void) {
    while(1);
}