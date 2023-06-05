#ifndef SYR2K_SH_ANN_H
#define SYR2K_SH_ANN_H

#define ANN_A __attribute__((annotate("target('a') scalar(range(-1, 1) final) bufferid('a')")))
#define ANN_B __attribute__((annotate("target('b') scalar(range(-100, 100) final) bufferid('b')")))

#endif
