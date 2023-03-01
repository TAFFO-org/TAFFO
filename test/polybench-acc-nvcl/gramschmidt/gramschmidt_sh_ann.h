#ifndef SYR2K_SH_ANN_H
#define SYR2K_SH_ANN_H

#define ANN_A __attribute__((annotate("target('a') scalar(range(-50, 50) final) bufferid('a')")))
#define ANN_R __attribute__((annotate("target('r') scalar(range(-31, 31) final) bufferid('r')")))
#define ANN_Q __attribute__((annotate("target('q') scalar(range(-10, 10) final) bufferid('q')")))

#endif
