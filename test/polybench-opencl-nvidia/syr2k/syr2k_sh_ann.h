#ifndef SYR2K_SH_ANN_H
#define SYR2K_SH_ANN_H

#define ANN_ALPHA __attribute__((annotate("target('alpha') scalar(range(32412, 32412) final) bufferid('alpha')")))
#define ANN_BETA __attribute__((annotate("target('beta') scalar(range(2123, 2123) final) bufferid('beta')")))
#define ANN_A __attribute__((annotate("target('a') scalar(range(0, 1) final) bufferid('a')")))
#define ANN_B __attribute__((annotate("target('b') scalar(range(0, 1) final) bufferid('b')")))
#define ANN_C __attribute__((annotate("target('c') scalar(range(0, 22053158) final) bufferid('c')")))

#endif
