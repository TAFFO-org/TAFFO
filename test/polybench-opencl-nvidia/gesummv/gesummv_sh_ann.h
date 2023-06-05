#ifndef GESUMMV_SH_ANN_H
#define GESUMMV_SH_ANN_H

#define ANN_A __attribute__((annotate("target('a') scalar(range(0, 1) final) bufferid('a')")))
#define ANN_B __attribute__((annotate("target('b') scalar(range(0, 1) final) bufferid('b')")))
#define ANN_X __attribute__((annotate("target('x') scalar(range(0, 1) final) bufferid('x')")))
#define ANN_Y __attribute__((annotate("target('y') scalar(range(0, 76200472) final) bufferid('y')")))
#define ANN_TMP __attribute__((annotate("target('tmp') scalar(range(0, 2000) final) bufferid('tmp')")))
#define ANN_ALPHA __attribute__((annotate("target('alpha') scalar(range(43532, 43532) final) bufferid('alpha')")))
#define ANN_BETA __attribute__((annotate("target('beta') scalar(range(43532, 43532) final) bufferid('beta')")))

#endif
