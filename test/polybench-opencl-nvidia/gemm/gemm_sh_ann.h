#ifndef COVARIANCE_SH_ANN_H
#define COVARIANCE_SH_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('a')")))
#define ANN_B __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('b')")))
#define ANN_C __attribute__((annotate("scalar(range(0,60000) final) target('') bufferid('c')")))
#define ANN_ALPHA __attribute__((annotate("scalar(range(0,500) final) target('') bufferid('alpha')")))
#define ANN_BETA __attribute__((annotate("scalar(range(0,50) final) target('') bufferid('beta')")))


#endif
