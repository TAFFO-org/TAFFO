#ifndef COVARIANCE_SH_ANN_H
#define COVARIANCE_SH_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(0,1000) final) target('') bufferid('a')")))
#define ANN_X __attribute__((annotate("scalar(range(0,200) final) target('') bufferid('x')")))
#define ANN_Y __attribute__((annotate("scalar(range(0,200) final) target('') bufferid('y')")))
#define ANN_W __attribute__((annotate("scalar(range(0,60000) final) target('') bufferid('w')")))
#define ANN_Z __attribute__((annotate("scalar(range(0,2) final) target('') bufferid('z')")))
#define ANN_V1 __attribute__((annotate("scalar(range(0,2) final) target('') bufferid('v1')")))
#define ANN_V2 __attribute__((annotate("scalar(range(0,2) final) target('') bufferid('v2')")))
#define ANN_U1 __attribute__((annotate("scalar(range(0,5000) final) target('') bufferid('u1')")))
#define ANN_U2 __attribute__((annotate("scalar(range(0,2) final) target('') bufferid('u2')")))
#define ANN_ALPHA __attribute__((annotate("scalar(range(1,2) final) target('') bufferid('alpha')")))
#define ANN_BETA __attribute__((annotate("scalar(range(1,2) final) target('') bufferid('beta')")))

#endif
