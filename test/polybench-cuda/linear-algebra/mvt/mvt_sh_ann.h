#ifndef MVT_SH_ANN_H
#define MVT_SH_ANN_H

#define ANN_A __attribute__((annotate("target('a') scalar(range(0, 1) final) bufferid('a')")))
#define ANN_X1 __attribute__((annotate("target('x1') scalar(range(0, 6000) final) bufferid('x1')")))
#define ANN_X2 __attribute__((annotate("target('x2') scalar(range(0, 6000) final) bufferid('x2')")))
#define ANN_Y_1 __attribute__((annotate("target('y_1') scalar(range(0, 1) final) bufferid('y_1')")))
#define ANN_Y_2 __attribute__((annotate("target('y_2') scalar(range(0, 1) final) bufferid('y_2')")))

#endif