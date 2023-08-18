#ifndef ADI_SHARED_ANN_H
#define ADI_SHARED_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(-2,2) final) target('') bufferid('a')")))
#define ANN_B __attribute__((annotate("scalar(range(0,4) final) target('') bufferid('b')")))
#define ANN_X __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('x')")))

#endif