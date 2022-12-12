#ifndef CORRELATION_SHARED_ANN_H
#define CORRELATION_SHARED_ANN_H

#define ANN_DATA __attribute__((annotate("scalar(range(-2,2) final) target('') bufferid('data')")))
#define ANN_MEAN __attribute__((annotate("scalar(range(0,10000) final) target('') bufferid('mean')")))
#define ANN_STD __attribute__((annotate("scalar(range(-10000,10000) final) target('') bufferid('std')")))
#define ANN_SYMMAT __attribute__((annotate("scalar(range(-2,2) final) target('') bufferid('symmat')")))
#define ANN_FLOAT_N __attribute__((annotate("scalar(range(0,4000000)) bufferid('float_n')")))
#define ANN_EPS __attribute__((annotate("scalar(range(0,1)) bufferid('eps')")))

#endif
