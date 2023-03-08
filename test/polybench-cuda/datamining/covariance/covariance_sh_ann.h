#ifndef COVARIANCE_SH_ANN_H
#define COVARIANCE_SH_ANN_H

#define ANN_DATA __attribute((annotate("target('') scalar(range(0, 2) final) bufferid('data')")))
#define ANN_MEAN __attribute((annotate("target('') scalar(range(0, 1000) final) bufferid('mean')")))
#define ANN_SYMMAT __attribute((annotate("target('cov') scalar(range(-32768, 32767) final) bufferid('symmat')")))
#define ANN_FLOAT_N __attribute((annotate("scalar(range(3214212.01, 3214212.01)) bufferid('float_n')")))

#endif
