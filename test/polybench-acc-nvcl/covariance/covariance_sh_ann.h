#ifndef COVARIANCE_SH_ANN_H
#define COVARIANCE_SH_ANN_H

#define ANN_DATA __attribute((annotate("scalar(range(0, 2) final)")))
#define ANN_MEAN __attribute((annotate("scalar(range(0, 1000) final)")))
#define ANN_SYMMAT __attribute((annotate("target('cov') scalar(range(-32768, 32767) final)")))
#define ANN_FLOAT_N __attribute((annotate("scalar(range(3214212.01, 3214212.01))")))

#endif
