#ifndef FDTD2D_SH_ANN_H
#define FDTD2D_SH_ANN_H

#define ANN_EX __attribute__((annotate("target('ex') scalar(range(-32768, 32767) final)")))
#define ANN_EY __attribute__((annotate("target('ey') scalar(range(-32768, 32767) final)")))
#define ANN_HZ __attribute__((annotate("target('hz') scalar(range(-32768, 32767) final)")))
#define ANN_FICT __attribute__((annotate("scalar(range(0, 1000) final)")))

#endif
