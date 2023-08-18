#ifndef ATAX_SHARED_ANN_H
#define ATAX_SHARED_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('a')")))
#define ANN_X __attribute__((annotate("scalar(range(0,2) final) target('') bufferid('x')")))
#define ANN_Y __attribute__((annotate("scalar(range(0,200000) final) target('') bufferid('y')")))
#define ANN_TMP __attribute__((annotate("scalar(range(0, 1000) final) target('') bufferid('tmp')")))

#endif
