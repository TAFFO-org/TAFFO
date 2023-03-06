#ifndef CORRELATION_SHARED_ANN_H
#define CORRELATION_SHARED_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(-1,1) final) target('a') bufferid('a')")))
#define ANN_B __attribute__((annotate("scalar(range(-1,1) final) target('b') bufferid('b')")))

#endif
