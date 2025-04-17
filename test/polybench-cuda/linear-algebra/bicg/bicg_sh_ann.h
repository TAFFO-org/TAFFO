#ifndef BICG_SHARED_ANN_H
#define BICG_SHARED_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('a')")))
#define ANN_R __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('r')")))
#define ANN_S __attribute__((annotate("scalar(range(0,2000) final) target('') bufferid('s')")))
#define ANN_P __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('p')")))
#define ANN_Q __attribute__((annotate("scalar(range(0,2000) final) target('') bufferid('q')")))

#endif
