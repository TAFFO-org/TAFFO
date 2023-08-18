#ifndef MM_SHARED_ANN_H
#define MM_SHARED_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('a')")))
#define ANN_B __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('b')")))
#define ANN_C __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('c')")))
#define ANN_D __attribute__((annotate("scalar(range(0,1) final) target('') bufferid('d')")))
#define ANN_E __attribute__((annotate("scalar(range(0, 200) final) target('') bufferid('e')")))
#define ANN_F __attribute__((annotate("scalar(range(0, 200) final) target('') bufferid('f')")))
#define ANN_G __attribute__((annotate("scalar(range(0,60000) final) target('') bufferid('g')")))

#endif
