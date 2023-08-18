#ifndef MM_SHARED_ANN_H
#define MM_SHARED_ANN_H

#define ANN_ALPHA __attribute__((annotate("scalar(range(1,2) final) target('') bufferid('alpha')"))) 
#define ANN_BETA __attribute__((annotate("scalar(range(1,2) final) target('') bufferid('beta')"))) 
#define ANN_TMP __attribute__((annotate("scalar(range(0, 500) final) target('') bufferid('tmp')")))
#define ANN_A __attribute__((annotate("scalar(range(0, 1) final) target ('') bufferid('a')"))) 
#define ANN_B __attribute__((annotate("scalar(range(0, 1) final) target ('') bufferid('b')"))) 
#define ANN_C __attribute__((annotate("scalar(range(0, 1) final) target ('') bufferid('c')"))) 
#define ANN_D __attribute__((annotate("scalar(range(0, 60000) final) target('') bufferid('d')")))

#endif