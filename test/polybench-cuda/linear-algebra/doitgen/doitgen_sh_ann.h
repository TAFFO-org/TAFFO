#ifndef DOITGEN_SHARED_ANN_H
#define DOITGEN_SHARED_ANN_H

#define ANN_A __attribute__((annotate("scalar(range(0,60000) final) target('') bufferid('a')")))
#define ANN_SUM __attribute__((annotate("scalar(range(0,60000) final) target('') bufferid('sum')")))
#define ANN_C4 __attribute__((annotate("scalar(range(0,100) final) target('') bufferid('c4')")))

#endif