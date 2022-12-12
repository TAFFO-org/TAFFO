#ifndef MM_SHARED_ANN_H
#define MM_SHARED_ANN_H

#define ANN_ALPHA __attribute__((annotate("scalar()"))) 
#define ANN_BETA __attribute__((annotate("scalar()"))) 
#define ANN_TMP __attribute__((annotate("scalar(range(-16384, 16384) final)")))
#define ANN_A __attribute__((annotate("scalar()"))) 
#define ANN_B __attribute__((annotate("scalar()"))) 
#define ANN_C __attribute__((annotate("scalar()"))) 
#define ANN_D __attribute__((annotate("target('D') scalar(range(-16384, 16384) final)")))

#endif