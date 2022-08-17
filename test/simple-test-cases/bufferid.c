///TAFFO_TEST_ARGS
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  __attribute__((annotate("scalar(range(-100, 100)) bufferid('buf1')"))) int buffer1[100];
  __attribute__((annotate("scalar(range(-10000, 10000)) bufferid('buf1')"))) int buffer2[100];

  for (int i=0; i<100; i++) {
    int *pbuf1 = buffer1+i;
    int *pbuf2 = &(buffer2[i]);
    *pbuf1 = (float)i / 100;
    *pbuf2 = (float)i / 100;
  }

  int n = 0;
  for (int i=0; i<100; i++) {
    if (buffer1[i] == buffer2[i])
      n++;
  }
  printf("%d\n", n);

  return 0;
}
