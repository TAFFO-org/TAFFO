///TAFFO_TEST_ARGS -disable-vra
#include <stdio.h>

int main(int argc, char *argv[])
{
  __attribute__((annotate("range 0 4"))) double magic = 1.234567890123456789;
  printf("%a\n", magic + 2.3456778912345678);
}

