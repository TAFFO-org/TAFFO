/// TAFFO_TEST_ARGS -Xvra -propagate-all
#include <stdio.h>

int main(int argc, char* argv[]) {
  __attribute__((annotate("scalar()"))) double magic = 1.234567890123456789;

  printf("Values Begin\n");
  printf("%f\n", magic + 2.3456778912345678);
  printf("Values End\n");
}
