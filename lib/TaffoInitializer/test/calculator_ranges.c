#include <stdio.h>

#define DO(op)                                                              \
  {                                                                         \
    float __attribute__((annotate("no_float 16 16 signed -100 100"))) tmp1; \
    float __attribute__((annotate("no_float 16 16 signed -200 200"))) tmp2; \
    tmp2 = stack[--sp];                                                     \
    tmp1 = stack[--sp];                                                     \
    stack[sp++] = tmp1 op tmp2;                                             \
  }

int main(int argc, char* argv[]) {
  float tmp;
  char buffer[256];
  float __attribute__((annotate("no_float 8 24 unsigned -1000 1000 1.0e-5"))) stack[32];
  int sp = 0;

  while (1) {
    scanf("%s", buffer);
    int n = sscanf(buffer, "%f", &tmp);
    if (n == 1) {
      stack[sp++] = tmp;
    }
    else {
      switch (buffer[0]) {
      case '+':
        DO(+);
        break;
      case '-':
        DO(-);
        break;
      case '*':
        DO(*);
        break;
      case '/':
        DO(/);
        break;
        /*
      case '%':
        stack[sp-2] = stack[sp-1] % stack[sp-2];
        sp--;
        break;*/
      case '=':
        printf("%f\n", stack[--sp]);
        break;
      case 'q':
        return 0;
      }
    }
  }
  return 0;
}
