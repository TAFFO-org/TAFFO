///TAFFO_TEST_ARGS -Xvra -propagate-all

float test(int a)
{
  __attribute((annotate("scalar(range(-32767, 32767))"))) float c[10];
  __attribute((annotate("scalar(range(-32767, 32767))"))) float *b = c;
  b[5] = a;
  return b[5];
}


