///TAFFO_TEST_ARGS -Xvra -propagate-all

float test(int a)
{
  float c[10];
  __attribute((annotate("force_no_float range -32767 32767"))) float *b = c;
  b[5] = a;
  return b[5];
}


