///TAFFO_TEST_ARGS -Xvra -propagate-all


float fpextfptrunc(__attribute((annotate("scalar(range(-32767, 32767))"))) float a, __attribute((annotate("scalar(range(-32767, 32767))"))) double b)
{
  __attribute((annotate("scalar(range(-32767, 32767))"))) double c = 123.0;
  c += a;
  return (float)c + b;
}

