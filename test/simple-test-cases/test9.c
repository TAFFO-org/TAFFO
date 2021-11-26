///TAFFO_TEST_ARGS -Xvra -propagate-all

float oven(int stuff, int baked, float cherry)
{
  float __attribute__((annotate("scalar(range(-32767, 32767))"))) cake = baked + stuff;
  return cake + cherry;
}


