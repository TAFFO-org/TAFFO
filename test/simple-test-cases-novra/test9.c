/// TAFFO_TEST_ARGS -disable-vra

float oven(int stuff, int baked, float cherry) {
  float __attribute__((annotate("range -32767 32767"))) cake = baked + stuff;
  return cake + cherry;
}
