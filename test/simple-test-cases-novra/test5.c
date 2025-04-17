/// TAFFO_TEST_ARGS -disable-vra

float fpextfptrunc(__attribute((annotate("range -32767 32767"))) float a,
                   __attribute((annotate("range -32767 32767"))) double b) {
  __attribute((annotate("range -32767 32767"))) double c = 123.0;
  c += a;
  return (float) c + b;
}
