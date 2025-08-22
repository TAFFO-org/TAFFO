float global;

float test(float param, int notafloat) {
  int notafloat2;
  float local __attribute((annotate("no_float")));

  local = 2.0;
  local *= param;
  local += notafloat;
  notafloat2 = local;
  return notafloat2;
}

int test2(int a) { return a + 2.0; }
