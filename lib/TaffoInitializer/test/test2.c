double __attribute__((annotate("no_float 32 32 signed -100 100 1e-6"))) global;

float slarti(float __attribute__((annotate("no_float 16 16 unsigned -200 200 1e-5"))) bart,
             float __attribute__((annotate("no_float 16 16 unsigned -300 300 2e-5"))) fast) {
  int pros = 42;
  double __attribute__((annotate("no_float 32 32 signed -1000 1000"))) tetnic;
  tetnic = bart;
  tetnic *= fast;
  tetnic += global;
  return tetnic - pros;
}
