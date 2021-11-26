///TAFFO_TEST_ARGS

typedef struct {
  float g;
  int h;
} def;

typedef struct {
  float a;
  int b;
  def c;
} abc;

int main(int argc, char *argv[])
{
  __attribute__((annotate("target('test') scalar()"))) float t1 = 123.0;
  __attribute__((annotate("backtracking scalar()"))) float t2 = 123.0;
  __attribute__((annotate("backtracking(true) scalar()"))) float t20 = 123.0;
  __attribute__((annotate("backtracking(yes) scalar()"))) float t21 = 123.0;
  __attribute__((annotate("backtracking(false) scalar()"))) float t22 = 123.0;
  __attribute__((annotate("backtracking(no) scalar()"))) float t23 = 123.0;
  __attribute__((annotate("target('test') backtracking scalar()"))) float t3 = 123.0;
  __attribute__((annotate("target('test quote @' quote') scalar()"))) float t30 = 123.0;
  __attribute__((annotate("target('test quote @@ quote') scalar()"))) float t31 = 123.0;
  __attribute__((annotate("scalar(range(-200, 200))"))) float t4 = 123.0;
  __attribute__((annotate("scalar(type(signed 32 8))"))) float t5 = 123.0;
  __attribute__((annotate("scalar(error(0.5))"))) float t6 = 123.0;
  __attribute__((annotate("scalar(error(0.5) range(-200, 200) type(signed 0x1F 07))"))) float t7 = 123.0;
  __attribute__((annotate("struct[scalar(range(-200, 200)), void, struct[scalar(range(-200, 200)), void]]"))) abc t8;
  return 0;
}

