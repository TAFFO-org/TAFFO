//=============================================================================
// FILE:
//      input_for_hello.c
//
// DESCRIPTION:
//      Sample input file for HelloWorld and InjectFuncCall
//
// License: MIT
//=============================================================================
float foo(float a) {
  float boo = 2;
  float arr[] = {1.9, 1.23, 3.32, 2.52, 7.42, 8.91, 6.54};
  while (a > 1) {
    boo *= 2.1;
    boo += arr[((int)a) % 7];
    a /= 2.0;
  }
  return boo * 2;
}

float bar(int a, int b) {
  return (a + foo(b) * 2.4);
}

float fez(int a, int b, int c) {
  return (a + bar(a, b) * 2.1 + c * 3);
}

int main(int argc, char *argv[]) {
  int a = 123;
  float ret = 0;

  ret += foo(a);
  ret += bar(a, ret);
  ret += fez(a, ret, 123);

  if (ret / 2 > 10) ret += 1;

  return ret;
}
