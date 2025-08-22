// This test checks if annotations of classes and structs compile and if padding detection works (also on unnamed types)

#include <iostream>

class MyClass {
public:
  int i = 0;
  double d = 0.0;
  bool b = false;
  // padding here
  // unnamed struct
  struct {
    int i1 = 0;
    float f1 = 0.0f;
    bool b1 = false;
    // padding here
  } obj;
};

int main() {
  __attribute__((
    annotate("struct[void, scalar(range(0, 100)), void, struct[void, scalar(range(0, 0)), void]]"))) MyClass myObj;
  std::cout << "Values Begin\n";
  while (myObj.d < 100) {
    myObj.d += 0.5;
    std::cout << myObj.d << "\n";
  }
  std::cout << "Values End\n";
  return 0;
}
