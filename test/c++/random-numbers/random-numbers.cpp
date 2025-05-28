#include <iomanip>
#include <iostream>
#include <random>

#define iterations 100

int main() {
  std::mt19937 gen(0);
  std::uniform_real_distribution distA(-3000.0, 3000.0);
  std::uniform_real_distribution distB(-10.0, 10.0);

  for (std::size_t i = 0; i < iterations; i++) {
    __attribute((annotate("scalar(range(-3000, 3000))"))) double a = distA(gen);
    __attribute((annotate("scalar(range(-10, 10))"))) double b = distB(gen);

    double add = a + b;
    double sub = a - b;
    double mul = a * b;

    std::cout << add << "\n" << sub << "\n" << mul << "\n\n";
  }
  return 0;
}
