// This test generates random numbers with the c++ standard library and uses them in simple computations

#include <iomanip>
#include <iostream>
#include <random>

#define ITERATIONS 100

int main() {
  std::mt19937 gen(0);
  std::uniform_real_distribution distA(-3000.0, 3000.0);
  std::uniform_real_distribution distB(-10.0, 10.0);

  std::cout << "Values Begin\n";
  for (std::size_t i = 0; i < ITERATIONS; i++) {
    __attribute__((annotate("scalar(range(-3000, 3000))"))) double a = distA(gen);
    __attribute((annotate("scalar(range(-10, 10))"))) double b = distB(gen);

    double add = a + b;
    double sub = a - b;
    double mul = a * b;

    std::cout << add << "\n" << sub << "\n" << mul << "\n\n";
  }
  std::cout << "Values End\n";
  return 0;
}
