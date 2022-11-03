import random
import sys

count = int(sys.argv[1])
random.seed(42)
# numbers = [random.uniform(-1, 10) for i in range(count)]
r = random.random() + 1
numbers = [r for i in range(count)]
for n in numbers:
    print(f"{str(n)}")
