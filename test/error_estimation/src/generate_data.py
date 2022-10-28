import random
import sys

count = int(sys.argv[1])

numbers = [random.uniform(-1, 10) for i in range(count)]
for n in numbers:
    print(f"{n}")
