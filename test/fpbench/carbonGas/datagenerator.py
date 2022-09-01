#!/usr/bin/env python
from pip import main



from random import random
M = 10000


def randomFloat(min, max):
     return min + (random() * (max - min))



if __name__ == '__main__':
    print("static float arr[] = {")
    for i in range(M):
        s = str(randomFloat(0,1)) + ", "
        if i == M-1:
            s = s[:-2]
        print(s)    
    print("};\n")
    