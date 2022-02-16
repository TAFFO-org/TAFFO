#!/usr/bin/env python3

import png
import sys
import csv
from os.path import *

def png2rgb(file):
    r=png.Reader(file)
    img = r.asRGB()
    print(file, img)
    return img


def csave(img, file):
    f = open(file, 'w')
    picname = basename(file).split('.')[0]
    print('#include "picture_data.hpp"', file=f)
    print('', file=f)
    
    print('static const unsigned char pix[] = {', file=f)
    pixels = list(img[2])    
    for row in pixels:
        for p in row[:-1]: 
            f.write(str(p) + ',')
        f.write(str(row[-1]) + ',\n')
    print('};', file=f)
    
    print('\nconst t_picture_data pic_%s = {%d, %d, pix};' % (picname, img[0], img[1]), file=f)
    
    f.close()


if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print('Error: Oops! Too few arguments!')
        print('Usage: ' + sys.argv[0] + ' INPUT_FILE OUTPUT_FILE')
        exit(-1)

    input = str(sys.argv[1])
    output = str(sys.argv[2])
    
    img = png2rgb(input)
    csave(img, output)
    
    exit(0)
