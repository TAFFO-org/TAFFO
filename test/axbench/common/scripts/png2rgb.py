'''
Created on Apr 10, 2011

@author: Hadi Esmaeilzadeh <hadianeh@cs.washington.edu>
'''

import png
import sys
import csv

def png2rgb(file):
    r=png.Reader(file);
    img = r.asRGB()
    print(file, img)
    return img
pass

def rgb2png(img, file, depth=8):
    f = open(file, 'wb')
    pngWriter = png.Writer(img[0], img[1], bitdepth=depth, greyscale=False)
    pngWriter.write(f, img[2])
    f.close()
pass

def rgb2gray(img):
    r = 0.30
    g = 0.59
    b = 0.11
    
    pixels = []
    for row in img[2]:
        grayRow = []
        for i in range(0, len(row), 3):
            luminance = int(r * row[i] + g * row[i+1] + b * row[i+2] + 0.5) % 256
            for j in range(3): grayRow.append(luminance)
        pass
        pixels.append(tuple(grayRow))
    pass
            
    return (img[0], img[1], pixels, img[3])        
pass

def rgbsave(img, file):
    f = open(file, 'w')
    f.write(str(img[0]) + ',' + str(img[1]) + '\n')

    pixels = list(img[2])    
    for row in pixels:
        for p in row[:-1]: 
            f.write(str(p) + ',')
        f.write(str(p) + '\n')
    pass
    
    f.write('"' + str(img[3]) + '"')
    f.close()
pass

def rgbload(file):
    csvReader = csv.reader(open(file, 'r'), delimiter=',', quotechar='"')   
    
    i = 0
    pixels = []
    width = 0
    height = 0
    meta = {}
    for row in csvReader:
        if (i == 0):
            width = int(row[0])
            height = int(row[1])
            print(width, height)
        elif (i == height + 1):
            meta = row[0]
            print(meta)
            break
        else: 
            row = [int(e) for e in row]
            pixels.append(tuple(row))
        pass
        
        i = i + 1
    pass
    
    print(width, height, meta)
    return(width, height, pixels, meta)
pass



if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print('Error: Oops! Too few arguments!')
        print('Usage: ' + sys.argv[0] + ' OPERATION INPUT_FILE OUTPUT_FILE')
        exit(-1)
    pass

    opr = str(sys.argv[1])
    input = str(sys.argv[2])
    output = str(sys.argv[3])
    
    if (opr == 'rgb'):
        img = png2rgb(input)
        rgbsave(img, output)
    pass

    if (opr == 'png'):
        img = rgbload(input)
        rgb2png(img, output)
    pass

    if (opr == 'png16'):
        img = rgbload(input)
        rgb2png(img, output, 16)
    pass

    if (opr == 'gray'):
        img = png2rgb(input)
        img = rgb2gray(img)
        rgb2png(img, output)
    pass
    
    exit(0)

pass
