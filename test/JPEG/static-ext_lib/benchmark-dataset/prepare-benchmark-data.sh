#!/bin/bash

echo "Downloading small images dataset..."
wget http://images.cocodataset.org/zips/val2017.zip
echo "Unzipping images..."
unzip val2017.zip

echo "Downloading large images dataset..."
wget http://imagecompression.info/test_images/rgb16bit.zip
echo "Unzipping images..."
unzip rgb16bit.zip -d rgb16bit
cd rgb16bit

echo "Converting images..."
mogrify -format jpg *.ppm

rm *.ppm *.txt
