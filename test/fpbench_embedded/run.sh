#! /bin/bash

if [ ! -f main.o ]; then
    exit -1
fi

make  

if [  $? -ne 0 ]; then
    exit -1
fi

mv main.bin $1
exit 0



