#! /bin/bash

if [ ! -f main.o ]; then
    exit -1
fi

make ; mv main.bin $1

exit 0



