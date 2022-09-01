#!/bin/bash
cd /home/ilgeco/opt/TAFFO/build
cmake -G Ninja -DCMAKE_PREFIX_PATH=/home/ilgeco/bin/or-tools -DCMAKE_INSTALL_PREFIX=../dist -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=1 -DCMAKE_BUILD_TYPE=Debug ..; cmake --build . --target install -j 10