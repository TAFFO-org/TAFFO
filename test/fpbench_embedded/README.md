# FPBENCH with miosix

### Configuration

Remove comment from Makefile and run.py  and set to the proper path
```  make
#KPATH := path to miosix
```
Set run.py

``` python
SYSROOT = "/opt/arm-miosix-eabi/arm-miosix-eabi/"
```
to the correct toolchain


### Run 

``` bash
./run.py
```

