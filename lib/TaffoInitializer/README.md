# TAFFO Initializer Pass

This repository contains a LLVM pass responsible for creating the metadata used by the other passes of the TAFFO framework.
The emitted metadata can be controlled for each variable by means of annotations in the following formats:
```
[target:<Name>] (no_float | force_no_float) <IntBits> <FracBits> [<Min> <Max> [<InitialError>]]
```
or
```
[target:<Name>] range <Min> <Max> [<InitialError>]
```
where the parts enclosed in square brackets are optional, those in (... | ...) are mutually exclusive options, and those enclosed in `<...>` are numbers.

- If `target` is specified, the annotated variable is considered a target value by TAFFO Error Propagator, and its absolute error is displayed at the end of that pass, with the <Name> chosen by the user.
- `<IntBits>` and `<FracBits>` are the number of integer and fractional bits of the fixed point type the annotated variable is translated to by the converter pass.
- `<Min>` and `<Max>` are the bounds of the range of the annotated variable. They are double floating point values.
- `<InitialError>` is the initial error of this variable.
- If `range` is specified, the TAFFO conversion pass will not convert this variable to a fixed point type, but this pass will attach to it the range and error info needed by TAFFO Error Propagator.
  These annotations are removed by this pass.
