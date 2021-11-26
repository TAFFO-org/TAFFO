# TAFFO Metadata Format

## Generic Input Information Format

The input information associated to a variable takes the form of a metadata node formatted as follows:
 
```
!n = !{type, range, initialError, convertible}
```

Each item of the metadata tuple can be either a link to another metadata tuple or the integer zero (`i1 false`) in case a specific information does not apply.

### ``type`` Subnode

The ``type`` subnode, if present, specifies the target type of the variable after Taffo has completed its conversion to a different format.

```
!n = !{typeFlag, ...}
```

The following items are used to further specialize the type, depending on the value of typeFlag (a `MDString`).

#### ``fixp`` Type Flag

```
!n = !{!"fixp", i32 Width, i32 FracBits}
```

A fixed point type where:

- abs(Width) is the bit width of the fixed point format.
  Width > 0 if the format is unsigned,
  Width < 0 if the format is signed.
- FracBits is the number of bits to the right of the dot of the fixed point format.

### ``range`` Subnode

The ``range`` subnode, which must be present, specifies the range of values admitted for the variable.

```
!n = !{double Min, double Max}
```

- Min is the lower bound of the range.
- Max is the upper bound of the range.

### ``initialError`` Subnode

The ``initialError`` subnode, if present, specifies a priori an error value for a variable. This error is initially assumed to be correct and is used to kickstart the error propagation process.

```
!n = !{double Error}
```

### ``convertible`` Subnode

The ``convertible`` subnode is always present, and it specifies if the value being annotated should be processed by Conversion in order to change its type. It is always a boolean constant.

### Associating Input Information to IR nodes

An input information metadata can be associated to either an Instruction, the parameters of a function, or to a GlobalObject. 

#### Input Information on Non-Struct Instructions and GlobalObjects

The input information is attached to the instruction using the ``taffo.info`` metadata label.

**Example:**

```
@a = global i32 5, align 4, !taffo.info !0
[...]
%add = add nsw i32 %0, %1, !taffo.info !5
```

And, at the end of the file,

```
!0 = !{!1, !2, !3}
!1 = !{!"fixp", i32 32, i32 5}
!2 = !{double 2.0e+01, double 1.0e+2}
!3 = !{double 1.000000e-02}
[...]
!5 = !{!6, !7, 0}
!6 = !{!"fixp", i32 32, i32 5}
!7 = !{double 2.0e+01, double 1.0e+2}
```

**Related functions:**

```cpp
#include "ErrorPropagator/MDUtils/Metadata.h"

InputInfo* MetadataManager::retrieveInputInfo(const Instruction &I);
static void MetadataManager::setInputInfoMetadata(Instruction &I, const InputInfo &IInfo);

InputInfo* MetadataManager::retrieveInputInfo(const GlobalObject &V);
static void MetadataManager::setInputInfoMetadata(GlobalObject &V, const InputInfo &IInfo);
```

The functions above use the `InputInfo` struct for storing the data to be converted to metadata.
It contains the following public fields:

- ` TType *IType;`: this is a shared pointer to an object describing the type of the instruction.
  The only supported type is currently `FPType`, which describes a fixed point type, and is a subclass of `TType`.
  For further information on the member fields and functions of `FPType`, please refer to file `InputInfo.h`.
- `Range *IRange;`: this is a shared pointer to an instance of the `Range` struct, which simply contains the lower and upper bounds of the variable's range as `double` fields.
- `double *IError;`: a shared pointer to a `double` containing the initial error for the current instruction/global variable.

All of these fields are optional: if they are assigned the value `nullptr`, the TAFFO `MetadataManager` will emit `i1 false` for them.

**Example:**

```cpp
...
shared_ptr<FPType> Ty(new FPType(32U, 18U, false));
shared_ptr<Range> R(new FPType(1.0, 1.0e+02));
shared_ptr<double> Err(new double(1.0e-8));
InputInfo II(Ty, R, Err);
MetadataManager::setInputInfoMetadata(Instr, II);
```

The piece of code above associates the range [1.0, 100] and the initial error 10^-8 to instruction Instr,
which will be treated as an unsigned fixed point value with 18 fractionary bits and 12 integer bits (for a total width of 32 bits).

Since the fields of `InputInfo` are shared pointers, there is no need to worry about their ownership (this allows InputInfo to be used as the root owner of these objects, if needed).
The objects referred to by such pointers are cached into a singleton `MetadataManager` instance that can be retrieved with the `static MetadataManager& getMetadataManager()` function,
so that if multiple variables have the same range, type or initial error, a pointer to the same object is returned.

#### Input Information of Struct-Typed Instructions and GlobalObjects

In the case of structure (record) values, the input information metadata is associated to the elements of the structure. Each input information metadata is collected, in the same order of the struct members, as a metadata tuple tagged to the value as `!taffo.structinfo`.

Structure elements which shall not have input information metadata are associated to the `i1 0` metadata constant instead.
In case of nested structures, fields that have a structure type are associated to another metadata tuple containing field-ordered data, forming a recursive structure.

**Example:**

```
%struct.spell = type { float, i32 }
...
%spell.addr = alloca %struct.spell*, align 8, !taffo.structinfo !0
```

And, at the end of the file,

```
!0 = !{!1, i1 0}
!1 = !{!2, !3, !4}
!2 = !{!"fixp", i32 32, i32 5}
!3 = !{double 2.0e+01, double 1.0e+2}
!4 = !{double 1.000000e-02}
```

In this case, the i32 element of the structure does not have input information, while the float element does and is stored in the `!1` node.

**Related Functions:**

```cpp
#include "ErrorPropagator/MDUtils/Metadata.h"

StructInfo* retrieveStructInfo(const Instruction &I);
static void setStructInfoMetadata(Instruction &I, const StructInfo &SInfo);

StructInfo* retrieveStructInfo(const GlobalObject &V);
static void setStructInfoMetadata(GlobalObject &V, const StructInfo &SInfo);
```
These functions store data into `StructInfo` objects.
`StructInfo` objects contain a set of pointers to `MDInfo` objects
ordered in the same way as the fields of the corresponding struct sype.
A `MDInfo` boject can either be another `StructInfo` instance,
in case the corresponding field is itself a struct,
or an instance of `InputInfo`, for regular fields.
If a pointer to a `MDInfo` object is null, it means there is no data
for the corresponding field.

**Example:**

```cpp
...
Instruction I = ...;
FPType FPT(32, 12);
Range R(0,42);
double Err = 1e-8;
InputInfo IInfo(&FPT, &R, &Err);
MDInfo * S1[] = { nullptr, &IInfo };
StructInfo SI1(S1);
MDInfo * S2[] = { &IInfo, nullptr, &SI1 };
StructInfo SI2(S2);

MetadataManager::setStructInfoMetadata(I, SI2);
```

In the piece of code above, Instruction `I` could be of a type of this kind:

```cpp
struct {
  int FixedPointField1;
  any_type1 SomethingElse1;
  struct {
    any_type2 SomethingElse2;
    int FixedPointField2;
  } StructField;
};
```
The data contained in `IInfo` is attached to fields `FixedPointField1`
and `FixedPointField2`, while no data is associated to fields
`SomethingElse1` and `SomethingElse2`.


#### Input Information of Function Arguments

The input information metadata for all the arguments (whether struct-typed or not) is collected, in the same order of the arguments, as a metadata tuple, which is then tagged to the function as `!taffo.funinfo`.

For each argument, two consecutive items in the metadata tuple are allocated:

1. An `i32`-typed identifier specifying the type of the input info metadata:
	- `0` if the argument is not associated to any input info (in this case, the following tuple element is ignored)
	- `1` for non-struct argument input info (corresponding to `!taffo.inputinfo`)
	- `2` for struct argument input info (corresponding to `!taffo.structinfo`)
2. A reference to the metadata tuple of the argument

**Example:**

```
define i32 @slarti(i64 %x, {float, i32} %y, i32 %n) #0 !taffo.funinfo !2 {
[...]
```

at the end of the file:

```
!2 = !{i32 1, !3, i32 2, !15, i32 1, !5} ; !3 = %x, !4 = %y, !5 = %n

!3 = !{!6, !12, !7}
!6 = !{!"fixp", i32 -64, i32 10}
!12 = !{double 1.0e+00, double 5.0e+00}
!7 = !{double 1.000000e-03}

!15 = !{!4, i1 0}
!4 = !{!8, !13, !9}
!8 = !{!"fixp", i32 -64, i32 10}
!13 = !{double -5.0e+00, double 6.0e+00}
!9 = !{double 2.000000e-03}

!5 = !{!10, !14, !11}
!10 = !{!"fixp", i32 32, i32 0}
!14 = !{double 0.0e+00, double 2.0e+01}
!11 = !{double 0.000000e+01}
```

**Related functions:**

```cpp
#include "ErrorPropagator/MDUtils/Metadata.h"
void MetadataManager::retrieveArgumentInputInfo(const Function &F, SmallVectorImpl<MDInfo *> &ResII);
static void MetadataManager::setArgumentInputInfoMetadata(Function &F, const ArrayRef<MDInfo *> AInfo);
```

The `MDInfo` instances for these functions can be created in the same way as for Instructions and Global Variables. They can be either instances of InputInfo or StructInfo, depending on the type of each function argument.
The `MetadataManager::setArgumentInputInfoMetadata` function expects an array of pointers to `MDInfo` instances, one for each formal parameter, and in the same order.
The `MetadataManager::retrieveArgumentInputInfo` function populates vector `ResII` with pointers to an `MDInfo` object for each argument.
Again, `MetadataManager` takes ownership of the `MDInfo` pointers it returns, but does not gain ownership of those it receives.

## ErrorPropagation Specific Metadata

### Input Data

#### Target instructions

Instructions and global variables may be associated to a target variable in the original program.
Taffo will keep track of the errors propagated for all instructions that are associated to each target,
and display the maximum absolute error computed for that target at the end of the pass.
Targets may be specified with a MDString node containing the name of the target,
with lable `!taffo.target`.
Each instruction/global variable may be associated with a single target.

```
  ...
  %OptionPrice = alloca float, align 4, !taffo.info !17, !taffo.target !19
  ...
  store float %sub51, float* %OptionPrice, align 4, !taffo.info !17, !taffo.target !19
  ...
```

At the end of the file:

```
!19 = !{!"OptionPrice7"}
```
(The name of the target is OptionPrice7).

Related functions:

```cpp
#include "MDUtils/Metadata.h"
static void MetadataManager::setTargetMetadata(Instruction &I, StringRef Name);
static void MetadataManager::setTargetMetadata(GlobalObject &V, StringRef Name);
static Optional<StringRef> MetadataManager::retrieveTargetMetadata(const Instruction &I);
static Optional<StringRef> MetadataManager::retrieveTargetMetadata(const GlobalObject &V);
```

#### Starting point

A starting point for error propagation is a function whose errors will be propagated,
and for which the output error metadata will be emitted.
Functions not marked as starting points will not be propagated,
unless they are called in marked functions.
A function may be marked as a starting point by attaching to it a metadata node with any content,
labeled with `!taffo.start`.

```
define float @_Z4CNDFf(float %InputX) #0 !taffo.start !2 {
  ...
```
At the end of the file:
```
!2 = !{i1 true}
```

Related functions:

```cpp
#include "MDUtils/Metadata.h"
static void MetadataManager::setStartingPoint(Function &F);
static bool MetadataManager::isStartingPoint(const Function &F);
```

#### Constant Info

This kind of metadata is needed by the TAFFO error propagator to keep track
of value ranges and type data of converted constants.
It is kept as a list of input information metadata tuples,
attached to instructions containing constant operands,
in the same order of the latter.
For non-constant operands, `i1 flase` is used as a placeholder.
```
%7 = mul i64 %6, 1518500249, !taffo.constinfo !74
```
At the end of the file:
```
!74 = !{i1 false, !75}
!75 = !{!76, !77, i1 false, i2 0}
!76 = !{!"fixp", i32 32, i32 31}
!77 = !{double 0x3FE6A09E667F3BCC, double 0x3FE6A09E667F3BCC}
```

Related functions:

```cpp
#include "MDUtils/Metadata.h"
void retrieveConstInfo(const llvm::Instruction &I, llvm::SmallVectorImpl<InputInfo *> &ResII);
static void setConstInfoMetadata(llvm::Instruction &I, const llvm::ArrayRef<InputInfo *> CInfo);
```

#### Loop unroll count

An i32 integer constant attached to the terminator instruction of the loop header block,
with label `!taffo.unroll`.

```
for.body:                                         ; preds = %entry, %for.body
  ...
  br i1 %exitcond, label %for.body, label %for.end, !taffo.unroll !8
```

At the end of the file:

```
!8 = !{i32 20}
```

for an unroll count of 20.
Related functions:

```cpp
#include "MDUtils/Metadata.h"
static void MetadataManager::setLoopUnrollCountMetadata(llvm::Loop &L, unsigned UnrollCount);
static llvm::Optional<unsigned> MetadataManager::retrieveLoopUnrollCount(const llvm::Loop &L);
```

#### Max Recursion Count

The maximum number of recursive calls to this function allowed.
Represented as an i32 constant, labeled `!taffo.maxrec`.

```
define i32 @foo(i32 %x, i32 %y) !taffo.info !0 !taffo.maxrec !9 {
  ...
```

At the end of the file:

```
!9 = !{i32 15}
```

Related functions:

```cpp
#include "MDUtils/Metadata.h"
static void MetadataManager::setMaxRecursionCountMetadata(llvm::Function &F, unsigned MaxRecursionCount);
static unsigned MetadataManager::retrieveMaxRecursionCount(const llvm::Function &F);
```

### Output data

#### Computed absolute error

`double` value attached to each instruction with label `!taffo.abserror`.

```
%add = add nsw i32 %x, %y, !taffo.info !4, !taffo.abserror !5
```

At the end of the file:

```
!5 = !{double 2.500000e-02}
```

Related functions:

```cpp
#include "MDUtils/Metadata.h"
static void MetadataManager::setErrorMetadata(llvm::Instruction &, const AffineForm<inter_t> &);
static double MetadataManager::retrieveErrorMetadata(const Instruction &I);
```

#### Possible wrong comparison

If a cmp instruction may give a wrong result due to the absolute error of the operands,
this is signaled with `!taffo.wrongcmptol`.
The absolute tolerance computed for the comparison is attached to this label.
The relative threshold above which the comparison error is signaled can be set form command line.

```
%cmp = icmp slt i32 %a, %b, !taffo.wrongcmptol !5
```

End of file:

```
!5 = !{double 1.000000e+00}
```

Related functions:

```cpp
#include "MDUtils/Metadata.h"
static void MetadataManager::setCmpErrorMetadata(Instruction &I, const CmpErrorInfo &CEI);
static std::unique_ptr<CmpErrorInfo> MetadataManager::retrieveCmpError(const Instruction &I);
```