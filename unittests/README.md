 # TAFFO Unit Tests

A set of unit tests based on the Google Test framework.

Following the same structure of the TAFFO pipeline, there is a test suite for each of the 5 LLVM passes and a suite for the utilities:

- [InitializerTest](#InitializerTest)
- [RangeAnalysisTest](#RangeAnalysisTest)
- [DataTypeAllocTest](#DataTypeAllocTest)
- [ConversionTest](#ConversionTest)
- [ErrorAnalysisTest](#ErrorAnalysisTest)
- [TaffoUtilsTest](#TaffoUtilsTest)

## Adding new tests

Let's say we have implemented a `NewPass` and we want to test its component(s): the directory structure will look something like this:

```
.
├── lib
│   ├── NewPass
│   |   ├── Subfolder
│   │   |   ├── NewComponent.cpp
│   │   |   ├── NewComponent.h
│   │   |   └── CMakeLists.txt
│   |   └── CMakeLists.txt
|   └── CMakeLists.txt
|
├── unittests
│   ├── NewPass
│   |   ├── Subfolder
│   │   |   ├── NewComponentTest.cpp
│   │   |   └── CMakeLists.txt
│   |   └── CMakeLists.txt
│   ├── [...]
│   │
│   └── CMakeLists.txt
│
└── [...]
```

To correctly detect and run the tests, we have to edit the following CMakeLists.txt files:

- `lib/NewPass/Subfolder/CMakeLists.txt` 

```cmake
set(SELF NewPass)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
    NewComponent.cpp

    ADDITIONAL_HEADERS
    NewComponent.h
)

target_include_directories(obj.${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
```

- `lib/CMakeLists.txt`
```cmake
add_subdirectory(NewPass)

add_llvm_library(Taffo MODULE
    $<TARGET_OBJECTS:obj.NewPass>

    [...]
)
```

- `unittests/NewPass/Subfolder/CMakeLists.txt`

```cmake
set(SELF NewPassTest)
taffo_add_unittests(${SELF} ${SELF}
    NewComponentTest.cpp
    [...]
)
include_directories(${CMAKE_SOURCE_DIR}/lib/NewPass)
target_link_libraries(NewPassTests PRIVATE TaffoUtils obj.NewPass [...])
```

The test file (`unittests/NewPass/Subfolder/NewComponentTest.cpp`) will have the following structure:

```cpp
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"

#include "Subfolder/NewComponent.h"

namespace {
    using namespace llvm;

class NewComponentTest : public testing::Test {
protected:
    /* Attributes to build the testing environment for the components */
    LLVMContext Context;
    Constant *ConstantV;

    NewComponentTest() :
        ConstantV(ConstantInt::get(Type::getInt32Ty(Context), 0))
        {}

    /* Internal methods */
};

/* Actual tests */

TEST_F(NewComponentTest, MockTest) {
    ASSERT_TRUE(true);
}

}
```


## Running the tests

Once we created our test, we can compile and run the test suite with
```sh
mkdir build
cd build
cmake .. -DTAFFO_BUILD_ORTOOLS=ON
cmake --build .
ctest -VV
```


# Implemented tests

## InitializerTest

## RangeAnalysisTest

## DataTypeAllocTest

## ConversionTest

## ErrorAnalysisTest

## TaffoUtilsTest
