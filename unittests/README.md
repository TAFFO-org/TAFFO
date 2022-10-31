# TAFFO Unit Tests

A set of unit tests based on the Google Test framework.

Following the same structure of the TAFFO pipeline, there is a test suite for each of the 5 LLVM passes and a suite for
the utilities. Moreover, this directory also contains a [set of utilities](#testutils) to help with the development of the tests.

## Adding new tests

Let's say we have implemented a `NewPass` and we want to test its component(s): the directory structure will look
something like this:

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
target_link_libraries(NewPassTests PRIVATE TaffoUtils TestUtils obj.NewPass [...])
```

The test file (`unittests/NewPass/Subfolder/NewComponentTest.cpp`) will have the following structure:

```cpp
 
#include "TestUtils.h"

#include "Subfolder/NewComponent.h"

namespace {
    using namespace llvm;

class NewComponentTest : public taffo_test::Test {
protected:
    /* Attributes to build the testing environment for the components */
    Constant *ConstantV;

    NewComponentTest() :
        ConstantV(ConstantInt::get(Type::getInt32Ty(Context), 0))
        {}

    /* Internal methods */
};

/* Actual tests */

TEST_F(NewComponentTest, MockTest) {
    ASSERT_EQ(ConstantV->getUniqueInteger().getLimitedValue(), 0);
}

}
```
## TestUtils
The ```TestUtils.h``` header file contains a set of utilities that aim at simplifying the creation of tests, such as:
- ```taffo_test::Test``` class: a class that inherits from ```testing::Test``` and provides a ```LLVMContext``` and a ```Module``` attribute, needed in almost every test.
- a LLVM IR parser, that allows to create a ```Module``` from a string containing the IR code.
- a set of generators, to create LLVM objects (e.g. ```Function```, ```GlobalVariable```) and TAFFO objects (e.g. ```mdutils::InputInfo```) with a single line of code.


## Running the tests

Once we created our test, we can compile and run the test suite with

```sh
mkdir build
cd build
cmake .. -DTAFFO_BUILD_ORTOOLS=ON -DUNITTESTS=ON
cmake --build .
ctest -VV
```
The ```-DUNITTESTS=ON```, other than enabling the compilation of unit tests, also sets the preprocessing macro ```UNITTESTS``` to ```1```, which can be used to expose some internal functions and methods (which would otherwise be private) to the tests. Even though this is not the best practice, it can be a better (if not the only) way to test some components which otherwise would require complex and difficult to read code to be tested.

## Code coverage
It might be useful to check the code coverage of the tests: the following instructions explain how to use the ```gcc``` compiler and ```lcov``` to generate the coverage report. Attempts have been made to use ```clang``` and ```llvm-cov``` but they have not been successful; if you decide to go down this path, prepare for trouble.

In order to get coverage information we need to add the following compile arguments
```shell
$ mkdir build
$ cd build/
$ cmake .. -DCMAKE_CXX_FLAGS="-fprofile-arcs -ftest-coverage -w" \
           -DCMAKE_C_FLAGS="-fprofile-arcs -ftest-coverage -w" \
           -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
           -DCMAKE_CXX_COMPILER="g++-12" \
           -DCMAKE_C_COMPILER="gcc-12" \
           -DTAFFO_UNITTESTS="ON" \
           -DTAFFO_BUILD_ORTOOLS="ON" \
           -DTAFFO_BUILD_ILP_DTA="ON" 
$ cmake --build .
```
After building we need to create a baseline (```lcov_base.info```) with
```shell
$ lcov --capture --initial --rc lcov_branch_coverage=1 --directory . --output-file lcov_base.info
```
At this point we can run the tests with ```ctest``` and generate the coverage data by running
```shell
$ lcov --capture --rc lcov_branch_coverage=1 --directory . --output-file lcov.info
$ lcov --add-tracefile lcov_base.info --add-tracefile lcov.info --output-file lcov.info
```
The ```lcov.info``` file can then be used to generate a report with any compatible tool (e.g [VSCode-LCOV](https://marketplace.visualstudio.com/items?itemName=alexdima.vscode-lcov))

