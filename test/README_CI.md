# CI

## How to add a new test

Suppose you have developed a new test called *bestTest*

### Step 1

Add *bestTest* folder to test/CMakeLists.txt

```cmake
    ...
    add_subdirectory(bestTest)
```
### Step 2

Inside *bestTest* folder create a new CMakeLists.txt

```
TAFFO
├── test
|   ├── bestTest 
│   |   ├── CMakeLists.txt <-- New File
|   |   ├── ...
|   ├── CMakeLists.txt
...
```

### Step 3

Now we can add our test in the  bestTest/CMakeLists.txt
with the following command 

``` cmake
add_test( NAME <name> COMMAND <command> [<arg>...]
         [CONFIGURATIONS <config>...]
         [WORKING_DIRECTORY <dir>]
         [COMMAND_EXPAND_LISTS])
```

In our example we will add something like this

``` cmake
add_test(NAME bestTest COMMAND {./run.py | ./run.sh} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/test/bestTest")
```

### Step 4

By default a test passes if all of the following conditions are true:

- The test executable was found

- The test ran without exception

- The test exited with return code 0

That said, these behaviors can be modified using the set_property command:

``` cmake 
set_property(TEST test_name PROPERTY prop1 value1 value2 ...)     
```

To add a failing condition we use the **FAIL_REGULAR_EXPRESSION** property.

Suppose that ./run.sh inside test/bestTest will print out Awful Mistake in case of error.

We can add these lines to test/bestTest/CMakeList.txt

```cmake

set (failRegex "Awful Mistake")

set_property (TEST fpbench PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
```




### Property

**ENVIRONMENT**

    Specifies environment variables that should be defined for running a test. If set to a list of environment variables and values of the form MYVAR=value, those environment variables will be defined while the test is running. The environment is restored to its previous state after the test is done.
**LABELS**

    Specifies a list of text labels associated with a test. These labels can be used to group tests together based on what they test. For example, you could add a label of MPI to all tests that exercise MPI code.
**WILL_FAIL**

    If this option is set to true, then the test will pass if the return code is not 0, and fail if it is. This reverses the third condition of the pass requirements.
**PASS_REGULAR_EXPRESSION**

    If this option is specified, then the output of the test is checked against the regular expression provided (a list of regular expressions may be passed in as well). If none of the regular expressions match, then the test will fail. If at least one of them matches, then the test will pass.
**FAIL_REGULAR_EXPRESSION**

    If this option is specified, then the output of the test is checked against the regular expression provided (a list of regular expressions may be passed in as well). If none of the regular expressions match, then the test will pass. If at least one of them matches, then the test will fail.











