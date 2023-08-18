
add_test(NAME simple-test COMMAND ./run-test.sh WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/test/simple-test-cases")

set (failRegex "FAIL")

set_property (TEST simple-test
              PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")