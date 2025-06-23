add_test(NAME simple-test COMMAND ./run.py WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set (failRegex "ERR!")

set_property (TEST simple-test PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
