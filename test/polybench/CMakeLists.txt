add_test(NAME polybench COMMAND ./run.py WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set (failRegex "ERR!")

set_property (TEST polybench PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
