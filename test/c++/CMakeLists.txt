add_test(NAME c++ COMMAND ./run.py WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set (failRegex "ERR!")

set_property (TEST c++ PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
