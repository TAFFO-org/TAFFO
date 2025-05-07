add_test(NAME fpbench COMMAND ./run.py WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set (failRegex "ERR!")

set_property (TEST fpbench PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
