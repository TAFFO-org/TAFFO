
add_test(NAME fpbench COMMAND ./run.py -M=1 -debug WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/test/fpbench")






set (failRegex "ERR!")

set_property (TEST fpbench
              PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")