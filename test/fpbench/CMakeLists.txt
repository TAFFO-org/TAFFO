
add_test(NAME fpbench COMMAND ./run.py -M 1 WORKING_DIRECTORY "/home/ilgeco/opt/TAFFO/test/fpbench")

set (failRegex "ERR!")

set_property (TEST fpbench
              PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")