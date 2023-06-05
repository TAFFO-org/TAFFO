
add_test(NAME polybench COMMAND collect-fe-stats.sh WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set (failRegex "\[[ \n\r\t]*[-0-9]+[ \n\r\t]*\]")

set_property (TEST polybench
              PROPERTY FAIL_REGULAR_EXPRESSION "${failRegex}")
