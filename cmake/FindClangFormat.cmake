# Find clang-format in the system's PATH
find_program(CLANG_FORMAT_EXECUTABLE clang-format)

# Check if clang-format was found
if(NOT CLANG_FORMAT_EXECUTABLE)
  message(FATAL_ERROR "clang-format not found. Please install clang-format version 19 or higher.")
endif()

# Check the version of clang-format
execute_process(
  COMMAND ${CLANG_FORMAT_EXECUTABLE} --version
  OUTPUT_VARIABLE CLANG_FORMAT_VERSION_OUTPUT
  ERROR_VARIABLE CLANG_FORMAT_VERSION_ERROR
  RESULT_VARIABLE CLANG_FORMAT_VERSION_RESULT
)

# Extract the version number from the output
if(NOT CLANG_FORMAT_VERSION_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get clang-format version.")
endif()

# Extract the version number from the output (e.g., "clang-format version 19")
string(REGEX MATCH "[0-9]+" CLANG_FORMAT_VERSION ${CLANG_FORMAT_VERSION_OUTPUT})

# Compare the version number
if(CLANG_FORMAT_VERSION VERSION_LESS "19")
  message(FATAL_ERROR " clang-format version 19 or higher is required.\n"
                      " Installed version: ${CLANG_FORMAT_VERSION}")
else()
  message(STATUS "clang-format version ${CLANG_FORMAT_VERSION} found.")
endif()
