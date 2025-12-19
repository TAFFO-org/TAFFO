cmake_minimum_required(VERSION 3.14.7) # FetchContent
cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0079 NEW)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(Taffo)

set(BUILD_SHARED_LIBS OFF)

include(TaffoOptions)
include(TaffoFetchDeps)
include(FindClangFormat)

find_package(LLVM REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
include(AddLLVM)
include(LLVMConfig)
include(HandleLLVMOptions)

if((${LLVM_VERSION_MAJOR} LESS 18) OR (${LLVM_VERSION_MAJOR} GREATER 19))
  message(FATAL_ERROR
    " LLVM version ${LLVM_VERSION_MAJOR} is not supported!\n"
    " LLVM found at path: ${LLVM_DIR}")
endif()
message("LLVM found at path: ${LLVM_DIR}")

# to be kept in sync with LLVM!
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy-10;")
add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-covered-switch-default)

# build & install all tools by default
set(LLVM_BUILD_TOOLS ON)

# fix add_llvm_tool on macOS+Xcode
set(LLVM_MAIN_SRC_DIR ${CMAKE_SOURCE_DIR})

enable_testing()

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(test)
add_subdirectory(tool)

if (TAFFO_UNITTESTS)
  add_subdirectory(unittests)
endif()

# Set project git hooks
set(PROJECT_GIT_HOOKS_DIR "${CMAKE_SOURCE_DIR}/.githooks")
execute_process(
  COMMAND git config --local core.hooksPath "${PROJECT_GIT_HOOKS_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  RESULT_VARIABLE _git_cfg_res
)
if(NOT _git_cfg_res EQUAL 0)
  message(WARNING "Failed to set core.hooksPath (exit code ${_git_cfg_res})")
endif()
