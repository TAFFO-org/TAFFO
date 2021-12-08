cmake_minimum_required(VERSION 3.12) # for FindPython
cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0079 NEW)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(Taffo)

include(TaffoOptions)

find_package(LLVM REQUIRED CONFIG)

if (TAFFO_BUILD_ILP_DTA)
  # Protobuf is a dependency to ortools, but its cmake files might not have been
  # installed so we try to find it before ortool does in order to use the 
  # FindProtobuf.cmake we ship if necessary
  find_package(Protobuf REQUIRED)
  find_package(ortools CONFIG REQUIRED)
endif ()

add_definitions(${LLVM_DEFINITIONS})
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
include(AddLLVM)
include(HandleLLVMOptions)

# to be kept in sync with LLVM!
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy-10;")
add_compile_options(-Wall -Wextra)

# build & install all tools by default
set(LLVM_BUILD_TOOLS ON)

# fix add_llvm_tool on macOS+Xcode
set(LLVM_MAIN_SRC_DIR ${CMAKE_SOURCE_DIR})

enable_testing()

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(unittests)
add_subdirectory(tool)

