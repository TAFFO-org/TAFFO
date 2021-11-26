cmake_minimum_required(VERSION 3.9)

project(taffo-j2a)

find_package(Clang REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

ADD_LIBRARY(j2a SHARED src/taffoAnnotations.cpp src/annotation.cpp)
TARGET_COMPILE_FEATURES(j2a PUBLIC cxx_std_17)
TARGET_INCLUDE_DIRECTORIES(j2a PUBLIC include)
TARGET_LINK_LIBRARIES(j2a PRIVATE LLVMSupport)
TARGET_COMPILE_OPTIONS(j2a PRIVATE -fno-exceptions -fno-rtti)

ADD_EXECUTABLE(taffo-j2a main.cpp)
TARGET_COMPILE_FEATURES(taffo-j2a PUBLIC cxx_std_17)
TARGET_LINK_LIBRARIES(taffo-j2a PRIVATE j2a clangTooling clangBasic clangASTMatchers LLVMSupport)
TARGET_COMPILE_OPTIONS(taffo-j2a PRIVATE -fno-exceptions -fno-rtti -fno-omit-frame-pointer)

install(TARGETS taffo-j2a DESTINATION bin)
install(TARGETS j2a DESTINATION lib)
