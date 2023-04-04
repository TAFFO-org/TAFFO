set(SELF TaffoUtils)

add_library(${SELF} STATIC
  Metadata.h
  Metadata.cpp
  InputInfo.h
  InputInfo.cpp
  TypeUtils.h
  TypeUtils.cpp
  MultiValueMap.h
  DebugUtils.h
  MemSSAUtils.cpp
  MemSSAUtils.hpp
  WriteModule.cpp
  WriteModule.h
)
target_include_directories(${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET ${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
