set(SELF TaffoUtils)


add_library(${SELF}
  HandleSpecialFunction.h
  Metadata.h
  Metadata.cpp
  InputInfo.h
  InputInfo.cpp
  TypeUtils.h
  TypeUtils.cpp
  MultiValueMap.h
  LLVMVersions.cpp
  LLVMVersions.h
  DebugUtils.h
  DebugPrint.h
  MemSSAUtils.cpp
  MemSSAUtils.hpp
  PtrCasts.h
  OpenMPAnalyzer.cpp
  OpenMPAnalyzer.h
  WriteModule.h
  WriteModule.cpp

)
target_include_directories(${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET ${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
