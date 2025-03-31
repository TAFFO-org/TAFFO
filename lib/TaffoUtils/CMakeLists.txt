set(SELF TaffoUtils)

add_library(${SELF} STATIC
  TaffoInfo/TaffoInfo.hpp
  TaffoInfo/TaffoInfo.cpp
  TaffoInfo/ValueInfo.hpp
  TaffoInfo/ValueInfo.cpp
  TaffoInfo/NumericInfo.hpp
  TaffoInfo/NumericInfo.cpp
  TaffoInfo/RangeInfo.hpp
  TaffoInfo/RangeInfo.cpp
  MetadataManager.hpp
  MetadataManager.cpp
  Logger.hpp
  Logger.cpp
  InsertionOrderedMap.hpp
  BiMap.hpp
  PtrCasts.hpp
  SerializationUtils.hpp
  TypeUtils.h
  TypeUtils.cpp
  DebugUtils.h
  MemSSAUtils.cpp
  MemSSAUtils.hpp
  WriteModule.cpp
  WriteModule.h
)
target_include_directories(${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${SELF} PUBLIC nlohmann_json::nlohmann_json)
set_property(TARGET ${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
