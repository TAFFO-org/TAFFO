set(SELF TaffoCommon)

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
  Containers/InsertionOrderedMap.hpp
  Containers/BiMap.hpp
  Debug/DebugUtils.hpp
  Debug/Logger.hpp
  Debug/Logger.cpp
  Types/TransparentType.hpp
  Types/TransparentType.cpp
  Types/TypeUtils.hpp
  Types/TypeUtils.cpp
  PtrCasts.hpp
  SerializationUtils.hpp
  MemSSAUtils.cpp
  MemSSAUtils.hpp
  WriteModule.cpp
  WriteModule.hpp
)
target_include_directories(${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${SELF} PUBLIC nlohmann_json::nlohmann_json)
set_property(TARGET ${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
