set(SELF TaffoCommon)

add_llvm_library(${SELF} STATIC
  Containers/BiMap.hpp
  Debug/DebugUtils.hpp
  Debug/DebugUtils.cpp
  TaffoInfo/TaffoInfo.hpp
  TaffoInfo/TaffoInfo.cpp
  TaffoInfo/ValueInfo.hpp
  TaffoInfo/ValueInfo.cpp
  TaffoInfo/NumericInfo.hpp
  TaffoInfo/NumericInfo.cpp
  TaffoInfo/RangeInfo.hpp
  TaffoInfo/RangeInfo.cpp
  Types/TypeUtils.hpp
  Types/TypeUtils.cpp
  MetadataManager.hpp
  MetadataManager.cpp
  SerializationUtils.hpp
  SerializationUtils.cpp
  MemSSAUtils.cpp
  MemSSAUtils.hpp
)
target_include_directories(${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${SELF} PUBLIC
  TypeDeductionAnalysis
  nlohmann_json::nlohmann_json
)
set_property(TARGET ${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
