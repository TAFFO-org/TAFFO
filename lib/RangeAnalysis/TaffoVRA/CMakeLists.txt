set(SELF TaffoVRA)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  RangeOperationsCallWhitelist.cpp
  RangeOperations.cpp
  ValueRangeAnalysis.cpp
  CodeInterpreter.cpp
  VRAStore.cpp
  VRAnalyzer.cpp
  VRAGlobalStore.cpp
  VRAFunctionStore.cpp

  ADDITIONAL_HEADERS
  PtrCasts.hpp
  RangeOperationsCallWhitelist.hpp
  RangeOperations.hpp
  Range.hpp
  RangeNode.hpp
  ValueRangeAnalysis.hpp
  CodeInterpreter.hpp
  VRAStore.hpp
  VRAnalyzer.hpp
  VRAGlobalStore.hpp
  VRAFunctionStore.hpp
  VRALogger.hpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
