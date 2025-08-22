set(SELF TaffoVRA)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  ValueRangeAnalysisPass.hpp
  ValueRangeAnalysisPass.cpp
  RangeOperationsCallWhitelist.hpp
  RangeOperationsCallWhitelist.cpp
  RangeOperations.hpp
  RangeOperations.cpp
  CodeInterpreter.hpp
  CodeInterpreter.cpp
  VRAStore.hpp
  VRAStore.cpp
  VRAnalyzer.hpp
  VRAnalyzer.cpp
  VRAGlobalStore.hpp
  VRAGlobalStore.cpp
  VRAFunctionStore.hpp
  VRAFunctionStore.cpp
  VRALogger.hpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoCommon
)
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
