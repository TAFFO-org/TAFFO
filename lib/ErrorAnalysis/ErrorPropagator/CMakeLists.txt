set(SELF LLVMErrorPropagator)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  ErrorPropagator.cpp
  FunctionErrorPropagator.cpp
  RangeErrorMap.cpp
  StructErrorMap.cpp
  FunctionCopyMap.cpp
  Propagators.cpp
  PropagatorsUtils.cpp
  SpecialFunctions.cpp
  MemSSAUtils.cpp
  AffineForms.cpp
  FixedPoint.cpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
