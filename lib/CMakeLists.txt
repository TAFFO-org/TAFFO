add_subdirectory(InstructionMix)
add_subdirectory(TaffoUtils)

add_subdirectory(Initializer)
add_subdirectory(RangeAnalysis)
add_subdirectory(DataTypeAlloc)
add_subdirectory(Conversion)
add_subdirectory(ErrorAnalysis)

add_llvm_library(Taffo MODULE
  $<TARGET_OBJECTS:obj.TaffoInitializer>
  $<TARGET_OBJECTS:obj.TaffoVRA>
  $<TARGET_OBJECTS:obj.TaffoDTA>
  $<TARGET_OBJECTS:obj.LLVMFloatToFixed>
  $<TARGET_OBJECTS:obj.LLVMErrorPropagator>
  ../cmake/dummy.cpp

  LINK_LIBS
  TaffoUtils
  m
  stdc++

  PLUGIN_TOOL
  opt
)
