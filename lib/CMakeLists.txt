add_subdirectory(InstructionMix)
add_subdirectory(TaffoUtils)

add_subdirectory(Initializer)
add_subdirectory(RangeAnalysis)
add_subdirectory(DataTypeAlloc)
add_subdirectory(Conversion)
add_subdirectory(ErrorAnalysis)
add_subdirectory(TaffoMem2Reg)
add_subdirectory(FloatSizeAnalysis)

add_llvm_library(Taffo MODULE
  $<TARGET_OBJECTS:obj.TaffoInitializer>
  $<TARGET_OBJECTS:obj.TaffoVRA>
  $<TARGET_OBJECTS:obj.TaffoDTA>
  $<TARGET_OBJECTS:obj.LLVMFloatToFixed>
  $<TARGET_OBJECTS:obj.LLVMErrorPropagator>
  $<TARGET_OBJECTS:obj.TaffoMem2Reg>
  $<TARGET_OBJECTS:obj.FloatSizeAnalysis>
  PassPluginInfo.cpp
  ../cmake/dummy.cpp

  LINK_LIBS
  TaffoUtils
  stdc++
  m

  PLUGIN_TOOL
  opt
)

if (TAFFO_BUILD_ILP_DTA)
  target_link_libraries(Taffo PRIVATE ortools::ortools)
endif ()
