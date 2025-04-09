add_subdirectory(InstructionMix)
add_subdirectory(TaffoCommon)

add_subdirectory(TypeDeducer)
add_subdirectory(Initializer)
add_subdirectory(RangeAnalysis)
add_subdirectory(DataTypeAlloc)
add_subdirectory(Conversion)
#add_subdirectory(ErrorAnalysis)
add_subdirectory(TaffoMem2Reg)

add_llvm_library(Taffo MODULE
  $<TARGET_OBJECTS:obj.TypeDeducer>
  $<TARGET_OBJECTS:obj.TaffoInitializer>
  $<TARGET_OBJECTS:obj.TaffoVRA>
  $<TARGET_OBJECTS:obj.TaffoDTA>
  $<TARGET_OBJECTS:obj.LLVMFloatToFixed>
  #$<TARGET_OBJECTS:obj.LLVMErrorPropagator>
  $<TARGET_OBJECTS:obj.TaffoMem2Reg>
  PassPluginInfo.cpp
  ../cmake/dummy.cpp

  LINK_LIBS
  TaffoCommon
  stdc++
  m

  PLUGIN_TOOL
  opt
)

if (TAFFO_BUILD_ILP_DTA)
  target_link_libraries(Taffo PRIVATE ortools::ortools)
endif ()

target_link_libraries(Taffo PRIVATE nlohmann_json::nlohmann_json)

if (TAFFO_UNITTESTS)
  target_compile_definitions(obj.TaffoInitializer PUBLIC UNITTESTS)
  target_compile_definitions(obj.TaffoVRA PUBLIC UNITTESTS)
endif()
