add_subdirectory(InstructionMix)
add_subdirectory(TypeDeductionAnalysis)
add_subdirectory(TaffoCommon)

add_subdirectory(TaffoTypeDeducer)
add_subdirectory(TaffoInitializer)
add_subdirectory(TaffoVRA)
add_subdirectory(TaffoDTA)
add_subdirectory(TaffoConversion)
add_subdirectory(TaffoMemToReg)

add_llvm_library(Taffo MODULE
  $<TARGET_OBJECTS:obj.TaffoTypeDeducer>
  $<TARGET_OBJECTS:obj.TaffoInitializer>
  $<TARGET_OBJECTS:obj.TaffoVRA>
  $<TARGET_OBJECTS:obj.TaffoDTA>
  $<TARGET_OBJECTS:obj.TaffoConversion>
  $<TARGET_OBJECTS:obj.TaffoMemToReg>
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
