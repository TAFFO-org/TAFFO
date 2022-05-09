set(SELF TaffoDTA)

set(DTA_SRC
  TaffoDTA.cpp
  TaffoDTA.h
  DTAConfig.h
  )

if (TAFFO_BUILD_ILP_DTA)
  set(DTA_SRC ${DTA_SRC}
    ILP/Optimizer.cpp
    ILP/Model.cpp
    ILP/PhiWatcher.cpp
    ILP/PhiWatcher.h
    ILP/CPUCosts.cpp
    ILP/MemWatcher.cpp
    ILP/MemWatcher.h
    ILP/LoopAnalyzerUtil.cpp
    ILP/MetricPerf.cpp
    ILP/MetricBase.cpp
    ILP/MetricPerfInstruction.cpp
    ILP/MetricBase.h
    ILP/Optimizer.h
    ILP/OptimizerInfo.h
    ILP/Model.h
    ILP/CPUCosts.h
    ILP/LoopAnalyzerUtil.h
    ILP/Utils.h
    ILP/Utils.cpp
    )
endif ()

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY ${DTA_SRC})
target_link_libraries(obj.${SELF} PUBLIC TaffoUtils)
target_include_directories(obj.${SELF} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (TAFFO_BUILD_ILP_DTA)
  target_link_libraries(obj.${SELF} PUBLIC ortools::ortools)
  target_compile_definitions(obj.${SELF} PRIVATE TAFFO_BUILD_ILP_DTA)
endif ()

set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
