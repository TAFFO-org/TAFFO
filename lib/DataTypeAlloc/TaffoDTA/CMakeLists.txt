set(SELF TaffoDTA)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  TaffoDTA.cpp
  Optimizer.cpp
  Model.cpp
  OptimizerBinaryInstructions.cpp
  OptimizerInstructionHandling.cpp
  OptimizerConstantsHandling.cpp
  PhiWatcher.cpp
  CPUCosts.cpp
  MemWatcher.cpp
  LoopAnalyzerUtil.cpp
  MetricPerf.cpp
  MetricBase.cpp
  MetricPerfInstruction.cpp
  ADDITIONAL_HEADERS
  MetricBase.h
  MetricBaseForward.h
  TaffoDTA.h
  Optimizer.h
  Infos.h
  OptimizerInfo.h
  Model.h
  DTAConfig.h
  MemSSAUtils.hpp
  CPUCosts.h
  LoopAnalyzerUtil.h
  )

target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  ortools::ortools
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
