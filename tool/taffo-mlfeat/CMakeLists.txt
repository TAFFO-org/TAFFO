set(SELF taffo-mlfeat)

set(LLVM_LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  BitWriter
  CodeGen
  Core
  Coroutines
  IPO
  IRReader
  InstCombine
  Instrumentation
  MC
  ObjCARCOpts
  ScalarOpts
  Support
  Target
  TransformUtils
  Vectorize
  Passes
  )

add_llvm_tool(${SELF}
  taffo-mlfeat.cpp
  TaffoMLFeaturesAnalysis.cpp
  TaffoMLFeaturesAnalysis.h
  )
target_link_libraries(${SELF} PUBLIC
  InstructionMix
  TaffoUtils
  )

