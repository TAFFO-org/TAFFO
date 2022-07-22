set(SELF LLVMFloatToFixed)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  FixedPointType.cpp
  LLVMFloatToFixedPass.cpp
  MDCollector.cpp
  Conversion.cpp
  ConstantConversion.cpp
  IndirectCallConversion.cpp
  InstructionConversion.cpp
  OpenCLConversion.cpp
  MathIntrinsicConversion.cpp

  ADDITIONAL_HEADERS
  FixedPointType.h
  LLVMFloatToFixedPass.h
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
