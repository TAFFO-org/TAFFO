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
  CudaConversion.cpp
  MathIntrinsicConversion.cpp
  SinCos.cpp
  TAFFOMath.cpp
  ArcSinCos.cpp
  PositBuilder.cpp
  PositBuilderSoftware.cpp
  PositBuilderRISCV.cpp
  PositConstant.cpp


  ADDITIONAL_HEADERS
  FixedPointType.h
  LLVMFloatToFixedPass.h
  SinCos.h
  TAFFOMath.h

)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
