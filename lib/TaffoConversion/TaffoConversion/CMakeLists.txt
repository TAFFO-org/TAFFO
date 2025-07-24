set(SELF TaffoConversion)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  ConversionPass.hpp
  ConversionPass.cpp
  ConversionInfoBuilder.cpp
  Conversion.cpp
  ConstantConversion.cpp
  IndirectCallConversion.cpp
  InstructionConversion.cpp
  OpenCLConversion.cpp
  CudaConversion.cpp
  MathIntrinsicConversion.cpp
  FixedPointType.hpp
  FixedPointType.cpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoCommon
)
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
