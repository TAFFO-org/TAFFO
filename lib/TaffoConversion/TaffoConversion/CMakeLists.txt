set(SELF TaffoConversion)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  TaffoInfo/PhiInfo.hpp
  TaffoInfo/PhiInfo.cpp
  TaffoInfo/TaffoConvInfo.hpp
  TaffoInfo/TaffoConvInfo.cpp
  TaffoInfo/ValueConvInfo.hpp
  TaffoInfo/ValueConvInfo.cpp
  Types/ConversionType.hpp
  Types/ConversionType.cpp
  ConversionPass.hpp
  ConversionPass.cpp
  ValueConvInfoBuilder.cpp
  Conversion.cpp
  ConstantConversion.cpp
  IndirectCallConversion.cpp
  InstructionConversion.cpp
  OpenCLConversion.cpp
  CudaConversion.cpp
  MathIntrinsicConversion.cpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoCommon
)
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
