set(SELF TaffoInitializer)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  TaffoInitializerPass.hpp
  TaffoInitializerPass.cpp
  Annotations.cpp
  AnnotationParser.hpp
  AnnotationParser.cpp
  TaffoInfo/TaffoInitInfo.hpp
  TaffoInfo/TaffoInitInfo.cpp
  TaffoInfo/ValueInitInfo.hpp
  TaffoInfo/ValueInitInfo.cpp
  IndirectCallPatcher.hpp
  IndirectCallPatcher.cpp
  OpenCLKernelPatcher.hpp
  OpenCLKernelPatcher.cpp
  CudaKernelPatcher.hpp
  CudaKernelPatcher.cpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoCommon
)
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
