set(SELF TaffoInitializer)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  TaffoInitializerPass.cpp
  Annotations.cpp
  AnnotationParser.cpp
  IndirectCallPatcher.cpp
  OpenCLKernelPatcher.cpp
  CudaKernelPatcher.cpp

  ADDITIONAL_HEADERS
  TaffoInitializerPass.hpp
  AnnotationParser.hpp
  IndirectCallPatcher.hpp
  OpenCLKernelPatcher.hpp
  CudaKernelPatcher.hpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoCommon
)
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
