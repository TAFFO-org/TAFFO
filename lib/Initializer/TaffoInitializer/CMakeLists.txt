set(SELF TaffoInitializer)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  TaffoInitializerPass.cpp
  Annotations.cpp
  AnnotationParser.cpp

  ADDITIONAL_HEADERS
  AnnotationParser.h
  TaffoInitializerPass.h
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
