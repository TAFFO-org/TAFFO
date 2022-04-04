set(SELF DynamicTracing)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  InjectFuncCall.cpp

  ADDITIONAL_HEADERS
  InjectFuncCall.h
)
#target_link_libraries(obj.${SELF} PUBLIC
#  TaffoUtils
#  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
