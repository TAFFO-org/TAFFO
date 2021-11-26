set(SELF TaffoDTA)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  TaffoDTA.cpp

  ADDITIONAL_HEADERS
  TaffoDTA.h
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
