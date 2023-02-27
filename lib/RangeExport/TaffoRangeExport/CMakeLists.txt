set(SELF TaffoRangeExport)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  RangeExportPass.cpp

  ADDITIONAL_HEADERS
  RangeExportPass.h
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
