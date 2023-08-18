set(SELF TaffoMem2Reg)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  Mem2Reg.h
  Mem2Reg.cpp
  PromoteMemToReg.h
  PromoteMemToReg.cpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
