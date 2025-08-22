set(SELF TaffoMemToReg)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  MemToRegPass.hpp
  MemToRegPass.cpp
  PromoteMemToReg.hpp
  PromoteMemToReg.cpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoCommon
)
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
