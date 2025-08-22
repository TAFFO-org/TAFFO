set(SELF TaffoTypeDeducer)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
  TypeDeducerPass.hpp
  TypeDeducerPass.cpp
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoCommon
)
