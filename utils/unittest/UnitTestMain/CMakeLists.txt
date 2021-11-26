add_llvm_library(taffo_gtest_main
  TestMain.cpp

  LINK_LIBS
  taffo_gtest

  # This library is not meant to be in the install tree, only the build tree.
  BUILDTREE_ONLY
  )
if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(taffo_gtest_main PRIVATE -fno-rtti)
endif()
