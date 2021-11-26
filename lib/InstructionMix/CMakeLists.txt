set(SELF InstructionMix)
  
add_library(${SELF}
  InstructionMix.cpp
  InstructionMix.h
  )
target_include_directories(${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
