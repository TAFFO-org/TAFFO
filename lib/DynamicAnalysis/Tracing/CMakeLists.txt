set(SELF DynamicTracing)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
    Tracing.cpp
    InjectFuncCall.cpp
    ReadTrace.cpp
    NameVariables.cpp

    ADDITIONAL_HEADERS
    InjectFuncCall.h
    ReadTrace.h
    NameVariables.h
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
    TaffoVRA
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
