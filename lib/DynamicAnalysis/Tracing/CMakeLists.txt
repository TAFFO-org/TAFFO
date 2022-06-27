set(SELF DynamicTracing)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
    Tracing.cpp
    InjectFuncCall.cpp
    ReadTrace.cpp
    NameVariables.cpp
    MemoryGraph.cpp
    ConnectedComponents.cpp
    ExpandEqualValue.cpp
    TracingUtils.cpp
    StripAnnotations.cpp

    ADDITIONAL_HEADERS
    InjectFuncCall.h
    ReadTrace.h
    NameVariables.h
    MemoryGraph.h
    ConnectedComponents.h
    ExpandEqualValue.h
    TracingUtils.h
    StripAnnotations.h
)
target_link_libraries(obj.${SELF} PUBLIC
  TaffoUtils
    TaffoVRA
  )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
