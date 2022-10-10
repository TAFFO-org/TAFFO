set(SELF FloatSizeAnalysis)

add_llvm_library(${SELF} OBJECT BUILDTREE_ONLY
        FloatSizeAnalysisPass.cpp
        FloatSizeAnalysis.cpp
        FloatSizeAnalysis.h
        )
target_link_libraries(obj.${SELF} PUBLIC
        TaffoUtils
        TaffoVRA
        )
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)