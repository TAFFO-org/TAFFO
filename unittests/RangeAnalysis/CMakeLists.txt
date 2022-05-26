set(SELF RangeAnalysisTests)

taffo_add_unittest(${SELF} ${SELF}
        TaffoVRA/RangeOperationsTest.cpp
        )

include_directories(${CMAKE_SOURCE_DIR}/lib/RangeAnalysis)
target_link_libraries(${SELF} PRIVATE TaffoUtils TestUtils obj.TaffoVRA)
