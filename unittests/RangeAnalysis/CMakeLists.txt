set(SELF RangeAnalysisTests)

taffo_add_unittest(${SELF} ${SELF}
        TaffoVRA/RangeOperationsCallWhitelistTest.cpp
        TaffoVRA/RangeOperationsTest.cpp
        TaffoVRA/VRAGlobalStoreTest.cpp
        TaffoVRA/VRAStoreTest.cpp
        )

include_directories(${CMAKE_SOURCE_DIR}/lib/RangeAnalysis)
target_link_libraries(${SELF} PRIVATE TaffoUtils TestUtils obj.TaffoVRA)
