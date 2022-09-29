set(SELF DataTypeAllocTests)

taffo_add_unittest(${SELF} ${SELF}
        TaffoDTA/ILP/UtilsTest.cpp
        TaffoDTA/ILP/PhiWatcherTest.cpp
        )

include_directories(${CMAKE_SOURCE_DIR}/lib/DataTypeAlloc)
target_link_libraries(${SELF} PRIVATE TaffoUtils TestUtils obj.TaffoDTA)
