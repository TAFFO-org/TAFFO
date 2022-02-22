set(SELF TaffoUtilsTests)

taffo_add_unittest(${SELF} ${SELF}
    MultiValueMapTest.cpp)

target_link_libraries(${SELF} PRIVATE TaffoUtils)
