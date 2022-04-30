set(SELF InitializerTests)

taffo_add_unittest(${SELF} ${SELF}
        TaffoInitializer/AnnotationParserTest.cpp
        TaffoInitializer/AnnotationsTest.cpp
        )

include_directories(${CMAKE_SOURCE_DIR}/lib/Initializer)
target_link_libraries(${SELF} PRIVATE TaffoUtils TestUtils obj.TaffoInitializer)
