set(SELF InitializerTests)

taffo_add_unittest(${SELF} ${SELF}
        TaffoInitializer/AnnotationParserTest.cpp
        TaffoInitializer/AnnotationsTest.cpp
        TaffoInitializer/IndirectCallPatcherTest.cpp
        )

include_directories(${CMAKE_SOURCE_DIR}/lib/Initializer)
# we also include conversion because the FixM command line option is defined there (thank you Gabriele)
target_link_libraries(${SELF} PRIVATE TaffoUtils TestUtils obj.TaffoInitializer obj.LLVMFloatToFixed)
