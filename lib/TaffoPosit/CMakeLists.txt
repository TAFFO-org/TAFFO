set(SELF TaffoPosit)

# Building TAFFO with Debug options should not make the resulting compiler output slower
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_RELEASE})

add_library(${SELF} SHARED
    TaffoPosit.cpp)

install(TARGETS ${SELF})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION include)

target_include_directories(${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET ${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)
