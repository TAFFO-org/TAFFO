set(SELF TaffoPosit)

add_library(obj.${SELF} OBJECT
    TaffoPosit.cpp)

target_include_directories(obj.${SELF} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET obj.${SELF} PROPERTY POSITION_INDEPENDENT_CODE ON)

### Also build as a shared library
# Building TAFFO with Debug options should not make the resulting compiler output slower
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_RELEASE})

add_library(${SELF} SHARED
    $<TARGET_OBJECTS:obj.${SELF}>)

install(TARGETS ${SELF})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION include)
