##
## Download dependencies if required
##

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

if (TAFFO_BUILD_ILP_DTA)
  if (TAFFO_BUILD_ORTOOLS)
    message("-- Fetching ORTools")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    
    FetchContent_Declare(
      or-tools
      GIT_REPOSITORY "https://github.com/google/or-tools.git"
      GIT_TAG "v9.12"
      GIT_SHALLOW ON
      GIT_PROGRESS ON)
    # TODO: disable some of this stuff, not sure we need it all
    set(BUILD_Protobuf ON)
    set(BUILD_absl ON)
    set(BUILD_SCIP ON)
    set(BUILD_CoinUtils ON)
    set(BUILD_Osi ON)
    set(BUILD_Clp ON)
    set(BUILD_Cgl ON)
    set(BUILD_Cbc ON)
    set(BUILD_Eigen3 ON)
    set(BUILD_re2 ON)
    set(USE_XPRESS OFF)
    set(USE_HIGHS OFF)
    set(BUILD_TESTING OFF)
    set(BUILD_CXX_SAMPLES OFF)
    set(BUILD_CXX_EXAMPLES OFF)

    FetchContent_MakeAvailable(or-tools)

    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message("-- ORTools fetched")
  else ()
    # Protobuf is a dependency to ortools, but its cmake files might not have been
    # installed so we try to find it before ortool does in order to use the 
    # FindProtobuf.cmake we ship if necessary
    find_package(Protobuf REQUIRED)
    find_package(ortools CONFIG REQUIRED)
  endif ()
endif ()

FetchContent_Declare(
  json
  GIT_REPOSITORY "https://github.com/nlohmann/json.git"
  GIT_TAG "v3.11.3"
)
FetchContent_MakeAvailable(json)

execute_process(
  COMMAND git submodule update --init --recursive
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  RESULT_VARIABLE SUBMOD_RET)

if (NOT SUBMOD_RET EQUAL 0)
  message(FATAL_ERROR "Submodule sync failed")
endif()
