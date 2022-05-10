set(SELF taffo-costmodel)

add_executable(${SELF}
  taffo-costmodel.c
  benchmark.h
  )

target_link_libraries(${SELF} m)

install(TARGETS ${SELF})

