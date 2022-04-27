set(SELF taffo-costmodel)

add_executable(${SELF}
  taffo-costmodel.c
  timing.c
  timing.h
  config.h
  )

# target_compile_definitions(${SELF} PRIVATE
#  -DNOFLT80 -DNOFLT128 -DNBRUN=128 -DMEMSIZE=10000000
#  )
target_link_libraries(${SELF} m)

install(TARGETS ${SELF})

