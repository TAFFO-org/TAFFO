file(
  WRITE
  ${CMAKE_CURRENT_BINARY_DIR}/taffo-config.sh
  "printf '#!/usr/bin/env bash\\nLLVM_DIR=\"%s\"\\n\' \"${LLVM_INSTALL_PREFIX}\" | cat - \"${CMAKE_CURRENT_SOURCE_DIR}/taffo.sh\" > taffo"
  )
add_custom_command(
  OUTPUT
  taffo
  
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/taffo-config.sh
  taffo.sh
  
  COMMAND
  "/bin/sh" "taffo-config.sh"
  )
  
add_custom_target(TaffoTool
  ALL
  
  DEPENDS
  taffo
  )

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/taffo
  
  DESTINATION
  bin 
  
  PERMISSIONS 
  OWNER_EXECUTE OWNER_WRITE OWNER_READ
  GROUP_EXECUTE GROUP_READ
  WORLD_EXECUTE WORLD_READ
  )


  install(DIRECTORY
  ILP

  TYPE 
  DATA)
