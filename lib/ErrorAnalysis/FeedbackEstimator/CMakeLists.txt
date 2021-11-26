find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_python3_module(pickle REQUIRED)
find_python3_module(copy REQUIRED)

install(
  FILES 
  taffo-fe.py
  
  RENAME
  taffo-fe
  
  DESTINATION
  bin 
  
  PERMISSIONS 
  OWNER_EXECUTE OWNER_WRITE OWNER_READ
  GROUP_EXECUTE GROUP_READ
  WORLD_EXECUTE WORLD_READ
  )
