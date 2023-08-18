#ifndef BUFFER_ID_FILES
#define BUFFER_ID_FILES

#include "InputInfo.h"
#include <string>
#include <map>

#define DEBUG_TYPE "taffo-dta"

namespace tuner
{

using BufferIDTypeMap = std::map<std::string, std::unique_ptr<mdutils::TType>>;

void ReadBufferIDFile(std::string Fn, BufferIDTypeMap& OutMap);
void WriteBufferIDFile(std::string Fn, BufferIDTypeMap& Map);

}

#endif
