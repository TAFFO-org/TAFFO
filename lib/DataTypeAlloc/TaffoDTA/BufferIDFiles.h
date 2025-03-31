#ifndef BUFFER_ID_FILES
#define BUFFER_ID_FILES

#include "TaffoInfo/ValueInfo.hpp"
#include <string>
#include <map>

#define DEBUG_TYPE "taffo-dta"

namespace tuner {

using BufferIDTypeMap = std::map<std::string, std::shared_ptr<taffo::NumericType>>;

void ReadBufferIDFile(std::string Fn, BufferIDTypeMap& OutMap);
void WriteBufferIDFile(std::string Fn, BufferIDTypeMap& Map);

} // namespace tuner

#endif
