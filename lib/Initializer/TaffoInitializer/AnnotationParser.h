#include <string>
#include <sstream>
#include "TaffoInitializerPass.h"
#include "InputInfo.h"


#ifndef __ANNOTATION_PARSER_H__
#define __ANNOTATION_PARSER_H__


namespace taffo {


class AnnotationParser {
  std::istringstream sstream;
  std::string nextToken;
  std::string error;
  
  void reset();
  
  bool parseOldSyntax();
  
  bool parseNewSyntax();
  bool initializeInputInfo(std::shared_ptr<mdutils::MDInfo>& thisMd);
  bool parseScalar(std::shared_ptr<mdutils::MDInfo>& thisMd);
  bool parseStruct(std::shared_ptr<mdutils::MDInfo>& thisMd);
  char skipWhitespace();
  bool expectString(std::string& res);
  bool peek(std::string kw) {
    std::streampos pos = sstream.tellg();
    bool res;
    if (!(res = expect(kw))) {
      sstream.clear();
      sstream.seekg(pos);
    }
    return res;
  };
  bool expect(std::string kw);
  bool expectInteger(int64_t& res);
  bool expectReal(double& res);
  bool expectBoolean(bool& res);
  
public:
  llvm::Optional<std::string> target;
  bool startingPoint;
  bool backtracking;
  unsigned int backtrackingDepth;
  std::shared_ptr<mdutils::MDInfo> metadata;
  
  bool parseAnnotationString(llvm::StringRef annString);
  llvm::StringRef lastError();
};


}


#endif // __ANNOTATION_PARSER_H__
