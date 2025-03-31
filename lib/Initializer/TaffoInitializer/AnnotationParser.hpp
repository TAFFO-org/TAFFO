#ifndef ANNOTATION_PARSER_HPP_
#define ANNOTATION_PARSER_HPP_

#include "TaffoInfo/ValueInfo.hpp"
#include "TaffoInitializerPass.hpp"
#include <sstream>
#include <string>

#define DEBUG_TYPE "taffo-init"

namespace taffo {

class AnnotationParser {
  std::istringstream stringStream;
  std::string nextToken;
  std::string error;

  void reset();

  bool parseOldSyntax(llvm::Type *type);

  bool parseNewSyntax(llvm::Type *type);
  bool parseScalar(std::shared_ptr<ValueInfo> &thisValueInfo, llvm::Type *type);
  bool parseStruct(std::shared_ptr<ValueInfo> &thisValueInfo, llvm::Type *type);
  char skipWhitespace();
  bool expectString(std::string &res);
  bool peek(std::string kw);
  bool expect(std::string kw);
  bool expectInteger(int64_t &res);
  bool expectReal(double &res);
  bool expectBoolean(bool &res);

public:
  bool startingPoint;
  bool backtracking;
  unsigned int backtrackingDepth;
  std::shared_ptr<ValueInfo> valueInfo;

  bool parseAnnotationString(llvm::StringRef annotationStr, llvm::Type *type);
  llvm::StringRef lastError();
};

} // namespace taffo

#undef DEBUG_TYPE
#endif // ANNOTATION_PARSER_HPP_
