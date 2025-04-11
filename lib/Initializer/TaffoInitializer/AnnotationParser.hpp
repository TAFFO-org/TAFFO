#ifndef ANNOTATION_PARSER_HPP_
#define ANNOTATION_PARSER_HPP_

#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueInfo.hpp"
#include "TaffoInitializerPass.hpp"
#include "llvm/IR/Value.h"
#include <memory>
#include <sstream>
#include <string>

#define DEBUG_TYPE "taffo-init"

namespace taffo {

class AnnotationParser {
  std::istringstream stringStream;
  std::string nextToken;
  std::string error;
  std::shared_ptr<ValueInfo> valueInfoBuild;

  void reset();

  bool parseSyntax(llvm::Type *type);
  bool parseScalar(std::shared_ptr<ValueInfo> &thisValueInfo, llvm::Type *type);
  bool parseStruct(std::shared_ptr<ValueInfo> &thisValueInfo, llvm::Type *type);
  char skipWhitespace();
  bool expectString(std::string &res);
  bool peek(std::string kw);
  bool expect(std::string kw);
  bool expectInteger(int64_t &res);
  bool expectReal(double &res);
  bool expectBoolean(bool &res);
  std::shared_ptr<ValueInfo>&& buildValueInfo();

public:
  bool startingPoint;
  bool backtracking;
  unsigned int backtrackingDepth;

  bool parseAnnotationAndGenValueInfo(llvm::StringRef annotationStr, llvm::Value* annotatedValue);
  llvm::StringRef getLastError();
};

} // namespace taffo

#undef DEBUG_TYPE
#endif // ANNOTATION_PARSER_HPP_
