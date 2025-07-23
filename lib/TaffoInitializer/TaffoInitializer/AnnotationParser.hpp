#pragma once

#include "InitializerPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueInfo.hpp"

#include <sstream>
#include <string>

#define DEBUG_TYPE "taffo-init"

namespace taffo {

class AnnotationParser {
public:
  bool startingPoint;
  bool backtracking;
  unsigned backtrackingDepth;

  bool parseAnnotationAndGenValueInfo(const std::string& annotationStr, llvm::Value* annotatedValue);
  llvm::StringRef getLastError();

private:
  std::istringstream stringStream;
  std::string nextToken;
  std::string error;
  std::shared_ptr<ValueInfo> valueInfoBuild;

  void reset();
  bool parseSyntax(const std::shared_ptr<tda::TransparentType>& type);
  bool parseScalar(std::shared_ptr<ValueInfo>& thisValueInfo, const std::shared_ptr<tda::TransparentType>& type);
  bool parseStruct(std::shared_ptr<ValueInfo>& thisValueInfo, const std::shared_ptr<tda::TransparentType>& type);
  char skipWhitespace();
  bool expectString(std::string& res);
  bool peek(std::string kw);
  bool expect(std::string kw);
  bool expectInteger(int64_t& res);
  bool expectReal(double& res);
  bool expectBoolean(bool& res);
  std::shared_ptr<ValueInfo>&& buildValueInfo();
};

} // namespace taffo

#undef DEBUG_TYPE
