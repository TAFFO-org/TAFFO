#pragma once

#include "ValueConvInfo.hpp"

using namespace llvm;
using namespace tda;
using namespace taffo;

std::string ValueConvInfo::toString() const {
  std::stringstream ss;
  ss << "{ ";
  ss << "oldType: " << (oldType ? oldType->toString() : "null") << ", ";
  if (conversionDisabled)
    ss << "disabled, ";
  else if (newType)
    ss << "newType: " << *newType << ", ";
  if (isConverted)
    ss << "converted, ";
  ss << (isArgumentPlaceholder ? "argPlaceholder, " : "");
  ss << (isBacktrackingNode ? "backtracking, " : "");
  if (isRoot)
    ss << "root, ";
  else {
    ss << "roots: { ";
    bool first = true;
    for (Value* root : roots) {
      if (!first)
        ss << ", ";
      ss << root->getNameOrAsOperand();
      first = false;
    }
    ss << " }";
  }
  ss << " }";
  return ss.str();
}
