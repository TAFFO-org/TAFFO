#pragma once

#include "../FixedPointType.hpp"
#include "TransparentType.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Value.h>

namespace taffo {

struct ConversionInfo : tda::Printable {
  bool isBacktrackingNode;
  bool isRoot;
  llvm::SmallPtrSet<llvm::Value*, 5> roots;
  unsigned fixpTypeRootDistance = UINT_MAX;
  bool isConversionDisabled = false;
  bool isArgumentPlaceholder = false;

  // significant iff origType is a float or a pointer to a float and if operation == Convert
  std::shared_ptr<FixedPointType> fixpType = std::make_shared<FixedPointScalarType>();
  std::shared_ptr<tda::TransparentType> origType = nullptr;

  std::string toString() const override {
    std::stringstream ss;
    ss << "{ ";
    ss << (isBacktrackingNode ? "backtracking, " : "");
    ss << (isRoot ? "root, " : "");
    ss << "fixpTypeRootDistance: " << (fixpTypeRootDistance == UINT_MAX ? "inf" : std::to_string(fixpTypeRootDistance))
       << ", ";
    ss << (isConversionDisabled ? "disabled, " : "");
    ss << (isArgumentPlaceholder ? "argPlaceholder, " : "");
    ss << "origType: " << (origType ? origType->toString() : "null") << ", ";
    ss << "fixpType: " << (fixpType ? fixpType->toString() : "null") << ", ";
    ss << "roots: { ";
    bool first = true;
    for (llvm::Value* root : roots) {
      if (!first)
        ss << ", ";
      ss << root->getNameOrAsOperand();
      first = false;
    }
    ss << " } }";
    return ss.str();
  }
};

} // namespace taffo
