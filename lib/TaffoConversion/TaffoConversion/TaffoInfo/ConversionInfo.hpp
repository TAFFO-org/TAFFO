#pragma once

#include "../FixedPointType.hpp"
#include "SerializationUtils.hpp"
#include "Types/TransparentType.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

namespace taffo {

struct ConversionInfo : taffo::Printable {
  bool isBacktrackingNode;
  bool isRoot;
  llvm::SmallPtrSet<llvm::Value*, 5> roots;
  unsigned int fixpTypeRootDistance = UINT_MAX;

  /* Disable type conversion even if the instruction
   * produces a floating point value */
  bool noTypeConversion = false;
  bool isArgumentPlaceholder = false;

  // significant iff origType is a float or a pointer to a float
  // and if operation == Convert
  std::shared_ptr<FixedPointType> fixpType = std::make_shared<FixedPointScalarType>();
  std::shared_ptr<TransparentType> origType = nullptr;

  std::string toString() const override {
    std::stringstream ss;
    ss << "ConversionInfo: { ";
    ss << "isBacktrackingNode: " << (isBacktrackingNode ? "true" : "false") << ", ";
    ss << "isRoot: " << (isRoot ? "true" : "false") << ", ";
    ss << "fixpTypeRootDistance: " << fixpTypeRootDistance << ", ";
    ss << "noTypeConversion: " << (noTypeConversion ? "true" : "false") << ", ";
    ss << "isArgumentPlaceholder: " << (isArgumentPlaceholder ? "true" : "false") << ", ";
    ss << "origType: ";
    if (origType)
      ss << origType->toString();
    else
      ss << "null";
    ss << ", fixpType: ";
    if (fixpType)
      ss << *fixpType;
    else
      ss << "null";
    ss << ", roots: {";
    bool first = true;
    for (llvm::Value* v : roots) {
      if (!first)
        ss << ", ";
      ss << v;
      first = false;
    }
    ss << "} }";
    return ss.str();
  }
};

} // namespace taffo
