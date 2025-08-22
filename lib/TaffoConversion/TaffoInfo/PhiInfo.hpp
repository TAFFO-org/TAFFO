#pragma once

#include "Utils/PrintUtils.hpp"

#include <llvm/IR/Value.h>

namespace taffo {

struct PhiInfo : tda::Printable {
  llvm::Value* oldPhi = nullptr;
  llvm::Value* newPhi = nullptr;

  std::string toString() const override;
};

} // namespace taffo
