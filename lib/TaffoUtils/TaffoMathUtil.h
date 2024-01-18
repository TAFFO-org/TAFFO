#pragma once
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Function.h"
#include <string>

namespace TaffoMath
{

// FIXME: better find libm function
//  with this code even a function like cosplayer is matched
static bool isSupportedLibmFunction(const llvm::Function *F, bool enabled = false)
{
  static const auto names = llvm::SmallVector<std::string, 7>({"sin", "cos", "_ZSt3cos", "_ZSt3sin", "asin", "acos", "abs"});
  for (auto &name : names) {
    if (F->getName().startswith(name)) {
      return true && enabled;
    }
  }
  return false && enabled;
}
} // namespace TaffoMath