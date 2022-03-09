#pragma once

#include "llvm/IR/Value.h"
#include <string>

namespace tuner {

  /// Utility function that generates an unique string ID from a value
  std::string uniqueIDForValue(llvm::Value *value);

}

