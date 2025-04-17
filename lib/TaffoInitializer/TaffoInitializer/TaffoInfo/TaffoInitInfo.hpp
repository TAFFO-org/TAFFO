#pragma once

#include "ValueInitInfo.hpp"

namespace taffo {

class InitializerPass;

class TaffoInitInfo {
private:
  friend class InitializerPass;

  llvm::DenseMap<llvm::Value*, ValueInitInfo> valueInitInfo;

  ValueInitInfo& getValueInitInfo(llvm::Value* value);
  ValueInitInfo& getOrCreateValueInitInfo(llvm::Value* value);
  ValueInitInfo&
  createValueInitInfo(llvm::Value* value, unsigned int rootDistance = UINT_MAX, unsigned int backtrackingDepth = 0);
  bool hasValueInitInfo(const llvm::Value* value) const;
};

} // namespace taffo
