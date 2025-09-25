#pragma once

#include "ValueInitInfo.hpp"

namespace taffo {

class TaffoInitInfo {
  friend class InitializerPass;

  llvm::DenseMap<llvm::Value*, ValueInitInfo> valueInitInfo;

  ValueInitInfo& getValueInitInfo(const llvm::Value* value);
  ValueInitInfo& getOrCreateValueInitInfo(llvm::Value* value);
  ValueInitInfo& createValueInitInfo(llvm::Value* value, unsigned rootDistance = UINT_MAX);
  bool hasValueInitInfo(const llvm::Value* value) const;
};

} // namespace taffo
