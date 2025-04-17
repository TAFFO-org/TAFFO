#pragma once

#include "Containers/BiMap.hpp"

#include <llvm/Analysis/LoopInfo.h>

namespace taffo {

class MetadataManager {
private:
  MetadataManager() = default;
  static MetadataManager& getInstance();

public:
  MetadataManager(const MetadataManager&) = delete;
  MetadataManager& operator=(const MetadataManager&) = delete;

  static void setIdValueMapping(const BiMap<std::string, llvm::Value*>& idValueMapping, llvm::Module& m);
  static BiMap<std::string, llvm::Value*> getIdValueMapping(llvm::Module& m);

  static void setIdLoopMapping(const BiMap<std::string, llvm::Loop*>& idLoopMapping, llvm::Module& m);
  static BiMap<std::string, llvm::Loop*> getIdLoopMapping(llvm::Module& m);

  static void setIdTypeMapping(const BiMap<std::string, llvm::Type*>&, llvm::Module& m);
  static BiMap<std::string, llvm::Type*> getIdTypeMapping(llvm::Module& m);

  static void getCudaKernels(llvm::Module& m, llvm::SmallVectorImpl<llvm::Function*> kernels);
};

} // namespace taffo
