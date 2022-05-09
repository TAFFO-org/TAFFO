#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Instructions.h"
#include <vector>

namespace tuner {
  
  class MemWatcher
  {
  private:
    llvm::DenseMap<llvm::Value *, std::vector<llvm::LoadInst *>> pairsToClose;


  public:
    void openPhiLoop(llvm::LoadInst *phiNode, llvm::Value *requestedValue);

    llvm::LoadInst *getPhiNodeToClose(llvm::Value *value);

    void closePhiLoop(llvm::LoadInst *phiNode, llvm::Value *requestedNode);

    void dumpState();
  };

}
