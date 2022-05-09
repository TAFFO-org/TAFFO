#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Instructions.h"
#include <vector>

namespace tuner {

  // This class contains references to phi node that has no been closed yet
  class PhiWatcher
  {
  private:
    llvm::DenseMap<llvm::Value *, std::vector<llvm::PHINode *>> pairsToClose;

  public:
    void openPhiLoop(llvm::PHINode *phiNode, llvm::Value *requestedValue);

    llvm::PHINode *getPhiNodeToClose(llvm::Value *value);

    void closePhiLoop(llvm::PHINode *phiNode, llvm::Value *requestedNode);

    void dumpState();
  };

}
