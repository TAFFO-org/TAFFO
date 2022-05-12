#ifndef TAFFO_MEMORYGRAPH_H
#define TAFFO_MEMORYGRAPH_H

#include <list>
#include <llvm/IR/Instructions.h>
#include <memory>
#include <unordered_map>

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "TracingUtils.h"


namespace taffo {

class MemoryGraph
{
public:
  MemoryGraph(llvm::Module &Module);

  const std::list<std::pair<int, int>>& getEdges() {
    return edges;
  }

  int getNodeCount() {
    return index + 1;
  }

  std::shared_ptr<ValueWrapper> getNode(int i) {
    return indexToInst.at(i);
  }

private:
  llvm::Module &M;

  std::unordered_map<std::shared_ptr<ValueWrapper>, int,
                     std::hash<std::shared_ptr<taffo::ValueWrapper>>,
                     std::equal_to<std::shared_ptr<taffo::ValueWrapper>>> instToIndex;
  std::unordered_map<int, std::shared_ptr<ValueWrapper>> indexToInst;
  std::list<std::pair<int, int>> edges;
  std::list<llvm::Value*> queue;
  std::unordered_map<llvm::Value*, bool> visited;
  int index = -1;

  int assignOrGetIndex(std::shared_ptr<ValueWrapper> Inst);
  void seedRoots();
  void makeGraph();
  void addToGraph(std::shared_ptr<ValueWrapper> src, std::shared_ptr<ValueWrapper> dst);
  void queuePush(llvm::Value* V);
  llvm::Value* queuePop();
  void markVisited(llvm::Value* V);
  bool isVisited(llvm::Value* V);
  void handleAllocaInst(llvm::AllocaInst* allocaInst);
  void handleStoreInst(llvm::StoreInst* storeInst);
  void handleLoadInst(llvm::LoadInst* loadInst);
  void handleGEPInst(llvm::GetElementPtrInst* gepInst);
  void handleMallocLikeInst(llvm::CallInst* mallocLikeInst);
  void handleGlobalVar(llvm::GlobalVariable* globalVariable);
  void handlePointerCastInst(llvm::CastInst* castInst);
  void handlePtrToIntCast(llvm::PtrToIntInst* ptrToIntInst);
  void addUsesToGraph(llvm::Value* V);
};

} // namespace taffo

#endif // TAFFO_MEMORYGRAPH_H
