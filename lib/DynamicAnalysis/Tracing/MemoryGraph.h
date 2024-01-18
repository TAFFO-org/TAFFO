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
  std::list<std::shared_ptr<ValueWrapper>> queue;
  std::unordered_map<std::shared_ptr<ValueWrapper>, bool,
                     std::hash<std::shared_ptr<taffo::ValueWrapper>>,
                     std::equal_to<std::shared_ptr<taffo::ValueWrapper>>> visited;
  int index = -1;

  int assignOrGetIndex(std::shared_ptr<ValueWrapper> Inst);
  void seedRoots();
  void makeGraph();
  void addToGraph(std::shared_ptr<ValueWrapper> src, std::shared_ptr<ValueWrapper> dst);
  void queuePush(std::shared_ptr<ValueWrapper> V);
  std::shared_ptr<ValueWrapper> queuePop();
  void queuePushStruct(llvm::Value *V);
  void markVisited(std::shared_ptr<ValueWrapper> V);
  bool isVisited(std::shared_ptr<ValueWrapper> V);
  void handleStoreInst(const std::shared_ptr<ValueWrapper>& src, llvm::StoreInst* storeInst, llvm::Use* UseObject);
  void handleGEPInst(const std::shared_ptr<ValueWrapper>& src, llvm::GetElementPtrInst* gepInst, llvm::Use* UseObject);
  void handlePtrToIntCast(const std::shared_ptr<ValueWrapper>& src, llvm::PtrToIntInst* ptrToIntInst, llvm::Use* UseObject);
  void handleGenericInst(const std::shared_ptr<ValueWrapper> &srcWrapper, llvm::Value *UseInst, llvm::Use *UseObject);
  void handleFuncArg(const std::shared_ptr<ValueWrapper> &srcWrapper, llvm::CallBase *callSite, llvm::Use *UseObject);
};

} // namespace taffo

#endif // TAFFO_MEMORYGRAPH_H
