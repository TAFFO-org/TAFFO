#ifndef TAFFO_MEMORYGRAPH_H
#define TAFFO_MEMORYGRAPH_H

#include <list>
#include <memory>
#include <unordered_map>

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include "TracingUtils.h"

extern llvm::cl::opt<bool> Fixm;

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

  void print_graph() {
    llvm::dbgs() << "MEMORY GRAPH BEGIN" << "\n";
    for (auto x : edges) {
      llvm::dbgs() << "-------:" << "\n";
      llvm::dbgs() << "src: ";
      getNode(x.first)->print_debug(llvm::dbgs()) << "\n";
      llvm::dbgs() << "dst: ";
      getNode(x.second)->print_debug(llvm::dbgs()) << "\n";
    }
    llvm::dbgs() << "MEMORY GRAPH END" << "\n";
  }

  void print_connected_components(const std::unordered_map<int, std::list<int>>& cc) {
    llvm::dbgs() << "CONNECTED COMPONENTS BEGIN" << "\n";
    for (auto &it : cc) {
      std::list<int> l = it.second;
      llvm::dbgs() << "-------:" << "\n";
      llvm::dbgs() << "CONNECTED COMPONENT " << it.first << "\n";
      for (auto x : l) {
        getNode(x)->print_debug(llvm::dbgs()) << "\n";
      }
    }
    llvm::dbgs() << "CONNECTED COMPONENTS END" << "\n";
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
  void handlePtrToIntCast(const std::shared_ptr<ValueWrapper>& src, llvm::PtrToIntInst* ptrToIntInst, llvm::Use* UseObject);
  void handleGenericInst(const std::shared_ptr<ValueWrapper> &srcWrapper, llvm::Value *UseInst, llvm::Use *UseObject);
  void handleFuncArg(const std::shared_ptr<ValueWrapper> &srcWrapper, llvm::CallBase *callSite, llvm::Use *UseObject);
  void handleCastInst(const std::shared_ptr<ValueWrapper> &srcWrapper, llvm::CastInst *castInst, llvm::Use *UseObject);
  void handleGEPInst(const std::shared_ptr<ValueWrapper> &srcWrapper, llvm::Value *gepInst, llvm::Use *UseObject);
  void handleGenericWrapper(const std::shared_ptr<ValueWrapper> &srcWrapper, const std::shared_ptr<ValueWrapper> &dstWrapper);
};

} // namespace taffo

#endif // TAFFO_MEMORYGRAPH_H
