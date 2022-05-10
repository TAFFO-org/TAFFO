#include "MemoryGraph.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"

namespace taffo
{

using namespace llvm;

MemoryGraph::MemoryGraph(llvm::Module &Module): M{Module}
{
  seedRoots();
}

bool MemoryGraph::isMallocLike(const llvm::Function *F) const
{
  const llvm::StringRef FName = F->getName();
  // TODO make sure this works in other platforms
  return FName == "malloc" || FName == "calloc" || FName == "_Znwm" || FName == "_Znam";
}

bool MemoryGraph::isMallocLike(const llvm::Value *Inst) const
{
  if (auto *callInst = dyn_cast<CallInst>(Inst)) {
    auto *callee = callInst->getCalledFunction();
    return callee && isMallocLike(callee);
  }
  return false;
}

int MemoryGraph::assignOrGetIndex(std::shared_ptr<ValueWrapper> Inst)
{
  auto it = instToIndex.find(Inst);
  if (it != instToIndex.end()) {
    return it->second;
  } else {
    index++;
    instToIndex[Inst] = index;
    indexToInst[index] = Inst;
    return index;
  }
}

void MemoryGraph::queuePush(llvm::Value *V)
{
  if (isVisited(V)) {
    queue.push_back(V);
  }
}

llvm::Value *MemoryGraph::queuePop()
{
  auto* V = queue.front();
  queue.pop_front();
  return V;
}

void MemoryGraph::seedRoots()
{
  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration()) {
      continue;
    }
    for (auto &BB : F.getBasicBlockList()) {
      for (auto &Inst : BB.getInstList()) {
        if (Inst.isDebugOrPseudoInst()) {
          continue;
        }
        if (auto *allocaInst = dyn_cast<AllocaInst>(&Inst)) {
          queuePush(allocaInst);
        } else if (isMallocLike(&Inst)) {
          queuePush(&Inst);
        }
      }
    }
  }
}

bool MemoryGraph::isExternalCallWithPointer(const llvm::CallInst *callInst, int argNo) const
{
  auto *arg = callInst->getOperand(argNo);
  auto *fun = callInst->getCalledFunction();
  if (!fun) {
    // conservatively consider all unknown functions with pointer arg as external
    return arg->getType()->isPointerTy();
  }
  if (arg->getType()->isPointerTy() && fun->getBasicBlockList().empty()) {
    // this is an external function, don't touch it
    return true;
  }
  return false;
}

std::shared_ptr<ValueWrapper> MemoryGraph::wrapValue(llvm::Value *V, int argNo)
{
  std::shared_ptr<ValueWrapper> wrapper;
  auto *callInst = dyn_cast<CallInst>(V);
  if (callInst && !isMallocLike(callInst)) {
    wrapper = std::make_shared<FunCallArgWrapper>(
        callInst, argNo, isExternalCallWithPointer(callInst, argNo));
  } else {
    wrapper = std::make_shared<InstWrapper>(V);
  }
  return wrapper;
}

void MemoryGraph::addToGraph(std::shared_ptr<ValueWrapper> src, std::shared_ptr<ValueWrapper> dst)
{
  auto srcIndex = assignOrGetIndex(src);
  auto dstIndex = assignOrGetIndex(dst);
  edges.emplace_back(srcIndex, dstIndex);
}

void MemoryGraph::markVisited(llvm::Value *V)
{
  visited[V] = true;
}

bool MemoryGraph::isVisited(llvm::Value *V)
{
  return visited.count(V) > 0;
}

void MemoryGraph::makeGraph()
{
  while (!queue.empty()) {
    auto* Inst = queuePop();

    if (isVisited(Inst)) {
      continue ;
    }
    if (auto *allocaInst = dyn_cast<AllocaInst>(Inst)) {
      handleAllocaInst(allocaInst);
    } else if (auto *storeInst = dyn_cast<StoreInst>(Inst)) {
      handleStoreInst(storeInst);
    } else if (auto *loadInst = dyn_cast<LoadInst>(Inst)) {
      handleLoadInst(loadInst);
    } else if (auto *gepInst = dyn_cast<GetElementPtrInst>(Inst)) {
      handleGEPInst(gepInst);
    } else {
      markVisited(Inst);
    }
  }
}

void MemoryGraph::addUsesToGraph(llvm::Value *V)
{
  auto srcWrapper = wrapValue(V);
  for (auto &Inst: V->uses()) {
    auto *dstInst = Inst.getUser();
    auto dstWrapper = wrapValue(dstInst, Inst.getOperandNo());
    addToGraph(srcWrapper, dstWrapper);
    queuePush(dstInst);
  }
}

void MemoryGraph::handleAllocaInst(llvm::AllocaInst *allocaInst)
{
  addUsesToGraph(allocaInst);
  markVisited(allocaInst);
}

void MemoryGraph::handleStoreInst(llvm::StoreInst *storeInst)
{
  auto *dstInst = storeInst->getValueOperand();
  auto srcWrapper = wrapValue(storeInst);
  auto dstWrapper = wrapValue(dstInst);
  addToGraph(srcWrapper, dstWrapper);
  markVisited(storeInst);
  queuePush(dstInst);
}

void MemoryGraph::handleLoadInst(llvm::LoadInst *loadInst)
{
  markVisited(loadInst);
}

void MemoryGraph::handleGEPInst(llvm::GetElementPtrInst *gepInst)
{
  addUsesToGraph(gepInst);
  markVisited(gepInst);
}

} // namespace taffo