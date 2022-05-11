#include "MemoryGraph.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

namespace taffo
{

using namespace llvm;

MemoryGraph::MemoryGraph(llvm::Module &Module): M{Module}
{
  seedRoots();
  makeGraph();
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
  errs() << "assignOrGetIndex:" << "\n";
  errs() << "-------" << "\n";
  auto it = instToIndex.find(Inst);
  if (it != instToIndex.end()) {
    errs() << "index exists:" << "\n";
    errs() << it->second << "\n";
    errs() << *(Inst->value) << "\n";
    return it->second;
  } else {
    index++;
    instToIndex[Inst] = index;
    indexToInst[index] = Inst;
    errs() << "new index:" << "\n";
    errs() << index << "\n";
    errs() << *(Inst->value) << "\n";
    return index;
  }
}

void MemoryGraph::queuePush(llvm::Value *V)
{
  if (!isVisited(V)) {
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
  auto &argType = callInst->getOperandUse(argNo);
  auto *fun = callInst->getCalledFunction();
  if (!fun) {
    // conservatively consider all unknown functions with pointer arg as external
    return argType->getType()->isPointerTy();
  }
  if (argType->getType()->isPointerTy() && fun->getBasicBlockList().empty()) {
    // this is an external function, don't touch it
    return true;
  }
  return false;
}

std::shared_ptr<ValueWrapper> MemoryGraph::wrapValue(llvm::Value *V)
{
  return std::make_shared<InstWrapper>(V);
}

std::shared_ptr<ValueWrapper> MemoryGraph::wrapValueUse(llvm::Use *V)
{
  std::shared_ptr<ValueWrapper> wrapper;
  auto *callInst = dyn_cast<CallInst>(V->getUser());
  if (callInst && !isMallocLike(callInst)) {
    wrapper = std::make_shared<FunCallArgWrapper>(
        callInst, V->getOperandNo(), isExternalCallWithPointer(callInst, V->getOperandNo()));
  } else {
    wrapper = std::make_shared<InstWrapper>(V->getUser());
  }
  return wrapper;
}

void MemoryGraph::addToGraph(std::shared_ptr<ValueWrapper> src, std::shared_ptr<ValueWrapper> dst)
{
//  errs() << "adding to graph:" << "\n";
//  errs() << "-------:" << "\n";
//  errs() << *src->value << "\n";
//  errs() << *dst->value << "\n";
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
    }
    markVisited(Inst);
  }
}

void MemoryGraph::addUsesToGraph(llvm::Value *V)
{
  auto srcWrapper = wrapValue(V);
  for (auto &Inst: V->uses()) {
    auto *dstInst = Inst.getUser();
    auto dstWrapper = wrapValueUse(&Inst);
    addToGraph(srcWrapper, dstWrapper);
    queuePush(dstInst);
  }
}

void MemoryGraph::handleAllocaInst(llvm::AllocaInst *allocaInst)
{
  addUsesToGraph(allocaInst);
}

void MemoryGraph::handleStoreInst(llvm::StoreInst *storeInst)
{
  auto *dstInst = storeInst->getValueOperand();
  auto srcWrapper = wrapValue(storeInst);
  auto dstWrapper = wrapValue(dstInst);
  addToGraph(srcWrapper, dstWrapper);
  queuePush(dstInst);
}

void MemoryGraph::handleLoadInst(llvm::LoadInst *loadInst)
{
  if (loadInst->getType()->isPointerTy()) {
    addUsesToGraph(loadInst);
  }
}

void MemoryGraph::handleGEPInst(llvm::GetElementPtrInst *gepInst)
{
  addUsesToGraph(gepInst);
}

} // namespace taffo