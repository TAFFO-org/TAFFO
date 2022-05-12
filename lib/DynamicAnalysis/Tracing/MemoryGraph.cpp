#include "MemoryGraph.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

#include "TracingUtils.h"

namespace taffo
{

using namespace llvm;

MemoryGraph::MemoryGraph(llvm::Module &Module): M{Module}
{
  seedRoots();
  makeGraph();
}

int MemoryGraph::assignOrGetIndex(std::shared_ptr<ValueWrapper> Inst)
{
//  errs() << "assignOrGetIndex:" << "\n";
//  errs() << "-------" << "\n";
  auto it = instToIndex.find(Inst);
  if (it != instToIndex.end()) {
//    errs() << "index exists:" << "\n";
//    errs() << it->second << "\n";
//    errs() << *(Inst->value) << "\n";
    return it->second;
  } else {
    index++;
    instToIndex[Inst] = index;
    indexToInst[index] = Inst;
//    errs() << "new index:" << "\n";
//    errs() << index << "\n";
//    errs() << *(Inst->value) << "\n";
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
  for (llvm::GlobalVariable &V : M.globals()) {
    queuePush(&V);
  }

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
        } else if (TracingUtils::isMallocLike(&Inst)) {
          queuePush(&Inst);
        }
      }
    }
  }
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
    } else if (TracingUtils::isMallocLike(Inst)) {
      handleMallocLikeInst(dyn_cast<CallInst>(Inst));
    } else if (auto *globalVar = dyn_cast<GlobalVariable>(Inst)) {
      handleGlobalVar(globalVar);
    } else if (auto *storeInst = dyn_cast<StoreInst>(Inst)) {
      handleStoreInst(storeInst);
    } else if (auto *loadInst = dyn_cast<LoadInst>(Inst)) {
      handleLoadInst(loadInst);
    } else if (auto *gepInst = dyn_cast<GetElementPtrInst>(Inst)) {
      handleGEPInst(gepInst);
    } else if (auto *castInst = dyn_cast<CastInst>(Inst)) {
      if (castInst->getDestTy()->isPointerTy()) {
        handlePointerCastInst(castInst);
      } else if (auto *ptrToIntCastInst = dyn_cast<PtrToIntInst>(Inst)) {
        handlePtrToIntCast(ptrToIntCastInst);
      }
    }
    markVisited(Inst);
  }
}

void MemoryGraph::addUsesToGraph(llvm::Value *V)
{
  auto srcWrapper = ValueWrapper::wrapValue(V);
  for (auto &Inst: V->uses()) {
    auto *dstInst = Inst.getUser();
    auto dstWrapper = ValueWrapper::wrapValueUse(&Inst);
    addToGraph(srcWrapper, dstWrapper);
    queuePush(dstInst);
  }
}

void MemoryGraph::handleAllocaInst(llvm::AllocaInst *allocaInst)
{
  addUsesToGraph(allocaInst);
}

void MemoryGraph::handleMallocLikeInst(llvm::CallInst *mallocLikeInst)
{
  addUsesToGraph(mallocLikeInst);
}

void MemoryGraph::handleGlobalVar(llvm::GlobalVariable *globalVariable)
{
  addUsesToGraph(globalVariable);
}

void MemoryGraph::handleStoreInst(llvm::StoreInst *storeInst)
{
  auto *dstInst = storeInst->getValueOperand();
  if (dstInst->getType()->isPointerTy()) {
    auto srcWrapper = ValueWrapper::wrapValue(storeInst);
    auto dstWrapper = ValueWrapper::wrapValue(dstInst);
    addToGraph(srcWrapper, dstWrapper);
    queuePush(dstInst);
  }
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

void MemoryGraph::handlePointerCastInst(llvm::CastInst *castInst)
{
  addUsesToGraph(castInst);
}

void MemoryGraph::handlePtrToIntCast(llvm::PtrToIntInst *ptrToIntInst)
{
  auto srcWrapper = ValueWrapper::wrapValue(ptrToIntInst);

  std::list<llvm::Value*> localQueue;
  std::unordered_map<llvm::Value*, bool> localVisited;
  localQueue.push_back(ptrToIntInst);

  while (!localQueue.empty()) {
    auto *localInst = localQueue.front();
    localQueue.pop_front();
    if (localVisited.count(localInst) > 0) {
      continue ;
    }
    for (auto &Inst: localInst->uses()) {
      auto *dstInst = Inst.getUser();
      if (auto *intToPtrInst = dyn_cast<IntToPtrInst>(dstInst)) {
        auto dstWrapper = ValueWrapper::wrapValue(intToPtrInst);
        addToGraph(srcWrapper, dstWrapper);
      } else if (localVisited.count(dstInst) == 0) {
        localQueue.push_back(dstInst);
      }
      localVisited[localInst] = true;
    }
  }
}

} // namespace taffo