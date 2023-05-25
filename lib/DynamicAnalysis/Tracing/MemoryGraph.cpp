#include "MemoryGraph.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "TracingUtils.h"
#include "TaffoMathUtil.h"

namespace taffo
{

using namespace llvm;

/*
 * The idea of the memory graph:
 * 1. all pointer uses will be connected as they all share the same range and type
 * 2. all value uses that are equal will be connected: passed as an argument to a function, store
 */

MemoryGraph::MemoryGraph(llvm::Module &Module): M{Module}
{
  seedRoots();
  makeGraph();
}

int MemoryGraph::assignOrGetIndex(std::shared_ptr<ValueWrapper> Inst)
{
//  llvm::dbgs() << "assignOrGetIndex:" << "\n";
//  llvm::dbgs() << "-------" << "\n";
  auto it = instToIndex.find(Inst);
  if (it != instToIndex.end()) {
//    llvm::dbgs() << "index exists:" << "\n";
//    llvm::dbgs() << it->second << "\n";
//    llvm::dbgs() << *(Inst->value) << "\n";
    return it->second;
  } else {
    index++;
    instToIndex[Inst] = index;
    indexToInst[index] = Inst;
//    llvm::dbgs() << "new index:" << "\n";
//    llvm::dbgs() << index << "\n";
//    llvm::dbgs() << *(Inst->value) << "\n";
    return index;
  }
}

void MemoryGraph::queuePush(std::shared_ptr<ValueWrapper> V)
{
  if (!isVisited(V)) {
    queue.push_back(V);
  }
}

std::shared_ptr<ValueWrapper> MemoryGraph::queuePop()
{
  auto V = queue.front();
  queue.pop_front();
  return V;
}

void MemoryGraph::seedRoots()
{
  for (llvm::GlobalVariable &V : M.globals()) {
    queuePush(ValueWrapper::wrapValue(&V));
  }

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration()) {
      continue;
    }
    auto funName = F.getName().str();
    if (funName == "polybench_flush_cache" ||
        funName == "polybench_prepare_instruments" ||
        funName == "polybench_timer_start" ||
        funName == "rtclock" ||
        funName == "polybench_timer_stop" ||
        funName == "polybench_timer_print"
    ) {
      continue ;
    }
    for (auto &BB : F.getBasicBlockList()) {
      for (auto &Inst : BB.getInstList()) {
        if (Inst.isDebugOrPseudoInst()) {
          continue;
        }
        if (auto *allocaInst = dyn_cast<AllocaInst>(&Inst)) {
          queuePush(ValueWrapper::wrapValue(allocaInst));
        } else if (TracingUtils::isMallocLike(&Inst)) {
          queuePush(ValueWrapper::wrapValue(&Inst));
        }
      }
    }
  }
}

void MemoryGraph::addToGraph(std::shared_ptr<ValueWrapper> src, std::shared_ptr<ValueWrapper> dst)
{
//  llvm::dbgs() << "adding to graph:" << "\n";
//  llvm::dbgs() << "-------:" << "\n";
//  llvm::dbgs() << *src->value << "\n";
//  llvm::dbgs() << *dst->value << "\n";
  auto srcIndex = assignOrGetIndex(src);
  auto dstIndex = assignOrGetIndex(dst);
  edges.emplace_back(srcIndex, dstIndex);
}

void MemoryGraph::markVisited(std::shared_ptr<ValueWrapper> V)
{
  visited[V] = true;
}

bool MemoryGraph::isVisited(std::shared_ptr<ValueWrapper> V)
{
  return visited.count(V) > 0;
}

void MemoryGraph::makeGraph()
{
  while (!queue.empty()) {
    auto wrappedInst = queuePop();
    auto* Inst = wrappedInst->value;

    if (isVisited(wrappedInst)) {
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
      } else if (castInst->getDestTy()->isFloatingPointTy() || castInst->getDestTy()->isIntegerTy()) {
        addUsesToGraph(Inst);
      }
    } else if (auto *funArg = dyn_cast<Argument>(Inst)) {
      addUsesToGraph(funArg);
    } else {
      addUsesToGraph(Inst);
    }
    markVisited(wrappedInst);
  }
}

void MemoryGraph::addUsesToGraph(llvm::Value *V)
{
  auto srcWrapper = ValueWrapper::wrapValue(V);
  for (auto &Inst: V->uses()) {
    auto *dstInst = Inst.getUser();
    // passing as an argument to a function is fine for both pointers and values
    if (isa<CallInst, InvokeInst>(dstInst)) {
      auto *callSite = dyn_cast<CallBase>(dstInst);
      auto argNo = callSite->getArgOperandNo(&Inst);
      auto *fun = callSite->getCalledFunction();
      if (fun &&
          (!fun->getBasicBlockList().empty() || TaffoMath::isSupportedLibmFunction(fun, Fixm)) &&
          !fun->isVarArg()) {
        llvm::dbgs() << "Arg: " << *V << "\nfunction: " << fun->getName() << "\nargNo: " << argNo << "\n";
        auto *formalArg = fun->getArg(argNo);
        auto dstArgWrapper = ValueWrapper::wrapValue(formalArg);
        addToGraph(srcWrapper, dstArgWrapper);
        queuePush(dstArgWrapper);
      }
    // the rest of uses only are fine if it's a pointer or a store or a cast
    } else if (V->getType()->isPointerTy() || isa<StoreInst>(dstInst) || isa<CastInst>(dstInst)) {
      auto dstWrapper = ValueWrapper::wrapValueUse(&Inst);
      queuePush(dstWrapper);
      addToGraph(srcWrapper, dstWrapper);
    }
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
  auto srcWrapper = ValueWrapper::wrapValue(storeInst);
  auto dstWrapper = ValueWrapper::wrapValue(dstInst);
  addToGraph(srcWrapper, dstWrapper);
  queuePush(dstWrapper);
}

void MemoryGraph::handleLoadInst(llvm::LoadInst *loadInst)
{
  addUsesToGraph(loadInst);
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