#include "MemoryGraph.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Constant.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "TracingUtils.h"
#include "TypeUtils.h"

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

void MemoryGraph::queuePushStruct(Value* V) {
  auto* structType = fullyUnwrapPointerOrArrayType(V->getType());
  for (unsigned int i = 0; i < structType->getStructNumElements(); i++) {
    queuePush(ValueWrapper::wrapStructElem(V, i));
  }
}

void MemoryGraph::seedRoots()
{
  for (llvm::GlobalVariable &V : M.globals()) {
    if (isStructType(V.getValueType())) {
      queuePushStruct(&V);
    } else {
      queuePush(ValueWrapper::wrapValue(&V));
    }
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
          if (isStructType(allocaInst->getAllocatedType())) {
            queuePushStruct(allocaInst);
          } else {
            queuePush(ValueWrapper::wrapValue(allocaInst));
          }
        } else if (TracingUtils::isMallocLike(&Inst)) {
          // find the type of malloc from the bitcast
          llvm::Type *bitcastType = nullptr;
          for (auto &UseObject : Inst.uses()) {
            if (auto *bitcastInst = dyn_cast<BitCastInst>(UseObject)) {
              bitcastType = bitcastInst->getDestTy();
              break;
            }
          }
          if (bitcastType) {
            if (isStructType(bitcastType)) {
              queuePushStruct(&Inst);
            } else {
              queuePush(ValueWrapper::wrapValue(&Inst));
            }
          }
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

    for (auto &UseObject : Inst->uses()) {
      auto *UserInst = UseObject.getUser();
      if (auto *storeInst = dyn_cast<StoreInst>(UserInst)) {
        handleStoreInst(wrappedInst, storeInst, &UseObject);
      } else if (auto *gepInst = dyn_cast<GetElementPtrInst>(UserInst)) {
        handleGEPInst(wrappedInst, gepInst, &UseObject);
      } else if (auto *ptrToIntCastInst = dyn_cast<PtrToIntInst>(UserInst)) {
        handlePtrToIntCast(wrappedInst, ptrToIntCastInst, &UseObject);
      } else if (isa<CallInst, InvokeInst>(UserInst)) {
        auto *callSite = dyn_cast<CallBase>(UserInst);
        handleFuncArg(wrappedInst, callSite, &UseObject);
      } else {
        handleGenericInst(wrappedInst, UserInst, &UseObject);
      }
    }

    markVisited(wrappedInst);
  }
}

unsigned int getStructElemArgPos(const std::shared_ptr<ValueWrapper>& srcWrapper) {
  if (srcWrapper->isStructElem()) {
    auto *structElemWrapper = static_cast<taffo::StructElemWrapper *>(&(*srcWrapper));
    return structElemWrapper->argPos;
  } else if (srcWrapper->isStructElemFunCall()) {
    auto *structElemWrapper = static_cast<taffo::StructElemFunCallArgWrapper *>(&(*srcWrapper));
    return structElemWrapper->argPos;
  } else {
    llvm::dbgs() << "Not a struct element: " << *(srcWrapper->value)
                 << "\n";
    return 999999;
  }
}

std::shared_ptr<ValueWrapper> matchSrcWrapper(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::Value *UseInst) {
  std::shared_ptr<ValueWrapper> dstWrapper;
  if (srcWrapper->isStructElem() || srcWrapper->isStructElemFunCall()) {
    dstWrapper = ValueWrapper::wrapStructElem(UseInst, getStructElemArgPos(srcWrapper));
  } else {
    dstWrapper = ValueWrapper::wrapValue(UseInst);
  }
  return dstWrapper;
}

void MemoryGraph::handleGenericInst(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::Value *UseInst, llvm::Use* UseObject) {
  if (UseInst->getType()->isPointerTy() || isa<StoreInst>(UseInst)) {
    auto dstWrapper = matchSrcWrapper(srcWrapper, UseInst);
    queuePush(dstWrapper);
    addToGraph(srcWrapper, dstWrapper);
  }
}

void MemoryGraph::handleStoreInst(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::StoreInst *storeInst, llvm::Use* UseObject)
{
  // first, store the link between the source and the store
  handleGenericInst(srcWrapper, storeInst, UseObject);
  // second, store the link between the store and the stored value because it is not a use

  auto *dstInst = storeInst->getValueOperand();
  auto dstWrapper = matchSrcWrapper(srcWrapper, dstInst);
  if(srcWrapper->isStructElem() || srcWrapper->isStructElemFunCall()) {
    if (dyn_cast<Constant>(storeInst->getPointerOperand())) {
      // llvm currently doesn't support getting a GEP instance out of a ConstantExpr
      llvm::dbgs() << "!Constant in store destination!: " << *(storeInst->getPointerOperand())
                   << ", cannot analyze GEP destination (possibly writing struct field)"
                   << "\n";
    }
  }
  addToGraph(srcWrapper, dstWrapper);
  queuePush(dstWrapper);
}

void MemoryGraph::handleFuncArg(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::CallBase *callSite, llvm::Use* UseObject)
{
  auto argNo = callSite->getArgOperandNo(UseObject);
  auto *fun = callSite->getCalledFunction();
  if (fun && !fun->getBasicBlockList().empty() && !fun->isVarArg()) {
    llvm::dbgs() << "Arg: " << *callSite << "\nfunction: " << fun->getName() << "\nargNo: " << argNo << "\n";
    std::shared_ptr<ValueWrapper> dstArgWrapper;
    if (srcWrapper->isStructElem() || srcWrapper->isFunCallArg()) {
      dstArgWrapper = ValueWrapper::wrapStructElemFunCallArg(callSite, getStructElemArgPos(srcWrapper), argNo);
    } else {
      dstArgWrapper = ValueWrapper::wrapFunCallArg(callSite, argNo);
    }
    addToGraph(srcWrapper, dstArgWrapper);
    queuePush(dstArgWrapper);
  }
}

void MemoryGraph::handleGEPInst(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::GetElementPtrInst *gepInst, llvm::Use* UseObject)
{
  if (srcWrapper->isStructElem() || srcWrapper->isStructElemFunCall()) {
    if (gepInst->getResultElementType()->isPointerTy()) {
      // we are not yet accessing struct fields
      handleGenericInst(srcWrapper, gepInst, UseObject);
    } else {
      // we are accessing a struct field, only match the field, not the whole struct
      if (ConstantInt *CI = dyn_cast<ConstantInt>(gepInst->getOperand(2))) {
        uint64_t Idx = CI->getZExtValue();
        if (Idx == getStructElemArgPos(srcWrapper)) {
          auto dstWrapper = matchSrcWrapper(srcWrapper, gepInst);
          queuePush(dstWrapper);
          addToGraph(srcWrapper, dstWrapper);
        }
      }
    }
  } else {
    handleGenericInst(srcWrapper, gepInst, UseObject);
  }
}

void MemoryGraph::handlePtrToIntCast(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::PtrToIntInst *ptrToIntInst, llvm::Use* UseObject)
{
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
        auto dstWrapper = matchSrcWrapper(srcWrapper, intToPtrInst);
        queuePush(dstWrapper);
        addToGraph(srcWrapper, dstWrapper);
      } else if (localVisited.count(dstInst) == 0) {
        localQueue.push_back(dstInst);
      }
      localVisited[localInst] = true;
    }
  }
}

} // namespace taffo