#include "MemoryGraph.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Constant.h"
#include "ConstantsContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "TracingUtils.h"
#include "TaffoMathUtil.h"
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
//    llvm::dbgs() << "-------:" << "\n";
    llvm::dbgs() << "queue push: ";
    V->print_debug(llvm::dbgs()) << "\n";
//    llvm::dbgs() << "-------:" << "\n";
  }
}

std::shared_ptr<ValueWrapper> MemoryGraph::queuePop()
{
  auto V = queue.front();
  queue.pop_front();
//  llvm::dbgs() << "-------:" << "\n";
  llvm::dbgs() << "queue pop: ";
  V->print_debug(llvm::dbgs()) << "\n";
//  llvm::dbgs() << "-------:" << "\n";
  return V;
}

void MemoryGraph::queuePushStruct(Value* V) {
  auto* structType = fullyUnwrapPointerOrArrayType(V->getType());
  for (unsigned int i = 0; i < structType->getStructNumElements(); i++) {
    queuePush(ValueWrapper::wrapStructElem(V, i, structType));
  }
}

void MemoryGraph::seedRoots()
{
  for (llvm::GlobalVariable &V : M.globals()) {
    if (isStructType(V.getValueType())) {
      if (V.hasInitializer()) {
        queuePushStruct(&V);
      }
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
//  llvm::dbgs() << "-------:" << "\n";
  llvm::dbgs() << "adding to graph:" << "\n";
  llvm::dbgs() << "src: ";
  src->print_debug(llvm::dbgs()) << "\n";
  llvm::dbgs() << "dst: ";
  dst->print_debug(llvm::dbgs()) << "\n";
//  llvm::dbgs() << "-------:" << "\n";
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

    if (isVisited(wrappedInst)) {
      continue ;
    }

    for (auto &UseObject : wrappedInst->uses()) {
      auto *UserInst = UseObject.getUser();
      if (auto *storeInst = dyn_cast<StoreInst>(UserInst)) {
        handleStoreInst(wrappedInst, storeInst, &UseObject);
      } else if (isa<GetElementPtrInst, GetElementPtrConstantExpr>(UserInst)) {
        handleGEPInst(wrappedInst, UserInst, &UseObject);
      } else if (auto *ptrToIntCastInst = dyn_cast<PtrToIntInst>(UserInst)) {
        handlePtrToIntCast(wrappedInst, ptrToIntCastInst, &UseObject);
      } else if (isa<CallInst, InvokeInst>(UserInst)) {
        auto *callSite = dyn_cast<CallBase>(UserInst);
        handleFuncArg(wrappedInst, callSite, &UseObject);
      } else if (auto *castInst = dyn_cast<CastInst>(UserInst)) {
        handleCastInst(wrappedInst, castInst, &UseObject);
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

Type* getStructElemStructType(const std::shared_ptr<ValueWrapper>& srcWrapper) {
  if (srcWrapper->isStructElem()) {
    auto *structElemWrapper = static_cast<taffo::StructElemWrapper *>(&(*srcWrapper));
    return structElemWrapper->structType;
  } else if (srcWrapper->isStructElemFunCall()) {
    auto *structElemWrapper = static_cast<taffo::StructElemFunCallArgWrapper *>(&(*srcWrapper));
    return structElemWrapper->structType;
  } else {
    llvm::dbgs() << "Not a struct element: " << *(srcWrapper->value)
                 << "\n";
    return nullptr;
  }
}

std::shared_ptr<ValueWrapper> matchSrcWrapper(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::Value *UseInst) {
  auto srcBaseType = fullyUnwrapPointerOrArrayType(srcWrapper->value->getType());
  llvm::dbgs() << "MatchSrcWrapper>>>>>>>>>>\nMatchSrcWrapper type: " << "\n" <<
      *fullyUnwrapPointerOrArrayType(UseInst->getType()) << "\n" <<
      *fullyUnwrapPointerOrArrayType(srcWrapper->value->getType()) <<
      "\n";
  std::shared_ptr<ValueWrapper> dstWrapper;
  if (srcBaseType->isStructTy() &&
      (srcWrapper->isStructElem() || srcWrapper->isStructElemFunCall())) {
    if (!isa<Argument>(UseInst)) {
      dstWrapper = ValueWrapper::wrapStructElem(UseInst, getStructElemArgPos(srcWrapper), getStructElemStructType(srcWrapper));
    } else {
      auto *argument = dyn_cast<Argument>(UseInst);
      dstWrapper = ValueWrapper::wrapStructElemFunCallArg(argument->getParent(), getStructElemArgPos(srcWrapper), argument->getArgNo(), getStructElemStructType(srcWrapper));
    }
  } else {
    // src is not a struct, so create a plain value
    dstWrapper = ValueWrapper::wrapValue(UseInst);
  }
  return dstWrapper;
}

void MemoryGraph::handleGenericInst(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::Value *UseInst, llvm::Use* UseObject) {
  auto srcBaseType = fullyUnwrapPointerOrArrayType(srcWrapper->value->getType());
  auto dstBaseType = fullyUnwrapPointerOrArrayType(UseInst->getType());
  llvm::dbgs() << "handleGenericInst>>>>>>>>>>\nhandleGenericInst type: " << "\n" <<
      *fullyUnwrapPointerOrArrayType(UseInst->getType()) << "\n" <<
      *fullyUnwrapPointerOrArrayType(srcWrapper->value->getType()) <<
      "\n";
  if (srcBaseType->isStructTy() && (srcWrapper->isStructElem() || srcWrapper->isStructElemFunCall())) {
    // checking this because gep accessing fields will have a StructElemWrapper but not a struct register type as it
    // gets a field of a struct which is not itself a struct
    if (srcBaseType == dstBaseType) {
      // we are not yet accessing struct fields, just unwrapping pointersss
      // this will fail on recursive structs
      auto dstWrapper = matchSrcWrapper(srcWrapper, UseInst);
      handleGenericWrapper(srcWrapper, dstWrapper);
    } else {
      // no else because it means a nested struct, which is not supported
      llvm::dbgs() << "Unsupported case: Nested struct?" << "\n";
    }
  } else if (srcWrapper->value->getType()->isPointerTy() || isa<StoreInst>(UseInst)) {
    auto dstWrapper = ValueWrapper::wrapValue(UseInst);
    llvm::dbgs() << "-------:" << "\n";
    llvm::dbgs() << "handleGenericInst" << "\n";
    addToGraph(srcWrapper, dstWrapper);
    queuePush(dstWrapper);
    llvm::dbgs() << "-------:" << "\n";
  }
}

void MemoryGraph::handleGenericWrapper(const std::shared_ptr<ValueWrapper>& srcWrapper, const std::shared_ptr<ValueWrapper>& dstWrapper) {
//  if (srcWrapper->value->getType()->isPointerTy()) {
    llvm::dbgs() << "-------:" << "\n";
    llvm::dbgs() << "handleGenericWrapper" << "\n";
    addToGraph(srcWrapper, dstWrapper);
    queuePush(dstWrapper);
    llvm::dbgs() << "-------:" << "\n";
//  }
}

void MemoryGraph::handleCastInst(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::CastInst *castInst, llvm::Use* UseObject) {
  if (castInst->getDestTy()->isFloatingPointTy() || castInst->getDestTy()->isIntegerTy()) {
    auto dstWrapper = matchSrcWrapper(srcWrapper, castInst);
    llvm::dbgs() << "-------:" << "\n";
    llvm::dbgs() << "handleCastInst" << "\n";
    addToGraph(srcWrapper, dstWrapper);
    queuePush(dstWrapper);
    llvm::dbgs() << "-------:" << "\n";
  } else {
    handleGenericInst(srcWrapper, castInst, UseObject);
  }
}

void MemoryGraph::handleStoreInst(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::StoreInst *storeInst, llvm::Use* UseObject)
{
  auto *storeValueInst = storeInst->getValueOperand();
  if (storeValueInst == srcWrapper->value && !(srcWrapper->isFunCallArg() || srcWrapper->isStructElemFunCall())) {
    // prevent double-adding store from different paths
    return;
  }
  // first, store the link between the source and the store
  auto storeWrapper = matchSrcWrapper(srcWrapper, storeInst);
  handleGenericWrapper(srcWrapper, storeWrapper);
  // second, store the link between the store and the stored value because it is not a use
  std::shared_ptr<ValueWrapper> storeValueWrapper;
  if (storeValueInst->getType()->isPointerTy()) {
    // try to detect when we store structs
    storeValueWrapper = matchSrcWrapper(srcWrapper, storeValueInst);
  } else {
    storeValueWrapper = ValueWrapper::wrapValue(storeValueInst);
  }

  llvm::dbgs() << "-------:" << "\n";
  llvm::dbgs() << "handleStoreInst" << "\n";
  addToGraph(storeValueWrapper, storeWrapper);
  queuePush(storeValueWrapper);
  llvm::dbgs() << "-------:" << "\n";
}

void MemoryGraph::handleFuncArg(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::CallBase *callSite, llvm::Use* UseObject)
{
  auto argNo = callSite->getArgOperandNo(UseObject);
  auto *fun = callSite->getCalledFunction();
  if (fun &&
   (!fun->getBasicBlockList().empty() || TaffoMath::isSupportedLibmFunction(fun, Fixm)) &&
   !fun->isVarArg()) {
    llvm::dbgs() << "Arg: " << *callSite << "\nfunction: " << fun->getName() << "\nargNo: " << argNo << "\n";
    std::shared_ptr<ValueWrapper> dstArgWrapper;
    if (srcWrapper->isStructElem() || srcWrapper->isStructElemFunCall()) {
      dstArgWrapper = ValueWrapper::wrapStructElemFunCallArg(callSite->getCalledFunction(), getStructElemArgPos(srcWrapper), argNo, getStructElemStructType(srcWrapper));
    } else {
      dstArgWrapper = ValueWrapper::wrapFunCallArg(callSite->getCalledFunction(), argNo);
    }
    llvm::dbgs() << "-------:" << "\n";
    llvm::dbgs() << "handleFuncArg" << "\n";
    addToGraph(srcWrapper, dstArgWrapper);
    queuePush(dstArgWrapper);
    llvm::dbgs() << "-------:" << "\n";
  }
}


void MemoryGraph::handleGEPInst(const std::shared_ptr<ValueWrapper>& srcWrapper, llvm::Value *gepInst, llvm::Use* UseObject)
{
  if (srcWrapper->isStructElem() || srcWrapper->isStructElemFunCall()) {
    llvm::dbgs() << "GEP>>>>>>>>>>\nGEP inst type: " << "\n" <<
        *fullyUnwrapPointerOrArrayType(gepInst->getType()) << "\n" <<
        *fullyUnwrapPointerOrArrayType(srcWrapper->value->getType()) <<
        "\n";
    auto srcBaseType = fullyUnwrapPointerOrArrayType(srcWrapper->value->getType());
    auto dstBaseType = fullyUnwrapPointerOrArrayType(gepInst->getType());
    if (srcBaseType == dstBaseType) {
      // we are not yet accessing struct fields, just unwrapping pointers
      // this will fail on recursive structs
      auto gepWrapper = matchSrcWrapper(srcWrapper, gepInst);
      handleGenericWrapper(srcWrapper, gepWrapper);
    } else {
      // we are accessing a struct field, only match the field, not the whole struct
      ConstantInt *CI;
      if (auto *gep = dyn_cast<GetElementPtrConstantExpr>(gepInst)) {
        CI = dyn_cast<ConstantInt>(gep->getOperand(2));
      } else if (auto *gep = dyn_cast<GetElementPtrInst>(gepInst)) {
        CI = dyn_cast<ConstantInt>(gep->getOperand(2));
      }
      if (CI) {
        uint64_t Idx = CI->getZExtValue();
        if (Idx == getStructElemArgPos(srcWrapper)) {
          auto dstWrapper = matchSrcWrapper(srcWrapper, gepInst);
          llvm::dbgs() << "-------:" << "\n";
          llvm::dbgs() << "handleGEPInst" << "\n";
          addToGraph(srcWrapper, dstWrapper);
          queuePush(dstWrapper);
          llvm::dbgs() << "-------:" << "\n";
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
        llvm::dbgs() << "-------:" << "\n";
        llvm::dbgs() << "handlePtrToIntCast" << "\n";
        addToGraph(srcWrapper, dstWrapper);
        queuePush(dstWrapper);
        llvm::dbgs() << "-------:" << "\n";
      } else if (localVisited.count(dstInst) == 0) {
        localQueue.push_back(dstInst);
      }
      localVisited[localInst] = true;
    }
  }
}

} // namespace taffo