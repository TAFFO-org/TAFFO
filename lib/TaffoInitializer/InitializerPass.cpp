#include "CudaKernelPatcher.hpp"
#include "Debug/Logger.hpp"
#include "InitializerPass.hpp"
#include "OpenCLKernelPatcher.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/IR/Argument.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/Transforms/Utils/Cloning.h>

#define DEBUG_TYPE "taffo-init"

using namespace taffo;
using namespace tda;
using namespace llvm;

cl::opt<bool> manualFunctionCloning("manualclone",
                                    cl::desc("Enables function cloning only for annotated functions"),
                                    cl::init(false));

cl::opt<bool> openCLKernelMode("oclkern", cl::desc("Allows cloning of OpenCL kernel functions"), cl::init(false));

cl::opt<bool> cudaKernelMode("cudakern", cl::desc("Allows cloning of Cuda kernel functions"), cl::init(false));

PreservedAnalyses InitializerPass::run(Module& m, ModuleAnalysisManager&) {
  LLVM_DEBUG(log().logln("[InitializerPass]", Logger::Magenta));
  taffoInfo.initializeFromFile(TYPE_DEDUCER_TAFFO_INFO, m);

  if (openCLKernelMode) {
    LLVM_DEBUG(log() << "OpenCLKernelMode == true!\n");
    createOpenCLKernelTrampolines(m);
  }
  else if (cudaKernelMode) {
    LLVM_DEBUG(log() << "CudaKernelMode == true!\n");
    createCudaKernelTrampolines(m);
  }

  manageIndirectCalls(m);

  LLVM_DEBUG(log().logln("[Parsing annotations]", Logger::Blue));
  readAndRemoveGlobalAnnotations(m);
  readAndRemoveLocalAnnotations(m);
  AnnotationCount = infoPropagationQueue.size();
  LLVM_DEBUG(log() << Logger::Green << "Found " << AnnotationCount << " valid annotations\n"
                   << Logger::Reset);

  if (Function* startingPoint = findStartingPointFunctionGlobal(m)) {
    LLVM_DEBUG(
      Logger& logger = log();
      logger.log("Found starting point using global __taffo_vra_starting_function: ");
      logger.logln(startingPoint->getName()););
    taffoInfo.addStartingPoint(*startingPoint);
  }

  LLVM_DEBUG(log().logln("[Propagating info from roots]", Logger::Blue));
  propagateInfo();
  LLVM_DEBUG(
    log().logln("[Results]", Logger::Green);
    logInfoPropagationQueue(););
  saveValueWeights();

  taffoInfo.dumpToFile(INITIALIZER_TAFFO_INFO, m);
  LLVM_DEBUG(log().logln("[End of InitializerPass]", Logger::Magenta));
  return PreservedAnalyses::all();
}

void InitializerPass::saveValueWeights() {
  for (Value* value : infoPropagationQueue) {
    ValueInitInfo& valueInitInfo = taffoInitInfo.getValueInitInfo(value);
    if (isa<Instruction>(value) || isa<GlobalObject>(value))
      taffoInfo.setValueWeight(*value, valueInitInfo.getRootDistance());
  }
}

/**
 * @brief Propagates info from roots.
 *
 * Starting from the set of root values, this function propagates taffoInitInfo
 * through the IR until no new values are discovered
 */
void InitializerPass::propagateInfo() {
  Logger& logger = log();
  LLVM_DEBUG(
    logger.logln("[Initial propagation queue]", Logger::Bold);
    auto indenter = logger.getIndenter();
    indenter.increaseIndent();
    logInfoPropagationQueue(););

  SmallPtrSet<Value*, 8> visited;
  SmallPtrSet<CallBase*, 8> calls;
  unsigned iteration = 0;
  size_t prevQueueSize = 0;
  // Continue propagation until the queue size stabilizes
  while (prevQueueSize < infoPropagationQueue.size()) {
    LLVM_DEBUG(log() << Logger::Blue << "[Propagation iteration " << iteration << "]\n"
                     << Logger::Reset);
    prevQueueSize = infoPropagationQueue.size();
    iteration++;

    for (Value* value : infoPropagationQueue) {
      visited.insert(value);
      if (auto* call = dyn_cast<CallBase>(value))
        calls.insert(call);
      ValueInitInfo& valueInitInfo = taffoInitInfo.getValueInitInfo(value);

      auto indenter = logger.getIndenter();
      LLVM_DEBUG(
        logger.log("[Value] ", Logger::Bold).logValueln(value);
        indenter.increaseIndent();
        logger << "root distance: " << valueInitInfo.getRootDistance() << "\n";
        if (value->user_empty())
          logger.logln("value has no users: continuing"););

      // Process each user of the current value
      for (auto* user : value->users()) {
        // Skip PHI nodes already processed
        if (isa<PHINode>(user) && visited.contains(user))
          continue;

        auto indenter = logger.getIndenter();
        LLVM_DEBUG(
          logger.log("[User] ", Logger::Bold).logValueln(user);
          indenter.increaseIndent(););

        propagateInfo(value, user);

        auto userIter = std::ranges::find(infoPropagationQueue, user);
        if (userIter == infoPropagationQueue.end())
          infoPropagationQueue.push_back(user);
      }
    }
    for (CallBase* call : calls)
      if (!handledCalls.contains(call))
        cloneFunctionForCall(call);
  }
  LLVM_DEBUG(log().logln("[Propagation completed]", Logger::Blue));
}

void InitializerPass::propagateInfo(Value* value, Value* user) {
  if (!taffoInfo.hasValueInfo(*user))
    taffoInfo.createValueInfo(*user);

  ValueInitInfo& srcInitInfo = taffoInitInfo.getValueInitInfo(value);
  ValueInitInfo& dstInitInfo = taffoInitInfo.getOrCreateValueInitInfo(user);
  ValueInfo* srcInfo = &*taffoInfo.getValueInfo(*value);
  ValueInfo* dstInfo = &*taffoInfo.getValueInfo(*user);

  auto* userInst = dyn_cast<Instruction>(user);
  if (!userInst)
    return;

  if (auto* userGep = dyn_cast<GetElementPtrInst>(user)) {
    if (value != userGep->getPointerOperand()) {
      LLVM_DEBUG(log().logln("user is a gep, but value is not its pointer operand: skipping"));
      return;
    }
    if (auto* srcStructInfo = dyn_cast<StructInfo>(srcInfo)) {
      srcInfo = srcStructInfo->getField(userGep->indices());
      if (!srcInfo) {
        LLVM_DEBUG(log().logln("user is a gep, but value struct field has no valueInfo: skipping\n"));
        return;
      }
      LLVM_DEBUG(log() << "propagating info to value struct field based on user gep");
    }
  }
  else if (auto* store = dyn_cast<StoreInst>(user)) {
    // Choose as dst the store operand different from src value
    Value* operand = store->getPointerOperand();
    if (operand == value)
      operand = store->getValueOperand();
    if (!taffoInfo.hasValueInfo(*operand))
      taffoInfo.createValueInfo(*operand);
    dstInitInfo = taffoInitInfo.getOrCreateValueInitInfo(operand);
    dstInfo = &*taffoInfo.getValueInfo(*operand);
    LLVM_DEBUG(
      Logger& logger = log();
      logger << "propagating info from a store operand to the other:\n";
      logger.log("src: ").logValueln(value);
      logger.log("dst: ").logValueln(operand););
  }
  else if (!isa<LoadInst>(user) && !isa<PHINode>(user) && !userInst->isUnaryOp() && !userInst->isBinaryOp()
           && !userInst->isCast() && !isa<AtomicRMWInst>(user))
    return;

  propagateInfo(srcInfo, srcInitInfo, dstInfo, dstInitInfo);
}

void InitializerPass::propagateInfo(const ValueInfo* srcInfo,
                                    const ValueInitInfo& srcInitInfo,
                                    ValueInfo* dstInfo,
                                    ValueInitInfo& dstInitInfo) {
  unsigned dstRootDistance = dstInitInfo.getRootDistance();
  unsigned newDstRootDistance = srcInitInfo.getUserRootDistance();

  /* Propagate info only if the path to a root is shorter than the current */
  if (newDstRootDistance < dstRootDistance) {
    dstInitInfo.setRootDistance(newDstRootDistance);

    LLVM_DEBUG(
      Logger& logger = log();
      logger.setContextTag(__FUNCTION__);
      logger << "updated root distance: " << newDstRootDistance << "\n";
      logger << ", srcInfo = ";
      logger.logln(*srcInfo, Logger::Cyan);
      logger << ", ";);

    // TODO remove and correct taffo to lower relative errors in benchmarks:
    // Whole valueInfo copy should not be performed.
    // InitializerPass should be only responsible of setting conversionEnabled,
    // but right now, removing this causes big relative errors in some benchmarks
    if (!isa<StructInfo>(srcInfo) && !isa<StructInfo>(dstInfo))
      dstInfo->copyFrom(*srcInfo);

    if (srcInfo->getKind() == dstInfo->getKind())
      dstInfo->mergeConversionEnabled(*srcInfo);

    LLVM_DEBUG(
      Logger& logger = log();
      logger << "dstInfo";
      logger << " = ";
      logger.logln(*dstInfo, Logger::Cyan);
      logger.restorePrevContextTag(););
  }
  else
    LLVM_DEBUG(log().logln("already has info from a value closer or equally close to a root: continuing"));
}

void InitializerPass::cloneFunctionForCall(CallBase* call) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << Logger::Bold << "[" << __FUNCTION__ << "] " << Logger::Reset;
    logger.logValueln(call));
  indenter.increaseIndent();

  Function* oldF = call->getCalledFunction();
  if (!oldF) {
    LLVM_DEBUG(logger.logln("call to indirect function: skipping"));
    handledCalls[call] = nullptr;
    return;
  }
  if (isSpecialFunction(oldF)) {
    LLVM_DEBUG(logger.logln("call to special function: skipping"));
    handledCalls[call] = nullptr;
    return;
  }
  if (manualFunctionCloning) {
    if (!annotatedFunctions.contains(oldF)) {
      LLVM_DEBUG(logger.logln("call to disabled function: skipping"));
      handledCalls[call] = nullptr;
      return;
    }
  }

  assert(!handledCalls.contains(call) && "Call already handled");
  const auto newName = oldF->getName() + "_clone" + std::to_string(taffoInfo.getNumCloneFunctions(*oldF));
  Function* newF = Function::Create(oldF->getFunctionType(), oldF->getLinkage(), newName, oldF->getParent());
  call->setCalledFunction(newF);

  // Setting oldF as weak to avoid globalDCE and preserve the mapping between old function and cloned function
  taffoInfo.setOriginalFunctionLinkage(*oldF, oldF->getLinkage());
  oldF->setLinkage(GlobalValue::WeakAnyLinkage);

  annotatedFunctions.insert(newF);
  handledCalls[call] = newF;
  taffoInfo.addCloneFunction(*oldF, *newF);

  // Create Val2Val mapping and clone function
  ValueToValueMapTy valueMap;
  for (auto&& [oldArg, newArg] : zip(oldF->args(), newF->args())) {
    newArg.setName(oldArg.getName());
    valueMap.insert({&oldArg, &newArg});
  }
  SmallVector<ReturnInst*, 10> returns;
  CloneFunctionInto(newF, oldF, valueMap, CloneFunctionChangeType::GlobalChanges, returns);

  // Lambda to copy TransparentType and ValueInitInfo from src value to dst value
  auto copyInfo = [this](const Value* src, Value* dst) {
    if (taffoInfo.hasTransparentType(*src))
      taffoInfo.setTransparentType(*dst, taffoInfo.getTransparentType(*src)->clone());
    if (taffoInfo.hasValueInfo(*src))
      taffoInfo.setValueInfo(*dst, taffoInfo.getValueInfo(*src)->clone());
    if (taffoInitInfo.hasValueInitInfo(src)) {
      ValueInitInfo& oldValueInitInfo = taffoInitInfo.getValueInitInfo(src);
      ValueInitInfo& newValueInitInfo = taffoInitInfo.createValueInitInfo(dst);
      newValueInitInfo.setRootDistance(oldValueInitInfo.getRootDistance());
      infoPropagationQueue.push_back(dst);
    }
  };

  for (auto&& [oldValue, newValue] : valueMap)
    copyInfo(oldValue, newValue);
  copyInfo(call, newF);

  if (!openCLKernelMode && !cudaKernelMode)
    newF->setLinkage(GlobalVariable::LinkageTypes::InternalLinkage);

  FunctionCloned++;

  LLVM_DEBUG(
    logger.log("old function: ").logValueln(oldF);
    logger.log("new function: ").logValueln(newF);
    logger.logln("[Propagating info from call arguments to clone function arguments]", Logger::Bold);
    indenter.increaseIndent();
    if (newF->arg_empty())
      logger.logln("function has no arguments: continuing"););

  // Propagate ValueInitInfo from call args to function args
  for (Argument& arg : newF->args()) {
    auto indenter = logger.getIndenter();
    LLVM_DEBUG(
      logger << Logger::Reset << "[Arg " << arg.getArgNo() << "] " << Logger::Reset << arg << "\n";
      indenter.increaseIndent(););
    if (arg.user_empty()) {
      LLVM_DEBUG(log().logln("arg has no users: skipping"));
      continue;
    }
    Value* callArg = call->getArgOperand(arg.getArgNo());
    if (!taffoInitInfo.hasValueInitInfo(callArg)) {
      LLVM_DEBUG(log().logln("arg has no valueInitInfo in the call: skipping"));
      continue;
    }
    Value* argAlloca = nullptr;
    for (Value* argUser : arg.users())
      if (auto* store = dyn_cast<StoreInst>(argUser))
        if (store->getValueOperand() == &arg && isa<AllocaInst>(store->getPointerOperand()))
          argAlloca = store->getPointerOperand();

    std::shared_ptr<ValueInfo> callArgInfo = taffoInfo.getValueInfo(*callArg);
    ValueInitInfo& callArgInitInfo = taffoInitInfo.getValueInitInfo(callArg);

    std::shared_ptr<ValueInfo> argInfo = callArgInfo->clone();
    taffoInfo.setValueInfo(arg, argInfo);
    ValueInitInfo& argInitInfo = taffoInitInfo.getOrCreateValueInitInfo(&arg);
    argInitInfo.setRootDistance(callArgInitInfo.getUserRootDistance());
    infoPropagationQueue.push_back(&arg);

    if (argAlloca) {
      taffoInfo.setValueInfo(*argAlloca, argInfo->clone());
      ValueInitInfo& argAllocaInitInfo = taffoInitInfo.getOrCreateValueInitInfo(argAlloca);
      argAllocaInitInfo.setRootDistance(argInitInfo.getUserRootDistance());
      infoPropagationQueue.push_back(argAlloca);
    }

    // Propagate BufferID
    argInfo->bufferId = callArgInfo->bufferId;

    LLVM_DEBUG(
      logger.log("argInfo: ");
      logger.log(*argInfo, Logger::Cyan);
      logger.logln(" copied from call", Logger::Green);
      if (argAlloca) {
        logger.log("argInfo also copied to argument alloca: ", Logger::Green);
        logger.logln(*argAlloca);
      });
  }
}

void InitializerPass::logInfoPropagationQueue() {
  Logger& logger = log();
  if (infoPropagationQueue.size() < 1000) {
    for (Value* value : infoPropagationQueue) {
      logger.log("[Value] ", Logger::Bold).logValueln(value);
      auto indenter = logger.getIndenter();
      indenter.increaseIndent();
      logger << "valueInfo: " << taffoInfo.getValueInfo(*value) << "\n";
      logger << "valueInitInfo: " << taffoInitInfo.getValueInitInfo(value) << "\n";
    }
  }
  else
    logger.logln("Not logging the queue because it exceeds 1000 items", Logger::Yellow);
}
