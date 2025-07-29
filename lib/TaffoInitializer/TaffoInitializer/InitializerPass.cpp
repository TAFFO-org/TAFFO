#include "CudaKernelPatcher.hpp"
#include "Debug/Logger.hpp"
#include "IndirectCallPatcher.hpp"
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
  generateFunctionClones();
  LLVM_DEBUG(log().logln("[Propagating info after function cloning]", Logger::Blue));
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
    if (auto* inst = dyn_cast<Instruction>(value))
      if (auto scalarInfo = dyn_cast<ScalarInfo>(taffoInfo.getValueInfo(*value).get()))
        if (taffoInfo.isConversionDisabled(*inst)) {
          scalarInfo->conversionEnabled = false;
          LLVM_DEBUG(
            Logger& logger = log();
            logger.log("Disabled conversion of shared variable ", Logger::Yellow);
            logger.logValueln(inst););
        }
  }
}

/**
 * @brief Propagates info from roots.
 *
 * Starting from the set of root values, this function propagates taffoInitInfo
 * through the IR graph by iterating over each value's users (forward propagation)
 * and then backtracking through instruction operands (backward propagation)
 * until no new values are discovered
 */
void InitializerPass::propagateInfo() {
  Logger& logger = log();
  LLVM_DEBUG(
    logger.logln("[Initial propagation queue]", Logger::Bold);
    auto indenter = logger.getIndenter();
    indenter.increaseIndent();
    logInfoPropagationQueue(););

  // Propagate info from arguments alloca to formal arguments to call arguments
  // for (Value* value : infoPropagationQueue)
  //  if (auto* argAlloca = dyn_cast<AllocaInst>(value))
  //    for (User* argAllocaUser : argAlloca->users())
  //      if (auto* argStore = dyn_cast<StoreInst>(argAllocaUser))
  //        if (argStore->getPointerOperand() == argAlloca)
  //          if (auto* arg = dyn_cast<Argument>(argStore->getValueOperand())) {
  //            propagateInfo(argAlloca, arg);
  //            for (User* user : arg->getParent()->users())
  //              if (auto* call = dyn_cast<CallBase>(user))
  //                propagateInfo(arg, call->getArgOperand(arg->getArgNo()));
  //          }

  // Set to track processed values to avoid reprocessing
  SmallPtrSet<Value*, 8> visited;

  // Continue propagation until the queue size stabilizes
  unsigned iteration = 0;
  size_t prevQueueSize = 0;
  while (prevQueueSize < infoPropagationQueue.size()) {
    LLVM_DEBUG(log() << Logger::Blue << "[Propagation iteration " << iteration << "]\n"
                     << Logger::Reset);
    prevQueueSize = infoPropagationQueue.size();
    iteration++;

    // Forward propagation: process each value in the queue
    for (Value* value : infoPropagationQueue) {
      visited.insert(value);
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
          Logger& logger = log();
          logger.log("[User] ", Logger::Bold).logValueln(user);
          indenter.increaseIndent(););

        // Update valueInitInfo of the user based on the parent's valueInitInfo
        propagateInfo(value, user);

        // Determine the new backtracking depth for backward propagation
        unsigned newUserBtDepth = valueInitInfo.getUserBacktrackingDepth();
        if (auto* storeUser = dyn_cast<StoreInst>(user))
          if (newUserBtDepth <= 1) {
            Value* storedValue = storeUser->getValueOperand();
            if (getFullyUnwrappedType(storedValue)->isFloatingPointTy()) {
              LLVM_DEBUG(
                Logger& logger = log();
                logger.log("will backtrack to stored value: ", Logger::Cyan);
                logger.logValueln(storedValue););
              newUserBtDepth = 1;
            }
          }
        if (newUserBtDepth > 0) {
          ValueInitInfo& userInitInfo = taffoInitInfo.getValueInitInfo(user);
          userInitInfo.setBacktrackingDepth(std::max(newUserBtDepth, userInitInfo.getBacktrackingDepth()));
        }

        // Extract the user from the queue if already in
        auto userIter = std::ranges::find(infoPropagationQueue, user);
        if (userIter != infoPropagationQueue.end())
          infoPropagationQueue.erase(userIter);
        // Insert it at the end of the queue
        infoPropagationQueue.push_back(user);
      }
    }

    // Backward propagation: process operands of each instruction in reverse order
    for (auto iter = --infoPropagationQueue.end(); iter != infoPropagationQueue.begin(); iter--) {
      Value* value = *iter;
      ValueInitInfo& valueInitInfo = taffoInitInfo.getValueInitInfo(value);

      unsigned backtrackingDepth = valueInitInfo.getBacktrackingDepth();
      if (backtrackingDepth == 0)
        continue;
      auto* inst = dyn_cast<Instruction>(value);
      if (!inst)
        continue;

      auto indenter = logger.getIndenter();
      LLVM_DEBUG(
        Logger& logger = log();
        logger.log("[Backtracking] ", Logger::Bold).logValueln(value);
        indenter.increaseIndent();
        logger << "depth left = " << backtrackingDepth << "\n";);

      // Process each operand of the instruction
      for (Value* operand : inst->operands()) {
        auto indenter = logger.getIndenter();
        LLVM_DEBUG(
          Logger& logger = log();
          logger.log("[Operand] ", Logger::Bold).logValueln(operand);
          indenter.increaseIndent(););
        // Skip operands that are not a User or an Argument
        if (!isa<User>(operand) && !isa<Argument>(operand)) {
          LLVM_DEBUG(log().logln("not a user or an argument: ignoring"));
          continue;
        }
        // Skip functions and block addresses
        if (isa<Function>(operand) || isa<BlockAddress>(operand)) {
          LLVM_DEBUG(log().logln("is a function or a block address: ignoring"));
          continue;
        }
        // Skip constants
        if (isa<Constant>(operand)) {
          LLVM_DEBUG(log().logln("is a constant: ignoring"));
          continue;
        }
        if (!getFullyUnwrappedType(operand)->isFloatingPointTy()) {
          LLVM_DEBUG(log().logln("not a float: ignoring"));
          continue;
        }

        bool alreadyIn = false;
        auto userIter = std::ranges::find(infoPropagationQueue, operand);
        if (userIter != infoPropagationQueue.end()) {
          auto valuePosition = std::distance(infoPropagationQueue.begin(), iter);
          auto userPosition = std::distance(infoPropagationQueue.begin(), userIter);
          if (userPosition < valuePosition)
            alreadyIn = true;
          else
            infoPropagationQueue.erase(userIter);
        }

        if (!alreadyIn) {
          propagateInfo(value, operand);
          infoPropagationQueue.push_front(operand);
        }
        else
          LLVM_DEBUG(log().logln("already in queue: continuing"));

        ValueInitInfo& userInitInfo = taffoInitInfo.getOrCreateValueInitInfo(operand);
        userInitInfo.decreaseBacktrackingDepth();
      }
    }
  }
  LLVM_DEBUG(log().logln("[Propagation completed]", Logger::Blue));
}

void InitializerPass::propagateInfo(Value* src, Value* dst) {
  if (!taffoInfo.hasValueInfo(*dst))
    taffoInfo.createValueInfo(*dst);

  ValueInitInfo& srcInitInfo = taffoInitInfo.getValueInitInfo(src);
  ValueInitInfo& dstInitInfo = taffoInitInfo.getOrCreateValueInitInfo(dst);
  ValueInfo* srcInfo = &*taffoInfo.getValueInfo(*src);
  ValueInfo* dstInfo = &*taffoInfo.getValueInfo(*dst);

  unsigned dstRootDistance = dstInitInfo.getRootDistance();
  unsigned newDstRootDistance = srcInitInfo.getUserRootDistance();

  //// If dst is a gep and src is one of its index operands, skip propagation completely
  // if (auto* dstGep = dyn_cast<GetElementPtrInst>(dst);
  //     dstGep && is_contained(dstGep->operands(), src) && src != dstGep->getPointerOperand()) {
  //   LLVM_DEBUG(log().logln("dst is a gep, but src is just an index operand: continuing"));
  //     }
  //  If dst is a gep and src is not its pointer operand, skip propagation completely
  if (isa<GetElementPtrInst>(dst) && src != cast<GetElementPtrInst>(dst)->getPointerOperand()) {
    LLVM_DEBUG(log().logln("dst is a gep, but src is not its pointer operand: continuing"));
  }
  /* Propagate info only if the path to a root is shorter than the current */
  else if (newDstRootDistance < dstRootDistance) {
    dstInitInfo.setRootDistance(newDstRootDistance);
    std::shared_ptr<TransparentType> srcType = taffoInfo.getOrCreateTransparentType(*src);
    std::shared_ptr<TransparentType> dstType = taffoInfo.getOrCreateTransparentType(*dst);

    LLVM_DEBUG(
      Logger& logger = log();
      logger.setContextTag(__FUNCTION__);
      logger << "updated root distance: " << newDstRootDistance << "\n";
      logger << "srcType = ";
      logger.log(*srcType, Logger::Cyan);
      logger << ", srcInfo = ";
      logger.logln(*srcInfo, Logger::Cyan);
      logger << "dstType = ";
      logger.log(*dstType, Logger::Cyan);
      logger << ", ";);

    if (!srcType->isStructType() && !dstType->isStructType())
      dstInfo->copyFrom(*srcInfo);
    // TODO Manage structs (conversionEnabled of fields)
    if (dstInfo->isConversionEnabled() || srcInfo->isConversionEnabled())
      if (auto* dstScalarInfo = dyn_cast<ScalarInfo>(dstInfo))
        dstScalarInfo->conversionEnabled = true;

    LLVM_DEBUG(
      Logger& logger = log();
      logger << "dstInfo";
      logger << " = ";
      logger.logln(*dstInfo, Logger::Cyan);
      logger.restorePrevContextTag(););
  }
  else
    LLVM_DEBUG(log().logln("already has info from a value closer or equally close to a root: continuing"));

  // Copy BufferId across loads, geps and bitcasts
  if (isa<LoadInst>(dst) || isa<GetElementPtrInst>(dst) || isa<BitCastInst>(dst))
    dstInfo->bufferId = srcInfo->bufferId;
}

void InitializerPass::generateFunctionClones() {
  LLVM_DEBUG(log().logln("[Function cloning]", Logger::Blue));
  for (Value* value : infoPropagationQueue) {
    auto* call = dyn_cast<CallBase>(value);
    if (!call)
      continue;

    Function* oldF = call->getCalledFunction();
    if (!oldF) {
      LLVM_DEBUG(log().log("Skipping indirect function invoked by: ", Logger::Yellow).logValueln(value));
      continue;
    }
    if (isSpecialFunction(oldF)) {
      LLVM_DEBUG(log().log("Skipping special function invoked by: ", Logger::Yellow).logValueln(value));
      continue;
    }
    if (manualFunctionCloning) {
      if (!annotatedFunctions.contains(oldF)) {
        LLVM_DEBUG(log().log("Skipping disabled function invoked by: ", Logger::Yellow).logValueln(value));
        continue;
      }
    }

    Function* newF = cloneFunction(call);
    call->setCalledFunction(newF);
    annotatedFunctions.insert(newF);

    // Setting oldF as weak  to avoid globalDCE and preserve the mapping between old function and cloned function
    taffoInfo.setOriginalFunctionLinkage(*oldF, oldF->getLinkage());
    oldF->setLinkage(llvm::GlobalValue::WeakAnyLinkage);

    taffoInfo.setTaffoFunction(*oldF, *newF);
  }
  LLVM_DEBUG(log().logln("[Function cloning completed]", Logger::Blue));
}

Function* InitializerPass::cloneFunction(const CallBase* call) {
  Function* oldF = call->getCalledFunction();
  Function* newF = Function::Create(oldF->getFunctionType(), oldF->getLinkage(), oldF->getName(), oldF->getParent());

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
      taffoInfo.setTransparentType(*dst, taffoInfo.getTransparentType(*src));
    if (taffoInfo.hasValueInfo(*src))
      taffoInfo.setValueInfo(*dst, taffoInfo.getValueInfo(*src)->clone());
    if (taffoInitInfo.hasValueInitInfo(src)) {
      ValueInitInfo& oldValueInitInfo = taffoInitInfo.getValueInitInfo(src);
      ValueInitInfo& newValueInitInfo = taffoInitInfo.createValueInitInfo(dst);
      newValueInitInfo.setRootDistance(oldValueInitInfo.getRootDistance());
      newValueInitInfo.setBacktrackingDepth(oldValueInitInfo.getBacktrackingDepth());
      infoPropagationQueue.push_back(dst);
    }
  };

  for (auto&& [oldValue, newValue] : valueMap)
    copyInfo(oldValue, newValue);
  copyInfo(oldF, newF);

  if (!openCLKernelMode && !cudaKernelMode)
    newF->setLinkage(GlobalVariable::LinkageTypes::InternalLinkage);

  FunctionCloned++;

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger.log("[Cloning of] ", Logger::Bold).logValueln(oldF);
    indenter.increaseIndent();
    logger.log("new function: ").logValueln(newF);
    logger.log("for call: ").logValueln(call);
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
      Logger& logger = log();
      logger.log("argInfo: ");
      logger.log(*argInfo, Logger::Cyan);
      logger.logln(" copied from call", Logger::Green);
      if (argAlloca) {
        logger.log("argInfo also copied to argument alloca: ", Logger::Green);
        logger.logln(*argAlloca);
      });
  }
  return newF;
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
