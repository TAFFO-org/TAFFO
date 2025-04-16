#include "InitializerPass.hpp"

#include "Debug/Logger.hpp"
#include "Types/TypeUtils.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "IndirectCallPatcher.hpp"
#include "OpenCLKernelPatcher.hpp"
#include "CudaKernelPatcher.hpp"
#include "llvm/IR/Argument.h"
#include "llvm/IR/GlobalValue.h"

#include <llvm/Transforms/Utils/Cloning.h>

#define DEBUG_TYPE "taffo-init"

using namespace taffo;
using namespace llvm;

cl::opt<bool> manualFunctionCloning("manualclone",
  cl::desc("Enables function cloning only for annotated functions"),
  cl::init(false));

cl::opt<bool> openCLKernelMode("oclkern",
  cl::desc("Allows cloning of OpenCL kernel functions"),
  cl::init(false));

cl::opt<bool> cudaKernelMode("cudakern",
  cl::desc("Allows cloning of Cuda kernel functions"),
  cl::init(false));

PreservedAnalyses InitializerPass::run(Module &m, ModuleAnalysisManager &) {
  LLVM_DEBUG(log().logln("[InitializerPass]", raw_ostream::Colors::MAGENTA));
  TaffoInfo::getInstance().initializeFromFile("taffo_typededucer.json", m);

  if (openCLKernelMode) {
    LLVM_DEBUG(dbgs() << "OpenCLKernelMode == true!\n");
    createOpenCLKernelTrampolines(m);
  }
  else if (cudaKernelMode) {
    LLVM_DEBUG(dbgs() << "CudaKernelMode == true!\n");
    createCudaKernelTrampolines(m);
  }

  manageIndirectCalls(m);
  readAndRemoveGlobalAnnotations(m);
  readAndRemoveLocalAnnotations(m);
  AnnotationCount = infoPropagationQueue.size();
  removeNotFloats();

  if (Function *startingPoint = findStartingPointFunctionGlobal(m)) {
    LLVM_DEBUG(
      Logger &logger = log();
      logger.log("Found starting point using global __taffo_vra_starting_function: ");
      logger.logln(startingPoint->getName());
    );
    TaffoInfo::getInstance().addStartingPoint(*startingPoint);
  }

  LLVM_DEBUG(log().logln("[Propagating info from roots]", raw_ostream::Colors::BLUE));
  propagateInfo();
  generateFunctionClones();
  LLVM_DEBUG(log().logln("[Propagating info after function cloning]", raw_ostream::Colors::BLUE));
  propagateInfo();
  LLVM_DEBUG(
    log().logln("[Results]", raw_ostream::Colors::GREEN);
    logInfoPropagationQueue();
  );
  saveValueWeights();

  TaffoInfo::getInstance().dumpToFile("taffo_info_init.json", m);
  LLVM_DEBUG(log().logln("[End of InitializerPass]", raw_ostream::Colors::MAGENTA));
  return PreservedAnalyses::all();
}

void InitializerPass::saveValueWeights() {
  TaffoInfo &taffoInfo = TaffoInfo::getInstance();
  for (Value *value : infoPropagationQueue) {
    ValueInitInfo &valueInitInfo = taffoInitInfo.getValueInitInfo(value);
    if (isa<Instruction>(value) || isa<GlobalObject>(value))
      taffoInfo.setValueWeight(*value, valueInitInfo.getRootDistance());
    if (auto *inst = dyn_cast<Instruction>(value))
      if (auto scalarInfo = dyn_cast<ScalarInfo>(valueInitInfo.getValueInfo()))
        if (taffoInfo.isConversionDisabled(*inst)) {
          scalarInfo->conversionEnabled = false;
          LLVM_DEBUG(
            Logger &logger = log();
            logger.log("Disabled conversion of shared variable ", raw_ostream::Colors::YELLOW);
            logger.logValueln(inst);
          );
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
  LLVM_DEBUG(
    log().logln("[Initial propagation queue]", raw_ostream::Colors::BLACK);
    log().increaseIndent();
    logInfoPropagationQueue();
    log().decreaseIndent();
  );

  // Set to track processed values to avoid reprocessing
  SmallPtrSet<Value*, 8> visited;

  // Continue propagation until the queue size stabilizes
  unsigned int iteration = 0;
  size_t prevQueueSize = 0;
  while (prevQueueSize < infoPropagationQueue.size()) {
    LLVM_DEBUG(log() << raw_ostream::Colors::BLUE << "[Propagation iteration " << iteration << "]\n" << raw_ostream::Colors::RESET);
    prevQueueSize = infoPropagationQueue.size();
    iteration++;

    // Forward propagation: process each value in the queue
    for (Value *value : infoPropagationQueue) {
      visited.insert(value);
      ValueInitInfo &valueInitInfo = taffoInitInfo.getValueInitInfo(value);

      LLVM_DEBUG(
        Logger &logger = log();
        logger.log("[Value] ", raw_ostream::Colors::BLACK).logValueln(value);
        logger.increaseIndent();
        logger << "root distance: " << valueInitInfo.getRootDistance() << "\n";
        if (value->user_empty())
          logger.logln("value has no users: continuing");
      );

      // Process each user of the current value
      for (auto *user : value->users()) {
        // Skip PHI nodes already processed
        if (isa<PHINode>(user) && visited.contains(user))
          continue;

        LLVM_DEBUG(
          Logger &logger = log();
          logger.log("[User] ", raw_ostream::Colors::BLACK).logValueln(user);
          logger.increaseIndent();
        );

        // Update valueInitInfo of the user based on the parent's valueInitInfo
        propagateInfo(value, user);

        // Determine the new backtracking depth for backward propagation
        unsigned int newUserBtDepth = valueInitInfo.getUserBacktrackingDepth();
        if (auto *storeUser = dyn_cast<StoreInst>(user))
          if (newUserBtDepth <= 1) {
            Value *storedValue = storeUser->getValueOperand();
            if (getUnwrappedType(storedValue)->isFloatingPointTy()) {
              LLVM_DEBUG(
                Logger &logger = log();
                logger.log("will backtrack to stored value: ", raw_ostream::Colors::CYAN);
                logger.logValueln(storedValue);
              );
              newUserBtDepth = 1;
            }
          }
        if (newUserBtDepth > 0) {
          ValueInitInfo &userInitInfo = taffoInitInfo.getValueInitInfo(user);
          userInitInfo.setBacktrackingDepth(std::max(newUserBtDepth, userInitInfo.getBacktrackingDepth()));
        }

        // Extract the user from the queue if already in
        auto userIter = std::ranges::find(infoPropagationQueue, user);
        if (userIter != infoPropagationQueue.end())
          infoPropagationQueue.erase(userIter);
        // Insert it at the end of the queue
        infoPropagationQueue.push_back(user);

        LLVM_DEBUG(log().decreaseIndent());
      }
      LLVM_DEBUG(log().decreaseIndent());
    }

    // Backward propagation: process operands of each instruction in reverse order
    for (auto iter = --infoPropagationQueue.end(); iter != infoPropagationQueue.begin(); iter--) {
      Value *value = *iter;
      ValueInitInfo &valueInitInfo = taffoInitInfo.getValueInitInfo(value);

      unsigned int backtrackingDepth = valueInitInfo.getBacktrackingDepth();
      if (backtrackingDepth == 0)
        continue;
      auto *inst = dyn_cast<Instruction>(value);
      if (!inst)
        continue;

      LLVM_DEBUG(
        Logger &logger = log();
        logger.log("[Backtracking] ", raw_ostream::Colors::BLACK).logValueln(value);
        logger.increaseIndent();
        logger << "depth left = " << backtrackingDepth << "\n";
      );

      // Process each operand of the instruction
      for (Value *operand : inst->operands()) {
        LLVM_DEBUG(
          Logger &logger = log();
          logger.log("[Operand] ", raw_ostream::Colors::BLACK).logValueln(operand);
          logger.increaseIndent();
        );
        // Skip operands that are not a User or an Argument
        if (!isa<User>(operand) && !isa<Argument>(operand)) {
          LLVM_DEBUG(log().logln("not a user or an argument: ignoring").decreaseIndent());
          continue;
        }
        // Skip functions and block addresses
        if (isa<Function>(operand) || isa<BlockAddress>(operand)) {
          LLVM_DEBUG(log().logln("is a function or a block address: ignoring").decreaseIndent());
          continue;
        }
        // Skip constants
        if (isa<Constant>(operand)) {
          LLVM_DEBUG(log().logln("is a constant: ignoring").decreaseIndent());
          continue;
        }
        if (!getUnwrappedType(operand)->isFloatTy()) {
          LLVM_DEBUG(log().logln("not a float: ignoring").decreaseIndent());
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

        ValueInitInfo &userInitInfo = taffoInitInfo.getOrCreateValueInitInfo(operand);
        userInitInfo.decreaseBacktrackingDepth();

        LLVM_DEBUG(log().decreaseIndent());
      }
      LLVM_DEBUG(log().decreaseIndent());
    }
  }
  LLVM_DEBUG(log().logln("[Propagation completed]", raw_ostream::Colors::BLUE));
}

void InitializerPass::propagateInfo(Value *src, Value *dst) {
  TaffoInfo &taffoInfo = TaffoInfo::getInstance();
  if (!taffoInfo.hasValueInfo(*dst))
    taffoInfo.createValueInfo(*dst);

  ValueInitInfo &srcInitInfo = taffoInitInfo.getValueInitInfo(src);
  ValueInitInfo &dstInitInfo = taffoInitInfo.getOrCreateValueInitInfo(dst);
  ValueInfo *srcInfo = srcInitInfo.getValueInfo();
  ValueInfo *dstInfo = dstInitInfo.getValueInfo();

  unsigned int dstRootDistance = dstInitInfo.getRootDistance();
  unsigned int newDstRootDistance = srcInitInfo.getUserRootDistance();
  /* Propagate info only if the path to a root is shorter than the current */
  if (newDstRootDistance < dstRootDistance) {
    dstInitInfo.setRootDistance(newDstRootDistance);
    std::shared_ptr<TransparentType> srcType = taffoInfo.getOrCreateTransparentType(*src);
    std::shared_ptr<TransparentType> dstType = taffoInfo.getOrCreateTransparentType(*dst);

    LLVM_DEBUG(
      Logger &logger = log();
      logger.setContextTag(__FUNCTION__);
      logger << "updated root distance: " << newDstRootDistance << "\n";
      logger << "srcType = ";
      logger.log(*srcType, raw_ostream::Colors::CYAN);
      logger << ", srcInfo = ";
      logger.logln(*srcInfo, raw_ostream::Colors::CYAN);
      logger << "dstType = ";
      logger.log(*dstType, raw_ostream::Colors::CYAN);
      logger << ", ";
    );

    // TODO Manage structs
    // TODO Manage geps
    bool copied = false;
    bool wasEnabled = dstInfo->isConversionEnabled();
    if (!srcType->isStructType() && !dstType->isStructType()) {
      // If dst is a call create empty valueinfo without range
      if(isa<CallBase>(dst)) {
        if (auto *dstScalarInfo = dyn_cast<ScalarInfo>(dstInfo))
          dstScalarInfo->conversionEnabled = srcInfo->isConversionEnabled();
      } else {
        dstInfo->copyFrom(*srcInfo);
        copied = true;
      }

      if (wasEnabled)
        if (auto *dstScalarInfo = dyn_cast<ScalarInfo>(dstInfo))
          dstScalarInfo->conversionEnabled = true;
    }

    else if (srcInfo->isConversionEnabled()) {
      if (auto *dstScalarInfo = dyn_cast<ScalarInfo>(dstInfo))
        dstScalarInfo->conversionEnabled = true;
    }

    LLVM_DEBUG(
      Logger &logger = log();
      logger << "dstInfo";
      logger << " = ";
      logger.log(*dstInfo, raw_ostream::Colors::CYAN);
      if (copied)
        logger.log(" copied", raw_ostream::Colors::GREEN);
      logger << "\n";
      logger.restorePrevContextTag();
    );
  }
  else
    LLVM_DEBUG(log().logln("already has info from a value closer or equally close to a root: continuing"));

  // Copy BufferId across loads, geps and bitcasts
  if (isa<LoadInst>(dst) || isa<GetElementPtrInst>(dst) || isa<BitCastInst>(dst)) {
    dstInfo->bufferId = srcInfo->bufferId;
  }
}

void InitializerPass::generateFunctionClones() {
  LLVM_DEBUG(log().logln("[Function cloning]", raw_ostream::Colors::BLUE));
  for (Value *value : infoPropagationQueue) {
    auto *call = dyn_cast<CallBase>(value);
    if (!call)
      continue;

    Function *oldF = call->getCalledFunction();
    if (!oldF) {
      LLVM_DEBUG(log().log("Skipping indirect function invoked by: ", raw_ostream::Colors::YELLOW).logValueln(value));
      continue;
    }
    if (isSpecialFunction(oldF)) {
      LLVM_DEBUG(log().log("Skipping special function invoked by: ", raw_ostream::Colors::YELLOW).logValueln(value));
      continue;
    }
    if (manualFunctionCloning) {
      if (!annotatedFunctions.contains(oldF)) {
        LLVM_DEBUG(log().log("Skipping disabled function invoked by: ", raw_ostream::Colors::YELLOW).logValueln(value));
        continue;
      }
    }

    Function *newF = cloneFunction(call);
    call->setCalledFunction(newF);
    annotatedFunctions.insert(newF);

    TaffoInfo &taffoInfo = TaffoInfo::getInstance();

    //Setting oldF as weak  to avoid globalDCE and preserve the mapping between old function and cloned function
    taffoInfo.setOriginalFunctionLinkage(*oldF, oldF->getLinkage());
    oldF->setLinkage(llvm::GlobalValue::WeakAnyLinkage);

    taffoInfo.setTaffoFunction(*oldF, *newF);
  }
  LLVM_DEBUG(log().logln("[Function cloning completed]", raw_ostream::Colors::BLUE));
}

Function *InitializerPass::cloneFunction(const CallBase *call) {
  Function *oldF = call->getCalledFunction();
  Function *newF = Function::Create(
    oldF->getFunctionType(), oldF->getLinkage(), oldF->getName(), oldF->getParent());

  // Create Val2Val mapping and clone function
  ValueToValueMapTy valueMap;
  for (auto &&[oldArg, newArg] : zip(oldF->args(), newF->args())) {
    newArg.setName(oldArg.getName());
    valueMap.insert({&oldArg, &newArg});
  }
  SmallVector<ReturnInst*, 10> returns;
  CloneFunctionInto(newF, oldF, valueMap, CloneFunctionChangeType::GlobalChanges, returns);

  TaffoInfo &taffoInfo = TaffoInfo::getInstance();
  // Lambda to copy TransparentType and ValueInitInfo from src value to dst value
  auto copyInfo = [this, &taffoInfo](Value *src, Value *dst) {
    if (taffoInfo.hasTransparentType(*src))
      taffoInfo.setTransparentType(*dst, taffoInfo.getOrCreateTransparentType(*src));
    if (std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*src))
      taffoInfo.setValueInfo(*dst, valueInfo->clone());
    if (taffoInitInfo.hasValueInitInfo(src)) {
      ValueInitInfo &oldValueInitInfo = taffoInitInfo.getValueInitInfo(src);
      ValueInitInfo &newValueInitInfo = taffoInitInfo.createValueInitInfo(dst);
      newValueInitInfo.setRootDistance(oldValueInitInfo.getRootDistance());
      newValueInitInfo.setBacktrackingDepth(oldValueInitInfo.getBacktrackingDepth());
      infoPropagationQueue.push_back(dst);
    }
  };

  for (auto &&[oldValue, newValue] : valueMap)
    copyInfo(const_cast<Value*>(oldValue), newValue);
  copyInfo(oldF, newF);

  if (!openCLKernelMode && !cudaKernelMode)
    newF->setLinkage(GlobalVariable::LinkageTypes::InternalLinkage);

  FunctionCloned++;

  LLVM_DEBUG(
    Logger &logger = log();
    logger.log("[Cloning of] ", raw_ostream::Colors::BLACK).logValueln(oldF);
    logger.increaseIndent();
    logger.log("new function: ").logValueln(newF);
    logger.log("for call: ").logValueln(call);
    logger.logln("[Propagating info from call arguments to clone function arguments]", raw_ostream::Colors::BLACK);
    logger.increaseIndent();
    if (newF->arg_empty())
      logger.logln("function has no arguments: continuing");
  );

  // Propagate ValueInitInfo from call args to function args
  for (Argument &arg : newF->args()) {
    LLVM_DEBUG(
      Logger &logger = log();
      logger << raw_ostream::Colors::BLACK << "[Arg " << arg.getArgNo() << "] " << raw_ostream::Colors::RESET << arg << "\n";
      logger.increaseIndent();
    );
    if (arg.user_empty()) {
      LLVM_DEBUG(log().logln("arg has no users: skipping").decreaseIndent());
      continue;
    }
    Value *callArg = call->getArgOperand(arg.getArgNo());
    if (!taffoInitInfo.hasValueInitInfo(callArg)) {
      LLVM_DEBUG(log().logln("arg has no valueInitInfo in the call: skipping").decreaseIndent());
      continue;
    }
    Value *argAlloca = nullptr;
    for (Value *argUser : arg.users())
      if (auto *store = dyn_cast<StoreInst>(argUser))
        if (store->getValueOperand() == &arg && isa<AllocaInst>(store->getPointerOperand()))
          argAlloca = store->getPointerOperand();

    ValueInitInfo &callArgInitInfo = taffoInitInfo.getValueInitInfo(callArg);
    taffoInfo.setValueInfo(arg, callArgInitInfo.getValueInfo()->clone());
    ValueInitInfo &argInitInfo = taffoInitInfo.createValueInitInfo(&arg);
    argInitInfo.setRootDistance(callArgInitInfo.getUserRootDistance());
    infoPropagationQueue.push_back(&arg);

    if (argAlloca) {
      taffoInfo.setValueInfo(*argAlloca, callArgInitInfo.getValueInfo()->clone());
      ValueInitInfo &argAllocaInitInfo = taffoInitInfo.createValueInitInfo(argAlloca);
      argAllocaInitInfo.setRootDistance(argInitInfo.getUserRootDistance());
      infoPropagationQueue.push_back(argAlloca);
    }

    // Propagate BufferID
    argInitInfo.getValueInfo()->bufferId = callArgInitInfo.getValueInfo()->bufferId;

    LLVM_DEBUG(
      Logger &logger = log();
      logger.log("argInfo: ");
      logger.log(*argInitInfo.getValueInfo(), raw_ostream::Colors::CYAN);
      logger.logln(" copied from call", raw_ostream::Colors::GREEN);
      if (argAlloca) {
        logger.log("argInfo also copied to argument alloca: ", raw_ostream::Colors::GREEN);
        logger.logln(*argAlloca);
      }
      logger.decreaseIndent();
    );
  }
  LLVM_DEBUG(log().decreaseIndent(2));
  return newF;
}

void InitializerPass::logInfoPropagationQueue() {
  Logger &logger = log();
  if (infoPropagationQueue.size() < 1000) {
    for (Value *value : infoPropagationQueue) {
      logger.log("[Value] ", raw_ostream::Colors::BLACK).logValueln(value);
      logger.increaseIndent();
      logger << "valueInitInfo: " << taffoInitInfo.getValueInitInfo(value) << "\n";
      logger.decreaseIndent();
    }
  }
  else
    logger.log("Not logging the queue because it exceeds 1000 items", raw_ostream::Colors::YELLOW);
}
