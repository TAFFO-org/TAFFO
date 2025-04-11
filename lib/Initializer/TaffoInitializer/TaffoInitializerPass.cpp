#include "TaffoInitializerPass.hpp"

#include "Debug/Logger.hpp"
#include "Types/TypeUtils.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "IndirectCallPatcher.hpp"
#include "PtrCasts.hpp"
#include "OpenCLKernelPatcher.hpp"
#include "CudaKernelPatcher.hpp"

#include <llvm/Transforms/Utils/Cloning.h>
#include <memory>

#define DEBUG_TYPE "taffo-init"

using namespace taffo;
using namespace llvm;

cl::opt<bool> ManualFunctionCloning("manualclone",
  cl::desc("Enables function cloning only for annotated functions"),
  cl::init(false));

cl::opt<bool> OpenCLKernelMode("oclkern",
  cl::desc("Allows cloning of OpenCL kernel functions"),
  cl::init(false));

cl::opt<bool> CudaKernelMode("cudakern",
  cl::desc("Allows cloning of Cuda kernel functions"),
  cl::init(false));

PreservedAnalyses TaffoInitializerPass::run(Module &m, ModuleAnalysisManager &) {
  LLVM_DEBUG(Logger::getInstance().logln("[InitializerPass]", raw_ostream::Colors::MAGENTA));
  TaffoInfo::getInstance().initializeFromFile("taffo_typededucer.json", m);

  if (OpenCLKernelMode) {
    LLVM_DEBUG(dbgs() << "OpenCLKernelMode == true!\n");
    createOpenCLKernelTrampolines(m);
  }
  else if (CudaKernelMode) {
    LLVM_DEBUG(dbgs() << "CudaKernelMode == true!\n");
    createCudaKernelTrampolines(m);
  }

  LLVM_DEBUG(printAnnotatedObj(m));

  manageIndirectCalls(m);

  readAndRemoveLocalAnnotations(m);
  readGlobalAnnotations(m, true);
  readGlobalAnnotations(m, false);

  if (Function *startingPoint = findStartingPointFunctionGlobal(m)) {
    LLVM_DEBUG(dbgs() << "Found starting point using global __taffo_vra_starting_function: " << startingPoint->getName() << "\n");
    TaffoInfo::getInstance().addStartingPoint(*startingPoint);
  }

  ConvQueueType rootsQueue;
  rootsQueue.insert(global.begin(), global.end());
  rootsQueue.insert(local.begin(), local.end());
  AnnotationCount = rootsQueue.size();

  ConvQueueType queue;
  buildConversionQueueForRootValues(rootsQueue, queue);

  // Store valueInfo for each value in the queue
  for (const auto &[value, convQueueInfo] : queue)
    setValueInfo(value, convQueueInfo);

  removeAnnotationCalls(queue);

  SmallPtrSet<Function*, 10> callTrace;
  generateFunctionSpace(queue, global, callTrace);

  LLVM_DEBUG(printConversionQueue(queue));
  setFunctionArgsInfo(m, queue);

  TaffoInfo::getInstance().dumpToFile("taffo_info_init.json", m);
  LLVM_DEBUG(Logger::getInstance().logln("[End of InitializerPass]", raw_ostream::Colors::MAGENTA));
  return PreservedAnalyses::all();
}

void TaffoInitializerPass::removeAnnotationCalls(ConvQueueType &queue) {
  for (auto iter = queue.begin(); iter != queue.end();) {
    Value *v = iter->first;

    if (auto *call = dyn_cast<CallInst>(v))
      if (call->getCalledFunction() && call->getCalledFunction()->getName().starts_with("llvm.var.annotation")) {
        iter = queue.erase(iter);
        call->eraseFromParent();
        TaffoInfo::getInstance().eraseValue(*call);
        continue;
      }
    iter++;

    // TODO: remove global annotations
  }
}

void TaffoInitializerPass::setValueInfo(Value *value, const ConvQueueInfo &valueConvQueueInfo) {
  auto valueInfo = valueConvQueueInfo.valueInfo;
  if (isa<Instruction>(value) || isa<GlobalObject>(value))
    TaffoInfo::getInstance().setValueWeight(*value, valueConvQueueInfo.rootDistance);
  if (auto *inst = dyn_cast<Instruction>(value)) {
    if (auto scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo)) {
      if (TaffoInfo::getInstance().isConversionDisabled(*inst)) {
        LLVM_DEBUG(dbgs() << "Blocking conversion for shared variable" << *inst << "\n");
        scalarInfo->conversionEnabled = false;
      }
    }
  }
  TaffoInfo::getInstance().setValueInfo(*value, valueInfo);
}

/**
 * @brief Builds the conversion queue for root values.
 *
 * Starting from the set of root values, this function propagates conversion info
 * through the IR graph by iterating over each value's users (forward propagation)
 * and then backtracking through instruction operands (backward propagation)
 * until no new values are discovered
 *
 * @param roots The set of root values and their conversion info
 * @param queue The conversion queue to be built and updated
 */
void TaffoInitializerPass::buildConversionQueueForRootValues(const ConvQueueType &roots, ConvQueueType &queue) {
  LLVM_DEBUG(dbgs() << "***** begin " << __FUNCTION__ << "\nInitial ");

  // Insert all root entries into the queue
  queue.insert(roots.begin(), roots.end());
  LLVM_DEBUG(printConversionQueue(queue));

  // Set to track processed values to avoid reprocessing
  SmallPtrSet<Value*, 8> visited;

  // Continue propagation until the queue size stabilizes
  size_t prevQueueSize = 0;
  while (prevQueueSize < queue.size()) {
    LLVM_DEBUG(dbgs() << "***** " << __FUNCTION__ << " iter " << prevQueueSize << " < " << queue.size() << "\n";);
    prevQueueSize = queue.size();

    // Forward propagation: process each value in the queue
    for (auto [value, valueConvQueueInfo] : queue) {
      visited.insert(value);

      LLVM_DEBUG(dbgs() << "[Value] " << *value << "\n");
      if (auto *inst = dyn_cast<Instruction>(value))
        LLVM_DEBUG(dbgs() << " - function: " << inst->getFunction()->getName() << "\n");
      LLVM_DEBUG(dbgs() << " - root distance: " << valueConvQueueInfo.rootDistance << "\n");

      // Process each user of the current value
      for (auto *user : value->users()) {
        // Ignore user if it is an annotation string
        if (auto *userGlobalObject = dyn_cast<GlobalObject>(user))
          if (userGlobalObject->hasSection() && userGlobalObject->getSection() == "llvm.metadata")
            continue;
        // Skip PHI nodes already processed
        if (isa<PHINode>(user) && visited.contains(user))
          continue;

        // Extract existing conversion info of the user from the queue if available, otherwise initialize them
        auto userIter = queue.find(user);
        ConvQueueInfo userConvQueueInfo;
        if (userIter != queue.end()) {
          userConvQueueInfo = userIter->second;
          queue.erase(userIter);
        }

        LLVM_DEBUG(dbgs() << " [User] " << *user << "\n");
        if (auto *inst = dyn_cast<Instruction>(user))
          LLVM_DEBUG(dbgs() << "  - function: " << inst->getFunction()->getName() << "\n");

        // Determine the new backtracking depth for propagation
        unsigned int valueDepth = valueConvQueueInfo.backtrackingDepthLeft;
        valueDepth = std::min(valueDepth, valueDepth - 1);

        if (valueDepth <= 1 && isa<StoreInst>(user)) {
          auto *store = cast<StoreInst>(user);
          Type *valueOperandType = getUnwrappedType(store->getValueOperand());
          if (valueOperandType->isFloatingPointTy()) {
            LLVM_DEBUG(dbgs() << "  backtracking to stored value def instr\n");
            valueDepth = 1;
          }
        }
        if (valueDepth > 0) {
          unsigned int &userDepth = userConvQueueInfo.backtrackingDepthLeft;
          userDepth = std::max(valueDepth, userDepth);
        }
        // Update conversion info for the user based on the parent's conversion info
        createUserInfo(value, valueConvQueueInfo, user, userConvQueueInfo);
        // Insert user at the end of the queue with the updated conversion info
        queue.insert(user, userConvQueueInfo);
      }
    }

    // Backward propagation: process operands of each instruction in reverse order
    for (auto iter = queue.end(); iter != queue.begin();) {
      iter--;
      Value *value = iter->first;
      ConvQueueInfo valueConvQueueInfo = iter->second;

      unsigned int valueDepth = valueConvQueueInfo.backtrackingDepthLeft;
      if (valueDepth == 0)
        continue;
      auto *inst = dyn_cast<Instruction>(value);
      if (!inst)
        continue;

      LLVM_DEBUG(dbgs() << "BACKTRACKING " << *value << ", depth left = " << valueDepth << "\n");

      // Process each operand of the instruction
      int OpIdx = -1;
      for (Value *user : inst->operands()) {
        OpIdx++;
        // Skip operands that are not a User or an Argument
        if (!isa<User>(user) && !isa<Argument>(user)) {
          LLVM_DEBUG(dbgs() << " - " << user->getNameOrAsOperand() << " not a User or an Argument, ignoring\n");
          continue;
        }
        // Skip functions and block addresses
        if (isa<Function>(user) || isa<BlockAddress>(user)) {
          LLVM_DEBUG(dbgs() << " - " << user->getNameOrAsOperand() << " is a function/block address, ignoring\n");
          continue;
        }
        // Skip constants
        if (isa<Constant>(user)) {
          LLVM_DEBUG(dbgs() << " - " << user->getNameOrAsOperand() << " is a constant, ignoring\n");
          continue;
        }
        LLVM_DEBUG(dbgs() << " - " << *user);

        if (!getUnwrappedType(user)->isFloatTy()) {
          LLVM_DEBUG(dbgs() << " not a float, ignoring\n");
          continue;
        }

        bool alreadyIn = false;
        ConvQueueInfo userConvQueueInfo;
        userConvQueueInfo.backtrackingDepthLeft = std::min(valueDepth, valueDepth - 1);
        auto userIter = queue.find(user);
        if (userIter != queue.end()) {
          if (userIter < iter)
            alreadyIn = true;
          else
            queue.erase(userIter);
        }
        if (!alreadyIn) {
          LLVM_DEBUG(dbgs() << "  enqueued\n");
          createUserInfo(value, valueConvQueueInfo, user, userConvQueueInfo);
          queue.insertAt(queue.begin(), user, userConvQueueInfo);
        } else {
          LLVM_DEBUG(dbgs() << " already in, not updating info\n");
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "***** end " << __PRETTY_FUNCTION__ << "\n");
}

void TaffoInitializerPass::createUserInfo(Value *value, ConvQueueInfo &valueConvQueueInfo, Value *user, ConvQueueInfo &userConvQueueInfo) {
  std::shared_ptr<ValueInfo> &valueInfo = valueConvQueueInfo.valueInfo;
  unsigned int &valueRootDistance = valueConvQueueInfo.rootDistance;
  std::shared_ptr<ValueInfo> &userInfo = userConvQueueInfo.valueInfo;
  unsigned int &userRootDistance = userConvQueueInfo.rootDistance;
  TaffoInfo &taffoInfo = TaffoInfo::getInstance();
  /* Propagate metadata from the instruction closest to a root */
  if (userRootDistance > std::max(valueRootDistance, valueRootDistance + 1)) {
    userRootDistance = std::max(valueRootDistance, valueRootDistance + 1);
    LLVM_DEBUG(dbgs() << "  " << __FUNCTION__ << ":\n");
    LLVM_DEBUG(dbgs() << "   updated root distance: " << userRootDistance << "\n");
    std::shared_ptr<TransparentType> valueType = taffoInfo.getTransparentType(*value);
    std::shared_ptr<TransparentType> userType = taffoInfo.getTransparentType(*user);
    LLVM_DEBUG(dbgs() << "   valueType = " << *valueType << ", valueInfo = " << (valueInfo ? valueInfo->toString() : "(null)") << "\n");
    LLVM_DEBUG(dbgs() << "   userType = " << *userType << ", userInfo = " << (userInfo ? userInfo->toString() : "(null)") << "\n");
    if (isa<CallInst>(user))
      userInfo = std::make_shared<ScalarInfo>(nullptr, nullptr, nullptr, true);
    else if (!valueType->isStructType() && !userType->isStructType()) { //TODO FIX SOON
      userInfo = valueInfo->clone();
      LLVM_DEBUG(dbgs() << "   copied userInfo " << userInfo->toString() << " from valueInfo " << valueInfo->toString() << "\n");
    }
    else {
      if (userType->isStructType())
        userInfo = StructInfo::createFromTransparentType(std::static_ptr_cast<TransparentStructType>(userType));
      else
        userInfo = std::make_shared<ScalarInfo>(nullptr, nullptr, nullptr, true);
      LLVM_DEBUG(dbgs() << "   created userInfo " << userInfo->toString() << " because valueType != userType\n");
    }
  }

  // The conversion enabling flag shall be true if at least one of the parents of the children has it enabled
  std::shared_ptr<ScalarInfo> valueScalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo);
  std::shared_ptr<ScalarInfo> userScalarInfo = std::dynamic_ptr_cast<ScalarInfo>(userInfo);
  if (userScalarInfo && valueScalarInfo && valueScalarInfo->conversionEnabled)
    userScalarInfo->conversionEnabled = true;

  // Fix valueInfo if this is a GetElementPtrInst
  std::shared_ptr<ValueInfo> gepValueInfo = extractGEPValueInfo(value, valueInfo, user);
  if (gepValueInfo) {
    userInfo = gepValueInfo;
    userInfo->bufferId = valueInfo->bufferId;
  }
  // Copy BufferID across loads and bitcasts
  if (isa<BitCastInst>(user) || isa<LoadInst>(user)) {
    userInfo->bufferId = valueInfo->bufferId;
  }
}

std::shared_ptr<ValueInfo> TaffoInitializerPass::extractGEPValueInfo(
    const Value *value, std::shared_ptr<ValueInfo> valueInfo, const Value *user) {
  if (!valueInfo)
    return nullptr;
  assert(value && user);
  const auto *gepi = dyn_cast<GetElementPtrInst>(user);
  if (!gepi)
    return nullptr;

  if (gepi->getPointerOperand() != value) {
    /* if the used value is not the pointer, then it must be one of the
     * indices; keep everything as is */
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "[extractGEPValueInfo] begin\n");
  LLVM_DEBUG(dbgs() << "Initial GEP object metadata is " << valueInfo->toString() << "\n");

  Type *source_element_type = gepi->getSourceElementType();
  for (auto idx_it = gepi->idx_begin() + 1; // skip first index
       idx_it != gepi->idx_end(); ++idx_it) {
    LLVM_DEBUG(dbgs() << "[extractGEPValueInfo] source_element_type=" << *source_element_type << "\n");
    if (isa<ArrayType>(source_element_type) || isa<VectorType>(source_element_type))
      continue;

    if (const ConstantInt *int_i = dyn_cast<ConstantInt>(*idx_it)) {
      int n = static_cast<int>(int_i->getSExtValue());
      valueInfo = cast<StructInfo>(valueInfo.get())->getField(n);
      source_element_type =
          cast<StructType>(source_element_type)->getTypeAtIndex(n);
    } else {
      LLVM_DEBUG(dbgs() << "[extractGEPValueInfo] fail, non-const index encountered\n");
      return nullptr;
    }
  }
  if (valueInfo)
    LLVM_DEBUG(dbgs() << "[extractGEPValueInfo] end, valueInfo=" << valueInfo->toString() << "\n");
  else
    LLVM_DEBUG(dbgs() << "[extractGEPValueInfo] end, valueInfo=NULL\n");
  return valueInfo ? valueInfo->clone() : nullptr;
}

void TaffoInitializerPass::generateFunctionSpace(
    ConvQueueType &convQueue, ConvQueueType &global, SmallPtrSet<Function *, 10> &callTrace) {
  LLVM_DEBUG(dbgs() << "***** begin " << __PRETTY_FUNCTION__ << "\n");

  for (const auto &[value, _] : convQueue) {
    if (!isa<CallInst>(value) && !isa<InvokeInst>(value))
      continue;
    auto *call = dyn_cast<CallBase>(value);

    Function *oldF = call->getCalledFunction();
    if (!oldF) {
      LLVM_DEBUG(dbgs() << "found bitcasted funcptr in " << *value << ", skipping\n");
      continue;
    }
    if (isSpecialFunction(oldF))
      continue;
    if (ManualFunctionCloning) {
      if (enabledFunctions.count(oldF) == 0) {
        LLVM_DEBUG(dbgs() << "skipped cloning of function from call " << *value << ": function disabled\n");
        continue;
      }
    }

    std::vector<Value*> newVals;
    Function *newF = createFunctionAndQueue(call, convQueue, global, newVals);
    call->setCalledFunction(newF);
    enabledFunctions.insert(newF);

    TaffoInfo::getInstance().setTaffoFunction(*oldF, *newF);
    for (auto newValue : newVals) {
      auto *newInst = dyn_cast<Instruction>(newValue);
      if (!newInst || !TaffoInfo::getInstance().hasValueInfo(*newInst))
        setValueInfo(newValue, convQueue.at(newValue));
    }

    /* Reconstruct the value info for the values which are in the top-level
     * conversion queue and in the oldF
     * Allows us to properly process call functions */
    // TODO: REWRITE USING THE VALUE MAP RETURNED BY CloneFunctionInto
    for (BasicBlock &bb : *newF) {
      for (Instruction &inst : bb) {
        if (std::shared_ptr<ValueInfo> valueInfo = TaffoInfo::getInstance().getValueInfo(inst)) {
          ConvQueueInfo instConvQueueInfo;
          instConvQueueInfo.valueInfo = valueInfo->clone();
          int weight = TaffoInfo::getInstance().getValueWeight(inst);
          if (weight >= 0)
            instConvQueueInfo.rootDistance = weight;
          convQueue.insert(&inst, instConvQueueInfo);
          LLVM_DEBUG(dbgs() << "  enqueued & rebuilt valueInfo of " << inst << " in " << newF->getName() << "\n");
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "***** end " << __PRETTY_FUNCTION__ << "\n");
}

Function *TaffoInitializerPass::createFunctionAndQueue(
    CallBase *call, ConvQueueType &vals, ConvQueueType &global, std::vector<Value*> &convQueue) {
  LLVM_DEBUG(dbgs() << "***** begin " << __PRETTY_FUNCTION__ << "\n");

  /* vals: conversion queue of caller
   * global: global values to copy in all conversion queues
   * convQueue: output conversion queue of this function */

  Function *oldF = call->getCalledFunction();
  Function *newF = Function::Create(
      oldF->getFunctionType(), oldF->getLinkage(),
      oldF->getName(), oldF->getParent());

  TaffoInfo &taffoInfo = TaffoInfo::getInstance();

  // Create Val2Val mapping and clone function
  ValueToValueMapTy valueMap;
  for (auto &&[oldArg, newArg] : zip(oldF->args(), newF->args())) {
    newArg.setName(oldArg.getName());
    valueMap.insert({&oldArg, &newArg});
  }
  SmallVector<ReturnInst*, 100> returns;
  CloneFunctionInto(newF, oldF, valueMap, CloneFunctionChangeType::GlobalChanges, returns);
  for (auto &&[oldValue, newValue] : valueMap) {
    if (std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*oldValue))
      taffoInfo.setValueInfo(*newValue, valueInfo);
    if (taffoInfo.hasTransparentType(*oldValue))
      taffoInfo.setTransparentType(*newValue, taffoInfo.getTransparentType(*const_cast<Value*>(oldValue)));
  }
  if (std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*oldF))
    taffoInfo.setValueInfo(*newF, valueInfo);
  if (taffoInfo.hasTransparentType(*oldF))
    taffoInfo.setTransparentType(*newF, taffoInfo.getTransparentType(*oldF));

  if (!OpenCLKernelMode && !CudaKernelMode)
    newF->setLinkage(GlobalVariable::LinkageTypes::InternalLinkage);

  FunctionCloned++;

  ConvQueueType roots;
  LLVM_DEBUG(dbgs() << "Create function from " << oldF->getName() << " to " << newF->getName() << "\n");
  LLVM_DEBUG(dbgs() << "  CallBase instr " << *call << " [" << call->getFunction()->getName() << "]\n");
  for (Argument &newArg : newF->args()) {
    auto user_begin = newArg.user_begin();
    if (user_begin == newArg.user_end()) {
      LLVM_DEBUG(dbgs() << "  Arg nr. " << newArg.getArgNo() << " skipped, value has no uses\n");
      continue;
    }

    Value *callOperand = call->getOperand(newArg.getArgNo());
    Value *allocaOfArgument = user_begin->getOperand(1);

    for (; user_begin != newArg.user_end(); ++user_begin) {
      if (auto *store = dyn_cast<StoreInst>(*user_begin)) {
        allocaOfArgument = store->getOperand(1);
      }
    }

    if (!isa<AllocaInst>(allocaOfArgument))
      allocaOfArgument = nullptr;

    if (!vals.contains(callOperand)) {
      LLVM_DEBUG(dbgs() << "  Arg nr. " << newArg.getArgNo() << " skipped, callOperand has no valueInfo\n");
      continue;
    }

    ConvQueueInfo &callOperandQueueInfo = vals[callOperand];
    ConvQueueInfo argumentConvQueueInfo;
    vals.insert(&newArg, argumentConvQueueInfo);

    std::shared_ptr<ValueInfo> &argumentInfo = argumentConvQueueInfo.valueInfo;
    const std::shared_ptr<ValueInfo> &callOperandInfo = callOperandQueueInfo.valueInfo;

    // Mark the argument itself (set it as a new root as well in VRA-less mode)
    argumentInfo = callOperandInfo->clone();
    argumentConvQueueInfo.rootDistance = std::max(callOperandQueueInfo.rootDistance, callOperandQueueInfo.rootDistance + 1);
    if (!allocaOfArgument)
      roots.insert(&newArg, argumentConvQueueInfo);

    if (allocaOfArgument) {
      ConvQueueInfo &allocaConvQueueInfo = vals[allocaOfArgument];
      std::shared_ptr<ValueInfo> &allocaInfo = allocaConvQueueInfo.valueInfo;
      // Mark the alloca used for the argument (in O0 opt lvl)
      // let it be a root in VRA-less mode
      allocaInfo = callOperandInfo->clone();
      allocaConvQueueInfo.rootDistance = std::max(callOperandQueueInfo.rootDistance, callOperandQueueInfo.rootDistance + 2);
      roots.insert(allocaOfArgument, allocaConvQueueInfo);
    }

    // Propagate BufferID
    argumentInfo->bufferId = callOperandInfo->bufferId;

    LLVM_DEBUG(dbgs() << "  Arg nr. " << newArg.getArgNo() << " processed\n");
    LLVM_DEBUG(dbgs() << "    vi = " << *argumentInfo << "\n");
    if (allocaOfArgument)
      LLVM_DEBUG(dbgs() << "    enqueued alloca of argument " << *allocaOfArgument << "\n");
  }

  ConvQueueType tmpVals;
  roots.insertAt(roots.begin(), global.begin(), global.end());
  ConvQueueType localFix;
  readLocalAnnotations(*newF, localFix);
  roots.insertAt(roots.begin(), localFix.begin(), localFix.end());
  buildConversionQueueForRootValues(roots, tmpVals);
  for (auto val : tmpVals) {
    if (auto *inst = dyn_cast<Instruction>(val.first)) {
      if (inst->getFunction() == newF) {
        vals.insert(val);
        convQueue.push_back(val.first);
        LLVM_DEBUG(dbgs() << "  enqueued " << *inst << " in " << newF->getName() << "\n");
      }
    }
  }

  LLVM_DEBUG(dbgs() << "***** end " << __PRETTY_FUNCTION__ << "\n");
  return newF;
}

void TaffoInitializerPass::printConversionQueue(ConvQueueType &queue) {
  if (queue.size() < 1000) {
    dbgs() << "conversion queue:\n";
    for (const auto &[value, convQueueInfo] : queue) {
      dbgs() << "bt=" << convQueueInfo.backtrackingDepthLeft << " ";
      std::shared_ptr<ValueInfo> valueInfo = convQueueInfo.valueInfo;
      dbgs() << "vi=" << (valueInfo ? valueInfo->toString() : "(null)") << " ";
      if (auto *inst = dyn_cast<Instruction>(value))
        dbgs() << "fun=" << inst->getFunction()->getName() << " ";
      dbgs() << *value << "\n";
    }
    dbgs() << "\n";
  } else {
    dbgs() << "not printing the conversion queue because it exceeds 1000 items";
  }
}
