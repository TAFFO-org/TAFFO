#include "ConversionPass.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"
#include "TypeDeductionAnalysis.hpp"
#include "Types/ConversionType.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conv"

cl::opt<unsigned> maxTotalBitsConv("maxtotalbitsconv",
                                   cl::value_desc("bits"),
                                   cl::desc("Maximum amount of bits used in fmul and fdiv conversion."),
                                   cl::init(128));

cl::opt<unsigned> minQuotientFrac("minquotientfrac",
                                  cl::value_desc("bits"),
                                  cl::desc("minimum number of quotient fractional preserved"),
                                  cl::init(5));

PreservedAnalyses ConversionPass::run(Module& m, ModuleAnalysisManager&) {
  LLVM_DEBUG(log().logln("[ConversionPass]", Logger::Magenta));
  taffoInfo.initializeFromFile(DTA_TAFFO_INFO, m);
  dataLayout = &m.getDataLayout();

  SmallVector<Value*, 32> localValues;
  SmallVector<Value*, 32> globalValues;
  buildAllLocalConvInfo(m, localValues);
  buildGlobalConvInfo(m, globalValues);

  std::vector convQueue(localValues.begin(), localValues.end());
  convQueue.insert(convQueue.begin(), globalValues.begin(), globalValues.end());
  valueInfoCount = convQueue.size();

  createConversionQueue(convQueue);
  propagateCalls(convQueue, globalValues, m);
  LLVM_DEBUG(printConversionQueue(convQueue));
  conversionCount = convQueue.size();

  // Collect heap allocations before conversion
  auto heapAllocations = collectHeapAllocations(m);

  performConversion(convQueue);

  // Collect heap allocations after conversion
  HeapAllocationsVec newHeapAllocations = collectHeapAllocations(m);
  // Adjust size of heap allocations using the info before and after conversion
  adjustSizeOfHeapAllocations(m, heapAllocations, newHeapAllocations);

  closePhiLoops();
  cleanup(convQueue);

  convertIndirectCalls(m);

  cleanUpOpenCLKernelTrampolines(&m);
  cleanUpOriginalFunctions(m);

  taffoInfo.dumpToFile(CONVERSION_TAFFO_INFO, m);
  LLVM_DEBUG(log().logln("[End of ConversionPass]", Logger::Magenta));
  return PreservedAnalyses::none();
}

void ConversionPass::createConversionQueue(std::vector<Value*>& values) {
  Logger& logger = log();
  LLVM_DEBUG(logger.logln("[Creating conversion queue]", Logger::Blue));
  size_t current = 0;
  while (current < values.size()) {
    Value* value = values.at(current);
    ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(value);

    auto indenter = logger.getIndenter();
    LLVM_DEBUG(
      logger.log("[Value] ", Logger::Bold).logValueln(value);
      indenter.increaseIndent());

    SmallPtrSet<Value*, 8> userRoots;
    for (Value* oldRoot : valueConvInfo->roots)
      if (taffoConvInfo.getValueConvInfo(oldRoot)->roots.empty())
        userRoots.insert(oldRoot);
    valueConvInfo->roots.clear();
    valueConvInfo->roots.insert(userRoots.begin(), userRoots.end());
    if (userRoots.empty())
      userRoots.insert(value);

    if (auto* phi = dyn_cast<PHINode>(value))
      openPhiLoop(phi);

    for (auto* user : value->users()) {
      auto indenter = logger.getIndenter();
      LLVM_DEBUG(
        logger.log("[User] ", Logger::Bold).logValueln(user);
        indenter.increaseIndent());
      if (auto* inst = dyn_cast<Instruction>(user))
        if (functionPool.find(inst->getFunction()) != functionPool.end()) {
          LLVM_DEBUG(logger.logln("value belongs to an original function: skipping", Logger::Yellow));
          continue;
        }

      // Insert user at the end of the queue. If user is already in queue, move it to the end instead
      for (size_t i = 0; i < values.size();) {
        if (values[i] == user) {
          values.erase(values.begin() + i);
          if (i < current)
            current--;
        }
        else
          i++;
      }
      values.push_back(user);

      if (!taffoConvInfo.hasValueConvInfo(user)) {
        LLVM_DEBUG(logger.logln("value will not be converted because it has no valueConvInfo", Logger::Yellow));
        taffoInfo.getOrCreateTransparentType(*user); // Create transparent type if missing
        auto* userConvInfo = taffoConvInfo.createValueConvInfo(user);
        LLVM_DEBUG(log().log("new valueConvInfo: ").logln(*userConvInfo, Logger::Cyan));
      }

      if (auto* phi = dyn_cast<PHINode>(user))
        openPhiLoop(phi);
      taffoConvInfo.getValueConvInfo(user)->roots.insert(userRoots.begin(), userRoots.end());
    }
    current++;
  }
  LLVM_DEBUG(logger.logln("[Conversion queue created]", Logger::Blue));

  for (Value* value : values) {
    assert(taffoConvInfo.hasValueConvInfo(value) && "all values in the queue should have conversionInfo by now");
    if (!taffoConvInfo.getNewType(value) && taffoInfo.getTransparentType(*value)->containsFloatingPointType()
        && !isAlwaysConvertible(value)) {
      LLVM_DEBUG(
        logger.log("[Value] ", Logger::Bold).logValueln(value);
        auto indenter = logger.getIndenter();
        indenter.increaseIndent();
        logger.logln("value will not be converted because its conversionInfo is incomplete", Logger::Yellow));
      taffoConvInfo.getValueConvInfo(value)->setNewType(nullptr);
    }

    SmallPtrSetImpl<Value*>& roots = taffoConvInfo.getValueConvInfo(value)->roots;
    if (roots.empty()) {
      taffoConvInfo.getValueConvInfo(value)->isRoot = true;
      if (isa<Instruction>(value) && !isa<AllocaInst>(value)) {
        // Non-alloca roots must have been generated by backtracking
        taffoConvInfo.getValueConvInfo(value)->isBacktrackingNode = true;
      }
      roots.insert(value);
    }
  }
}

void ConversionPass::propagateCalls(std::vector<Value*>& convQueue,
                                    const SmallVectorImpl<Value*>& globalValues,
                                    Module& m) {
  SmallPtrSet<Function*, 16> oldFunctions;

  for (size_t i = 0; i < convQueue.size(); i++) {
    Value* value = convQueue[i];
    auto* call = dyn_cast<CallBase>(value);
    if (!call)
      continue;
    Function* oldF = call->getCalledFunction();
    // Bitcasted function pointers and such not handled
    if (!oldF)
      continue;
    bool alreadyHandledNewF;
    Function* newF = createConvertedFunctionForCall(call, &alreadyHandledNewF);
    if (!newF)
      continue;
    if (alreadyHandledNewF) {
      oldFunctions.insert(oldF);
      continue;
    }

    // Create Val2Val mapping and clone function
    ValueToValueMapTy origValToCloned;
    for (auto&& [oldArg, newArg] : zip(oldF->args(), newF->args())) {
      newArg.setName(oldArg.getName());
      origValToCloned.insert({&oldArg, &newArg});
    }
    SmallVector<ReturnInst*, 8> returns;
    CloneFunctionInto(newF, oldF, origValToCloned, CloneFunctionChangeType::GlobalChanges, returns);
    // after CloneFunctionInto, valueMap maps all values from the oldF to the newF (not just the arguments)

    for (const auto& [oldValue, newValue] : origValToCloned) {
      if (taffoInfo.hasValueInfo(*oldValue))
        taffoInfo.setValueInfo(*newValue, taffoInfo.getValueInfo(*oldValue));
      if (taffoInfo.hasTransparentType(*oldValue))
        taffoInfo.setTransparentType(*newValue, taffoInfo.getTransparentType(*oldValue)->clone());
    }
    /* CloneFunctionInto also fixes the attributes of the arguments.
     * This is not exactly what we want for OpenCL kernels because the alignment
     * after the conversion is not defined by us but by the OpenCL runtime.
     * So we need to compensate for this. */
    // TODO fix cuda
    if (newF->getCallingConv() == CallingConv::SPIR_KERNEL /*|| MetadataManager::isCudaKernel(m, oldF)*/) {
      /* OpenCL spec says the alignment is equal to the size of the type */
      SmallVector<AttributeSet, 4> NewAttrs(newF->arg_size());
      AttributeList OldAttrs = newF->getAttributes();
      for (unsigned ArgId = 0; ArgId < newF->arg_size(); ArgId++) {
        Argument* Arg = newF->getArg(ArgId);
        if (!Arg->getType()->isPointerTy())
          continue;
        Type* ArgTy = getFullyUnwrappedType(Arg);
        Align align(ArgTy->getScalarSizeInBits() / 8);
        AttributeSet OldArgAttrs = OldAttrs.getParamAttrs(ArgId);
        AttributeSet NewArgAttrs = OldArgAttrs.addAttributes(
          newF->getContext(),
          AttributeSet::get(newF->getContext(), {Attribute::getWithAlignment(newF->getContext(), align)}));
        NewAttrs[ArgId] = NewArgAttrs;
        LLVM_DEBUG(log() << "Fixed align of arg " << ArgId << " (" << *Arg << ") to " << align.value() << "\n");
      }
      newF->setAttributes(
        AttributeList::get(newF->getContext(), OldAttrs.getFnAttrs(), OldAttrs.getRetAttrs(), NewAttrs));
    }

    // propagate conversion
    std::vector<Value*> newQueue;
    for (auto&& [oldArg, newArg] : zip(oldF->args(), newF->args())) {
      ValueConvInfo* newArgConvInfo = taffoConvInfo.getValueConvInfo(&newArg);
      if (*newArgConvInfo->getNewType() != *newArgConvInfo->getOldType()) {
        // Create a fake value to maintain type consistency because
        // createConvertedFunctionForCall has RAUWed all arguments
        // FIXME: is there a cleaner way to do this?
        std::string name("placeholder");
        if (newArg.hasName())
          name += "." + newArg.getName().str();
        Value* placeholder = createPlaceholder(oldArg.getType(), &newF->getEntryBlock(), name);
        // Reimplement RAUW to defeat the same-type check
        while (!newArg.materialized_use_empty()) {
          Use& use = *(newArg.uses().begin());
          use.set(placeholder);
        }

        taffoInfo.setTransparentType(*placeholder, taffoInfo.getOrCreateTransparentType(oldArg)->clone());
        taffoInfo.setValueInfo(*placeholder, taffoInfo.getValueInfo(oldArg)->clone());

        ValueConvInfo* placeholderConvInfo = taffoConvInfo.createValueConvInfo(placeholder);
        *placeholderConvInfo = *newArgConvInfo;
        placeholderConvInfo->isArgumentPlaceholder = true;
        placeholderConvInfo->enableConversion();
        LLVM_DEBUG(log().log("new valueConvInfo: ").logln(*placeholderConvInfo, Logger::Cyan));
        convertedValues[placeholder] = &newArg;
        newQueue.push_back(placeholder);
      }
    }

    newQueue.insert(newQueue.end(), globalValues.begin(), globalValues.end());
    SmallVector<Value*, 32> newLocalValues;
    buildLocalConvInfo(*newF, newLocalValues);
    newQueue.insert(newQueue.end(), newLocalValues.begin(), newLocalValues.end());

    for (ReturnInst* ret : returns) {
      taffoConvInfo.getValueConvInfo(ret)->enableConversion();
      newQueue.push_back(ret);
    }

    LLVM_DEBUG(log() << "creating conversion queue of new function " << newF->getName() << "\n");
    createConversionQueue(newQueue);

    oldFunctions.insert(oldF);

    // Put the instructions from the new function in queue
    for (Value* newValue : newQueue)
      if (auto* inst = dyn_cast<Instruction>(newValue))
        if (inst->getFunction() == newF && !is_contained(convQueue, inst))
          convQueue.push_back(newValue);
  }

  // Remove instructions of the old function from the queue
  size_t i, j;
  for (i = 0, j = 0; j < convQueue.size(); j++) {
    convQueue[i] = convQueue[j];
    Value* value = convQueue[j];
    bool toDelete = false;
    if (auto* inst = dyn_cast<Instruction>(value)) {
      if (oldFunctions.contains(inst->getFunction())) {
        toDelete = true;
        if (auto* phi = dyn_cast_or_null<PHINode>(inst))
          phiNodeInfo.erase(phi);
      }
    }
    else if (auto* arg = dyn_cast<Argument>(value)) {
      if (oldFunctions.contains(arg->getParent()))
        toDelete = true;
    }
    if (!toDelete)
      i++;
  }
  convQueue.resize(i);
}

Function* ConversionPass::createConvertedFunctionForCall(CallBase* call, bool* alreadyHandledNewF) {
  Function* oldF = call->getCalledFunction();

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger.logln("[Creating converted function]", Logger::Blue);
    indenter.increaseIndent();
    logger.log("call: ").logValueln(call);
    logger.log("original function:  ");
    logFunctionSignature(oldF);
    logger << "\n";);

  if (isSpecialFunction(oldF)) {
    LLVM_DEBUG(logger << "special function: ignoring\n");
    return nullptr;
  }
  if (!taffoInfo.isCloneFunction(*oldF)) {
    LLVM_DEBUG(logger << "not a function clone: ignoring\n");
    return nullptr;
  }

  Function* newF = functionPool[oldF];
  if (newF) {
    LLVM_DEBUG(logger.log("converted function already exists: ").logValueln(newF));
    if (alreadyHandledNewF)
      *alreadyHandledNewF = true;
    return newF;
  }
  if (alreadyHandledNewF)
    *alreadyHandledNewF = false;

  Type* retLLVMType = oldF->getReturnType();
  ConversionType* retConvType = nullptr;
  if (!taffoConvInfo.getValueConvInfo(oldF)->isConversionDisabled()) {
    retConvType = taffoConvInfo.getNewType(oldF);
    retLLVMType = retConvType->toLLVMType();
  }

  std::vector<Type*> argLLVMTypes;
  std::vector<ConversionType*> argConvTypes;
  for (auto& oldArg : oldF->args()) {
    Type* newLLVMType = oldArg.getType();
    ConversionType* argConvType = nullptr;
    if (!taffoConvInfo.getValueConvInfo(&oldArg)->isConversionDisabled()) {
      argConvType = taffoConvInfo.getNewType(&oldArg);
      newLLVMType = argConvType->toLLVMType();
    }
    argLLVMTypes.push_back(newLLVMType);
    argConvTypes.push_back(argConvType);
  }

  std::string suffix;
  if (retConvType)
    suffix = retConvType->toString();
  else
    suffix = "taffo";

  FunctionType* newFunType = FunctionType::get(retLLVMType, argLLVMTypes, oldF->isVarArg());
  newF = Function::Create(newFunType, oldF->getLinkage(), oldF->getName() + "_" + suffix, oldF->getParent());

  setConversionResultInfo(newF, call, retConvType);
  ValueConvInfo* newConvInfo = taffoConvInfo.getValueConvInfo(newF);
  newConvInfo->enableConversion();

  for (auto&& [oldArg, newArg, argConvType] : zip(oldF->args(), newF->args(), argConvTypes)) {
    setConversionResultInfo(&newArg, &oldArg, argConvType);
    if (argConvType)
      newArg.setName(newArg.getName() + "." + argConvType->toString()); // append convType info to arg name
  }

  LLVM_DEBUG(
    logger.log("converted function: ");
    logFunctionSignature(newF);
    logger << "\n";);
  functionPool[oldF] = newF;
  functionCreated++;
  return newF;
}

void ConversionPass::openPhiLoop(PHINode* phi) {
  if (phi->materialized_use_empty()) {
    LLVM_DEBUG(log() << "phi" << *phi << " not currently used by anything; skipping placeholder creation\n");
    return;
  }

  PhiInfo info;
  TransparentType* type = taffoInfo.getTransparentType(*phi);
  ValueConvInfo* phiConvInfo = taffoConvInfo.getValueConvInfo(phi);

  info.oldPhi = createPlaceholder(phi->getType(), phi->getParent(), "oldPhi");
  copyValueInfo(info.oldPhi, phi, type);
  ValueConvInfo* oldPhiConvInfo = taffoConvInfo.createValueConvInfo(info.oldPhi);
  *oldPhiConvInfo = *phiConvInfo;
  oldPhiConvInfo->enableConversion();
  LLVM_DEBUG(log().log("new valueConvInfo: ").logln(*oldPhiConvInfo, Logger::Cyan));
  phi->replaceAllUsesWith(info.oldPhi);

  if (!phiConvInfo->isConversionDisabled()) {
    ConversionType* oldConvType = phiConvInfo->getOldType();
    ConversionType* newConvType = phiConvInfo->getNewType();
    TransparentType* newType = newConvType->toTransparentType();
    info.newPhi = createPlaceholder(newType->toLLVMType(), phi->getParent(), "newPhi");
    copyValueInfo(info.newPhi, phi, newType);
    ValueConvInfo* newPhiConvInfo = taffoConvInfo.createValueConvInfo(info.newPhi, oldConvType);
    newPhiConvInfo->enableConversion();
    setConversionResultInfo(info.newPhi, phi, newConvType);
    LLVM_DEBUG(log().log("new valueConvInfo: ").logln(*newPhiConvInfo, Logger::Cyan));
  }
  else
    info.newPhi = info.oldPhi;
  convertedValues[info.oldPhi] = info.newPhi;

  LLVM_DEBUG(log() << "created placeholder (non-converted=[" << *info.oldPhi << "], converted=[" << *info.newPhi
                   << "]) for phi " << *phi << "\n");

  phiNodeInfo[phi] = info;
}

void ConversionPass::closePhiLoops() {
  LLVM_DEBUG(log() << __PRETTY_FUNCTION__ << " begin\n");

  for (auto data : phiNodeInfo) {
    PHINode* origphi = data.first;
    PhiInfo& info = data.second;
    Value* substphi = convertedValues.at(origphi);

    LLVM_DEBUG(log() << "restoring data flow of phi " << *origphi << "\n");
    if (info.oldPhi != info.newPhi)
      info.oldPhi->replaceAllUsesWith(origphi);
    if (!substphi) {
      LLVM_DEBUG(log() << "phi " << *origphi << "could not be converted! Trying last resort conversion\n");
      substphi = getConvertedOperand(
        origphi, *taffoConvInfo.getNewType<ConversionScalarType>(origphi), nullptr, ConvTypePolicy::ForceHint);
      assert(substphi && "phi conversion has failed");
    }

    info.newPhi->replaceAllUsesWith(substphi);
    LLVM_DEBUG(log() << "restored data flow of original phi " << *origphi << " to new value " << *substphi << "\n");
  }

  LLVM_DEBUG(log() << __PRETTY_FUNCTION__ << " end\n");
}

bool potentiallyUsesMemory(Value* val) {
  if (!isa<Instruction>(val))
    return false;
  if (isa<BitCastInst>(val))
    return false;
  if (auto* call = dyn_cast<CallInst>(val)) {
    Function* f = call->getCalledFunction();
    if (!f)
      return true;
    if (f->isIntrinsic()) {
      Intrinsic::ID fiid = f->getIntrinsicID();
      if (fiid == Intrinsic::lifetime_start || fiid == Intrinsic::lifetime_end)
        return false;
    }
    return !f->doesNotAccessMemory();
  }
  return true;
}

void ConversionPass::cleanup(const std::vector<Value*>& queue) {
  std::vector<Value*> roots;
  for (Value* value : queue)
    if (taffoConvInfo.getValueConvInfo(value)->isRoot == true)
      roots.push_back(value);

  DenseMap<Value*, bool> isRootOk;
  for (Value* root : roots)
    isRootOk[root] = true;

  for (Value* value : queue) {
    Value* cqi = convertedValues.at(value);
    assert(cqi && "every value should have been processed at this point!!");
    // TODO fix soon
    /*if (cqi == conversionError) {
      if (!potentiallyUsesMemory(value))
        continue;
      LLVM_DEBUG(
        value->print(errs());
        if (auto* inst = dyn_cast<Instruction>(value))
          errs() << " in function " << inst->getFunction()->getName();
        errs() << " not converted; invalidates roots ");
      const auto& rootsAffected = taffoConvInfo.getValueConvInfo(value)->roots;
      for (Value* root : rootsAffected) {
        isRootOk[root] = false;
        LLVM_DEBUG(root->print(errs()));
      }
      LLVM_DEBUG(errs() << '\n');
    }*/
  }

  std::vector<Instruction*> toErase;

  auto clear = [&](bool (*toDelete)(const Instruction& Y)) {
    for (Value* value : queue) {
      auto* inst = dyn_cast<Instruction>(value);
      if (!inst || !toDelete(*inst))
        continue;
      if (convertedValues.at(value) == value) {
        LLVM_DEBUG(log() << *inst << " not deleted, as it was converted by self-mutation\n");
        continue;
      }
      const auto& roots = taffoConvInfo.getValueConvInfo(value)->roots;

      bool allOk = true;
      for (Value* root : roots) {
        if (!isRootOk[root]) {
          LLVM_DEBUG(
            inst->print(errs());
            errs() << " not deleted: involves root ";
            root->print(errs());
            errs() << '\n');
          allOk = false;
          break;
        }
      }
      if (allOk) {
        if (!inst->use_empty())
          inst->replaceAllUsesWith(UndefValue::get(inst->getType()));
        toErase.push_back(inst);
      }
    }
  };

  clear(isa<StoreInst>);

  // Remove calls manually because DCE does not do it as they may have side effects
  clear(isa<CallInst>);
  clear(isa<InvokeInst>);

  clear(isa<BranchInst>);

  // Remove old phis manually as DCE cannot remove values having a circular dependence on a phi
  phiNodeInfo.clear();
  clear(isa<PHINode>);

  for (Instruction* inst : toErase)
    taffoInfo.eraseValue(inst);
}

void ConversionPass::cleanUpOriginalFunctions(Module& m) {
  for (Function& f : m)
    if (taffoInfo.isOriginalFunction(f))
      f.setLinkage(taffoInfo.getOriginalFunctionLinkage(f));
}

ConversionPass::HeapAllocationsVec ConversionPass::collectHeapAllocations(Module& m) {
  Logger& logger = log();
  LLVM_DEBUG(logger.logln("[Collecting heap allocations]", Logger::Blue));
  std::vector<std::string> names {"malloc"};
  HeapAllocationsVec heapAllocations;

  for (const auto& name : names) {
    // Mangle name to find the function
    std::string mangledName;
    raw_string_ostream mangledNameStream(mangledName);
    Mangler::getNameWithPrefix(mangledNameStream, name, m.getDataLayout());
    mangledNameStream.flush();

    auto indenter = logger.getIndenter();
    LLVM_DEBUG(
      logger << "Searching " << name << " as " << mangledName << "\n";
      indenter.increaseIndent(););

    // Search function in module
    auto fun = m.getFunction(mangledName);
    if (fun == nullptr) {
      LLVM_DEBUG(log() << "not found\n\n");
      continue;
    }

    // Iterate over function users (actual heap allocations, e.g. call to malloc)
    for (auto* user : fun->users())
      if (auto* inst = dyn_cast<Instruction>(user)) {
        if (taffoInfo.hasTransparentType(*inst)) {
          TransparentType* type = taffoInfo.getTransparentType(*inst);
          auto indenter = logger.getIndenter();
          LLVM_DEBUG(
            logger.log("[Value] ", Logger::Bold).logValueln(inst);
            indenter.increaseIndent();
            logger.log("type: ").logln(*type, Logger::Cyan));
          heapAllocations.push_back({inst, type});
        }
      }
  }
  return heapAllocations;
}

void ConversionPass::adjustSizeOfHeapAllocations(Module& m,
                                                 const HeapAllocationsVec& oldHeapAllocations,
                                                 const HeapAllocationsVec& newHeapAllocations) {
  IRBuilder builder(m.getContext());

  Logger& logger = log();
  LLVM_DEBUG(logger.logln("[Adjusting size of heap allocations]", Logger::Blue));

  for (const auto& [oldUser, oldType] : oldHeapAllocations)
    for (auto& [newUser, newType] : newHeapAllocations)
      if (newUser == convertedValues[oldUser]) {
        auto indenter = logger.getIndenter();
        LLVM_DEBUG(
          logger.log("[Value] ", Logger::Bold).logValueln(oldUser);
          indenter.increaseIndent(););
        Value* oldSizeValue = oldUser->getOperand(0);
        Value* newSizeValue = adjustHeapAllocationSize(
          oldSizeValue, oldType->getPointedType(), newType->getPointedType(), dyn_cast<Instruction>(oldUser));
        newUser->setOperand(0, newSizeValue);
      }
}

Value* ConversionPass::adjustHeapAllocationSize(Value* oldSizeValue,
                                                const std::shared_ptr<TransparentType>& oldAllocatedType,
                                                const std::shared_ptr<TransparentType>& newAllocatedType,
                                                Instruction* insertionPoint) const {
  Logger& logger = log();
  LLVM_DEBUG(
    logger << "size: " << oldSizeValue->getNameOrAsOperand() << "\n";
    logger.log("old allocated type: ").logln(*oldAllocatedType, Logger::Cyan);
    logger.log("new allocated type: ").logln(*newAllocatedType, Logger::Cyan););

  unsigned oldSize = dataLayout->getTypeAllocSize(oldAllocatedType->toLLVMType());
  unsigned newSize = dataLayout->getTypeAllocSize(newAllocatedType->toLLVMType());

  if (oldSize == newSize) {
    LLVM_DEBUG(logger << "old type is the same size of new type: doing nothing\n");
    return oldSizeValue;
  }

  LLVM_DEBUG(logger << "Ratio: " << newSize << " / " << oldSize << "\n");

  auto* constantInt = dyn_cast<ConstantInt>(oldSizeValue);
  Value* newSizeValue;
  if (constantInt == nullptr) {
    IRBuilder builder(insertionPoint);
    newSizeValue = builder.CreateMul(oldSizeValue, ConstantInt::get(oldSizeValue->getType(), newSize));
    newSizeValue = builder.CreateAdd(newSizeValue, ConstantInt::get(oldSizeValue->getType(), oldSize - 1));
    newSizeValue = builder.CreateUDiv(newSizeValue, ConstantInt::get(oldSizeValue->getType(), oldSize));
  }
  else
    newSizeValue = ConstantInt::get(oldSizeValue->getType(),
                                    (constantInt->getUniqueInteger() * newSize + (oldSize - 1)).udiv(oldSize));

  if (newSize != oldSize)
    LLVM_DEBUG(logger << "size adjusted from " << oldSizeValue << " to " << newSizeValue << "\n");
  else
    LLVM_DEBUG(logger << "size did not change\n");
  return newSizeValue;
}

Instruction* ConversionPass::getFirstInsertionPointAfter(Value* value) const {
  if (auto* arg = dyn_cast<Argument>(value))
    return &*arg->getParent()->getEntryBlock().getFirstInsertionPt();
  if (auto* inst = dyn_cast<Instruction>(value)) {
    Instruction* insertionPoint = inst->getNextNode();
    if (!insertionPoint) {
      LLVM_DEBUG(log() << __FUNCTION__ << " called on a BB-terminating inst\n");
      return nullptr;
    }
    if (isa<PHINode>(insertionPoint))
      insertionPoint = insertionPoint->getParent()->getFirstNonPHI();
    return insertionPoint;
  }
  return nullptr;
}

Value* ConversionPass::copyValueInfo(Value* dst, const Value* src, const TransparentType* dstType) const {
  if (taffoInfo.hasValueInfo(*src)) {
    std::shared_ptr<ValueInfo> dstInfo = taffoInfo.getValueInfo(*src)->clone();
    taffoInfo.setValueInfo(*dst, dstInfo);
  }
  if (dstType)
    taffoInfo.setTransparentType(*dst, dstType->clone());
  else
    taffoInfo.setTransparentType(*dst, TransparentTypeFactory::create(dst->getType()));
  return dst;
}

void ConversionPass::updateNumericTypeInfo(Value* value, bool isSigned, int fractionalBits, int bits) const {
  assert(!taffoInfo.getTransparentType(*value)->isStructTT());
  std::shared_ptr<ScalarInfo> scalarInfo;
  if (taffoInfo.hasValueInfo(*value))
    scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*value));
  else {
    scalarInfo = std::make_shared<ScalarInfo>();
    taffoInfo.setValueInfo(*value, scalarInfo);
  }
  scalarInfo->numericType = std::make_shared<FixedPointInfo>(isSigned, bits, fractionalBits);
}

void ConversionPass::updateNumericTypeInfo(Value* value, const ConversionScalarType& convType) const {
  updateNumericTypeInfo(value, convType.isSigned(), convType.getFractionalBits(), convType.getBits());
}

void ConversionPass::logFunctionSignature(Function* fun) {
  Logger& logger = log();
  logger.log(*taffoConvInfo.getCurrentType(fun), Logger::Cyan) << " " << fun->getName().str() << "(";
  for (auto iter : enumerate(fun->args())) {
    if (iter.index() != 0)
      logger << ", ";
    logger.log(*taffoConvInfo.getCurrentType(&iter.value()), iter.index() % 2 == 0 ? Logger::Cyan : Logger::Blue);
  }
  logger << ")";
}

void ConversionPass::printConversionQueue(const std::vector<Value*>& queue) const {
  Logger& logger = log();
  if (queue.size() > 1000) {
    logger << "Not printing the conversion queue because it exceeds 1000 items\n";
    return;
  }
  logger.logln("[Conversion queue]", Logger::Blue);
  for (Value* value : queue) {
    const ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(value);
    logger.log("[Value] ", Logger::Bold).logValueln(value);
    auto indenter = logger.getIndenter();
    indenter.increaseIndent();
    logger.log("valueConvInfo: ").logln(*valueConvInfo, Logger::Cyan);
  }
}
