#include "ConversionPass.hpp"
#include "Debug/Logger.hpp"
#include "TaffoConversion/TaffoConversion/FixedPointType.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"
#include "TypeDeductionAnalysis.hpp"
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

#define DEBUG_TYPE "taffo-conversion"

cl::opt<unsigned> MaxTotalBitsConv("maxtotalbitsconv",
                                   cl::value_desc("bits"),
                                   cl::desc("Maximum amount of bits used in fmul and fdiv conversion."),
                                   cl::init(128));

cl::opt<unsigned> MinQuotientFrac("minquotientfrac",
                                  cl::value_desc("bits"),
                                  cl::desc("minimum number of quotient fractional preserved"),
                                  cl::init(5));

PreservedAnalyses ConversionPass::run(Module& m, ModuleAnalysisManager& analysisManager) {
  LLVM_DEBUG(log().logln("[ConversionPass]", Logger::Magenta));
  taffoInfo.initializeFromFile("taffo_info_dta.json", m);
  dataLayout = &m.getDataLayout();

  SmallVector<Value*, 32> local;
  SmallVector<Value*, 32> global;
  buildAllLocalConversionInfo(m, local);
  buildGlobalConversionInfo(m, global);

  std::vector values(local.begin(), local.end());
  values.insert(values.begin(), global.begin(), global.end());
  ValueInfoCount = values.size();

  createConversionQueue(values);
  propagateCalls(values, global, m);
  LLVM_DEBUG(printConversionQueue(values));
  ConversionCount = values.size();

  // Collect memory allocations before conversion
  auto memoryAllocations = collectMemoryAllocations(m);

  performConversion(m, values);

  // Collect memory allocations after conversion
  MemoryAllocationsVec newMemoryAllocations = collectMemoryAllocations(m);
  // Adjust size of memory allocations using the info before and after conversion
  adjustSizeOfMemoryAllocations(m, memoryAllocations, newMemoryAllocations);

  closePhiLoops();
  cleanup(values);

  convertIndirectCalls(m);

  cleanUpOpenCLKernelTrampolines(&m);
  cleanUpOriginalFunctions(m);

  taffoInfo.dumpToFile("taffo_info_conv.json", m);
  LLVM_DEBUG(log().logln("[End of ConversionPass]", Logger::Magenta));
  return PreservedAnalyses::none();
}

void ConversionPass::createConversionQueue(std::vector<Value*>& values) {
  Logger& logger = log();
  LLVM_DEBUG(logger.logln("[Creating conversion queue]", Logger::Blue));
  size_t current = 0;
  while (current < values.size()) {
    Value* value = values.at(current);
    auto indenter = logger.getIndenter();
    LLVM_DEBUG(
      logger.log("[Value] ", Logger::Bold).logValueln(value);
      indenter.increaseIndent());

    SmallPtrSet<Value*, 8> roots;
    for (Value* oldRoot : getConversionInfo(value)->roots)
      if (getConversionInfo(oldRoot)->roots.empty())
        roots.insert(oldRoot);
    getConversionInfo(value)->roots.clear();
    getConversionInfo(value)->roots.insert(roots.begin(), roots.end());
    if (roots.empty())
      roots.insert(value);

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

      if (!hasConversionInfo(user)) {
        LLVM_DEBUG(logger.logln("value will not be converted because it has no conversionInfo", Logger::Yellow));
        auto valueConversionInfo = newConversionInfo(user);
        valueConversionInfo->isConversionDisabled = true;
        valueConversionInfo->origType = taffoInfo.getOrCreateTransparentType(*user)->clone();
      }

      if (auto* phi = dyn_cast<PHINode>(user))
        openPhiLoop(phi);
      getConversionInfo(user)->roots.insert(roots.begin(), roots.end());
    }
    current++;
  }
  LLVM_DEBUG(logger.logln("[Conversion queue created]", Logger::Blue));

  for (Value* value : values) {
    assert(hasConversionInfo(value) && "all values in the queue should have conversionInfo by now");
    if (getFixpType(value)->isInvalid() && taffoInfo.getTransparentType(*value)->containsFloatingPointType()
        && !isKnownConvertibleWithIncompleteMetadata(value)) {
      LLVM_DEBUG(
        logger.log("[Value] ", Logger::Bold).logValueln(value);
        auto indenter = logger.getIndenter();
        indenter.increaseIndent();
        logger.logln("value will not be converted because its conversionInfo is incomplete", Logger::Yellow));
      getConversionInfo(value)->isConversionDisabled = true;
    }

    SmallPtrSetImpl<Value*>& roots = getConversionInfo(value)->roots;
    if (roots.empty()) {
      getConversionInfo(value)->isRoot = true;
      if (isa<Instruction>(value) && !isa<AllocaInst>(value)) {
        // Non-alloca roots must have been generated by backtracking
        getConversionInfo(value)->isBacktrackingNode = true;
      }
      roots.insert(value);
    }
  }
}

void ConversionPass::propagateCalls(std::vector<Value*>& values, SmallVectorImpl<Value*>& global, Module& m) {
  SmallPtrSet<Function*, 16> oldFunctions;

  for (size_t i = 0; i < values.size(); i++) {
    Value* value = values[i];
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

    LLVM_DEBUG(log() << "Converting function " << oldF->getName() << " : " << *oldF->getType() << " into "
                     << newF->getName() << " : " << *newF->getType() << "\n");

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
        taffoInfo.setTransparentType(*newValue, taffoInfo.getTransparentType(*oldValue));
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
    LLVM_DEBUG(log() << "After CloneFunctionInto, the function now looks like this:\n"
                     << *newF->getFunctionType() << "\n");

    // propagate conversion
    std::vector<Value*> newValues;
    for (auto&& [oldArg, newArg] : zip(oldF->args(), newF->args())) {
      if (oldArg.getType() != newArg.getType()) {
        // append fixp info to arg name
        newArg.setName(newArg.getName() + "." + getFixpType(&oldArg)->toString());

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
        *newConversionInfo(placeholder) = *getConversionInfo(&oldArg);
        convertedValues[placeholder] = &newArg;

        getConversionInfo(placeholder)->isArgumentPlaceholder = true;
        newValues.push_back(placeholder);

        // Copy valueInfo to the placeholder because it's the only place where ranges are stored
        std::shared_ptr<ValueInfo> argInfo = taffoInfo.getValueInfo(oldArg);
        if (std::shared_ptr<ScalarInfo> argScalarInfo = std::dynamic_ptr_cast_or_null<ScalarInfo>(argInfo)) {
          std::shared_ptr<ValueInfo> newInfo = argScalarInfo->clone();
          taffoInfo.setTransparentType(*placeholder, taffoInfo.getOrCreateTransparentType(oldArg)->clone());
          taffoInfo.setValueInfo(*placeholder, newInfo);
        }
        // No need to mark the argument itself
        // buildLocalConversionInfo will do it in a bit as its metadata has been cloned as well
      }
    }

    newValues.insert(newValues.end(), global.begin(), global.end());
    SmallVector<Value*, 32> localFixed;
    buildLocalConversionInfo(*newF, localFixed);
    newValues.insert(newValues.end(), localFixed.begin(), localFixed.end());

    /* Make sure that the new arguments have correct conversionInfo */
    for (auto&& [oldArg, newArg] : zip(oldF->args(), newF->args())) {
      if (oldArg.getType() != newArg.getType())
        *getConversionInfo(&newArg) = *getConversionInfo(&oldArg);
      if (hasConversionInfo(&newArg)) {
        auto fixpType = getFixpType(&newArg);
        taffoInfo.setTransparentType(newArg, fixpType->toTransparentType(taffoInfo.getTransparentType(newArg)));
      }
    }
    // Copy the return type on the call instruction to all the return instructions
    for (ReturnInst* v : returns) {
      if (!hasConversionInfo(call))
        continue;
      newValues.push_back(v);
      demandConversionInfo(v)->fixpType = getFixpType(call);
      getConversionInfo(v)->origType = taffoInfo.getOrCreateTransparentType(*v)->clone();
      getConversionInfo(v)->fixpTypeRootDistance = 0;
    }

    LLVM_DEBUG(log() << "creating conversion queue of new function " << newF->getName() << "\n");
    createConversionQueue(newValues);

    oldFunctions.insert(oldF);

    // Put the instructions from the new function in queue
    for (Value* newValue : newValues)
      if (auto* inst = dyn_cast<Instruction>(newValue))
        if (inst->getFunction() == newF && !is_contained(values, inst))
          values.push_back(newValue);
  }

  // Remove instructions of the old function from the queue
  size_t i, j;
  for (i = 0, j = 0; j < values.size(); j++) {
    values[i] = values[j];
    Value* value = values[j];
    bool toDelete = false;
    if (auto* inst = dyn_cast<Instruction>(value)) {
      if (oldFunctions.contains(inst->getFunction())) {
        toDelete = true;
        if (auto* phi = dyn_cast_or_null<PHINode>(inst))
          phiReplacementData.erase(phi);
      }
    }
    else if (auto* arg = dyn_cast<Argument>(value)) {
      if (oldFunctions.contains(arg->getParent()))
        toDelete = true;
    }
    if (!toDelete)
      i++;
  }
  values.resize(i);
}

Function* ConversionPass::createConvertedFunctionForCall(CallBase* call, bool* alreadyHandledNewF) {
  Function* oldF = call->getCalledFunction();

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger.logln("[Creating converted function]", Logger::Blue);
    indenter.increaseIndent();
    logger.log("original function: ").logValueln(oldF););

  if (isSpecialFunction(oldF)) {
    LLVM_DEBUG(logger << "special function: ignoring\n");
    return nullptr;
  }
  if (!taffoInfo.isTaffoCloneFunction(*oldF)) {
    LLVM_DEBUG(logger << "not a function clone: ignoring\n");
    return nullptr;
  }

  std::vector<Type*> argLLVMTypes;
  // To match already converted function
  std::vector<std::pair<int, std::shared_ptr<FixedPointType>>> argFixedPointTypes;

  std::string suffix;
  if (getFullyUnwrappedType(oldF)->isFloatingPointTy()) {
    std::shared_ptr<FixedPointType> retValType = getFixpType(call);
    suffix = retValType->toString();
    argFixedPointTypes.push_back({-1, retValType});
  }
  else
    suffix = "fixp";

  int i = 0;
  for (auto arg = oldF->arg_begin(); arg != oldF->arg_end(); arg++, i++) {
    Type* newType;
    if (hasConversionInfo(arg)) {
      argFixedPointTypes.push_back({i, getFixpType(arg)});
      newType = getLLVMFixedPointTypeForFloatValue(arg);
    }
    else
      newType = arg->getType();
    argLLVMTypes.push_back(newType);
  }

  Function* newF = functionPool[oldF];
  if (newF) {
    LLVM_DEBUG(logger << "converted function already exists: " << newF->getName() << " " << *newF->getType() << "\n");
    if (alreadyHandledNewF)
      *alreadyHandledNewF = true;
    return newF;
  }
  if (alreadyHandledNewF)
    *alreadyHandledNewF = false;

  auto oldRetType = taffoInfo.getTransparentType(*oldF);
  auto newRetType = oldRetType;
  if (hasConversionInfo(call))
    if (!getConversionInfo(call)->isConversionDisabled && !getFixpType(call)->isInvalid())
      newRetType = getFixpType(call)->toTransparentType(taffoInfo.getTransparentType(*call));

  FunctionType* newFunType = FunctionType::get(newRetType->toLLVMType(), argLLVMTypes, oldF->isVarArg());

  LLVM_DEBUG(
    logger << "creating function " << oldF->getName() << "_" << suffix << " with types ";
    for (auto [argIndex, fixedPointType] : argFixedPointTypes)
      logger << "(" << argIndex << ", " << *fixedPointType << ") ";
    logger << "\n";);

  newF = Function::Create(newFunType, oldF->getLinkage(), oldF->getName() + "_" + suffix, oldF->getParent());
  taffoInfo.setTransparentType(*newF, newRetType);
  LLVM_DEBUG(logger << "created function\n"
                    << *newF << "\n");
  functionPool[oldF] = newF;
  FunctionCreated++;
  return newF;
}

void ConversionPass::openPhiLoop(PHINode* phi) {
  PHIInfo info;

  if (phi->materialized_use_empty()) {
    LLVM_DEBUG(log() << "phi" << *phi << " not currently used by anything; skipping placeholder creation\n");
    return;
  }

  auto type = taffoInfo.getTransparentType(*phi);

  info.placeh_noconv = createPlaceholder(phi->getType(), phi->getParent(), "phi_noconv");
  *(newConversionInfo(info.placeh_noconv)) = *(getConversionInfo(phi));
  phi->replaceAllUsesWith(info.placeh_noconv);
  copyValueInfo(info.placeh_noconv, phi, type);
  if (isFloatingPointToConvert(phi)) {
    auto newType = getFixpType(phi)->toTransparentType(type);
    info.placeh_conv = createPlaceholder(newType->toLLVMType(), phi->getParent(), "phi_conv");
    *newConversionInfo(info.placeh_conv) = *getConversionInfo(phi);
    copyValueInfo(info.placeh_conv, phi, newType);
  }
  else {
    info.placeh_conv = info.placeh_noconv;
  }
  convertedValues[info.placeh_noconv] = info.placeh_conv;

  LLVM_DEBUG(log() << "created placeholder (non-converted=[" << *info.placeh_noconv << "], converted=["
                   << *info.placeh_conv << "]) for phi " << *phi << "\n");

  phiReplacementData[phi] = info;
}

void ConversionPass::closePhiLoops() {
  LLVM_DEBUG(log() << __PRETTY_FUNCTION__ << " begin\n");

  for (auto data : phiReplacementData) {
    PHINode* origphi = data.first;
    PHIInfo& info = data.second;
    Value* substphi = convertedValues.at(origphi);

    LLVM_DEBUG(log() << "restoring data flow of phi " << *origphi << "\n");
    if (info.placeh_noconv != info.placeh_conv)
      info.placeh_noconv->replaceAllUsesWith(origphi);
    if (!substphi) {
      LLVM_DEBUG(log() << "phi " << *origphi << "could not be converted! Trying last resort conversion\n");
      substphi = translateOrMatchAnyOperandAndType(origphi, getFixpType(origphi));
      assert(substphi && "phi conversion has failed");
    }

    info.placeh_conv->replaceAllUsesWith(substphi);
    LLVM_DEBUG(log() << "restored data flow of original phi " << *origphi << " to new value " << *substphi << "\n");
  }

  LLVM_DEBUG(log() << __PRETTY_FUNCTION__ << " end\n");
}

void ConversionPass::printConversionQueue(const std::vector<Value*>& queue) {
  Logger& logger = log();
  if (queue.size() > 1000) {
    logger << "Not printing the conversion queue because it exceeds 1000 items\n";
    return;
  }
  logger.logln("[Conversion queue]", Logger::Blue);
  for (Value* value : queue) {
    auto conversionInfo = getConversionInfo(value);
    logger.log("[Value] ", Logger::Bold).logValueln(value);
    auto indenter = logger.getIndenter();
    indenter.increaseIndent();
    logger.log("conversionInfo: ").logln(*conversionInfo, Logger::Cyan);
  }
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
    if (getConversionInfo(value)->isRoot == true)
      roots.push_back(value);

  DenseMap<Value*, bool> isRootOk;
  for (Value* root : roots)
    isRootOk[root] = true;

  for (Value* value : queue) {
    Value* cqi = convertedValues.at(value);
    assert(cqi && "every value should have been processed at this point!!");
    if (cqi == ConversionError) {
      if (!potentiallyUsesMemory(value))
        continue;
      LLVM_DEBUG(
        value->print(errs());
        if (auto* inst = dyn_cast<Instruction>(value))
          errs() << " in function " << inst->getFunction()->getName();
        errs() << " not converted; invalidates roots ");
      const auto& rootsAffected = getConversionInfo(value)->roots;
      for (Value* root : rootsAffected) {
        isRootOk[root] = false;
        LLVM_DEBUG(root->print(errs()));
      }
      LLVM_DEBUG(errs() << '\n');
    }
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
      const auto& roots = getConversionInfo(value)->roots;

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
  phiReplacementData.clear();
  clear(isa<PHINode>);

  for (Instruction* inst : toErase)
    taffoInfo.eraseValue(inst);
}

void ConversionPass::cleanUpOriginalFunctions(Module& m) {
  for (Function& f : m)
    if (taffoInfo.isOriginalFunction(f))
      f.setLinkage(taffoInfo.getOriginalFunctionLinkage(f));
}

ConversionPass::MemoryAllocationsVec ConversionPass::collectMemoryAllocations(Module& m) {
  Logger& logger = log();
  LLVM_DEBUG(logger.logln("[Collecting memory allocations]", Logger::Blue));
  std::vector<std::string> names {"malloc"};
  MemoryAllocationsVec memoryAllocations;

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

    // Iterate over function users (actual memory allocations, e.g. call to malloc)
    for (auto* user : fun->users())
      if (auto* inst = dyn_cast<Instruction>(user)) {
        if (taffoInfo.hasTransparentType(*inst)) {
          std::shared_ptr<TransparentType> type = taffoInfo.getTransparentType(*inst);
          auto indenter = logger.getIndenter();
          LLVM_DEBUG(
            logger.log("[Value] ", Logger::Bold).logValueln(inst);
            indenter.increaseIndent();
            logger.log("type: ").logln(*type, Logger::Cyan));
          memoryAllocations.push_back({inst, type});
        }
      }
  }
  return memoryAllocations;
}

void ConversionPass::adjustSizeOfMemoryAllocations(Module& m,
                                                   const MemoryAllocationsVec& oldMemoryAllocations,
                                                   const MemoryAllocationsVec& newMemoryAllocations) {
  IRBuilder builder(m.getContext());

  Logger& logger = log();
  LLVM_DEBUG(logger.logln("[Adjusting size of memory allocations]", Logger::Blue));

  for (const auto& [oldUser, oldType] : oldMemoryAllocations)
    for (auto& [newUser, newType] : newMemoryAllocations)
      if (newUser == convertedValues[oldUser]) {
        auto indenter = logger.getIndenter();
        LLVM_DEBUG(
          logger.log("[Value] ", Logger::Bold).logValueln(oldUser);
          indenter.increaseIndent(););
        Value* oldSizeValue = oldUser->getOperand(0);
        Value* newSizeValue = adjustMemoryAllocationSize(
          oldSizeValue, oldType->getPointedType(), newType->getPointedType(), dyn_cast<Instruction>(oldUser));
        newUser->setOperand(0, newSizeValue);
      }
}

Value* ConversionPass::adjustMemoryAllocationSize(Value* oldSizeValue,
                                                  const std::shared_ptr<TransparentType>& oldAllocatedType,
                                                  const std::shared_ptr<TransparentType>& newAllocatedType,
                                                  Instruction* insertionPoint) {
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
