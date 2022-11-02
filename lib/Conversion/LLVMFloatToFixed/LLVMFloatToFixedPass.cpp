#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"
#include "Metadata.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/IR/InstrTypes.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>


using namespace llvm;
using namespace flttofix;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"


PreservedAnalyses Conversion::run(Module &M, ModuleAnalysisManager &AM)
{
  FloatToFixed Impl;
  return Impl.run(M, AM);
}


using MLHVec = std::vector<std::pair<llvm::User *, llvm::Type *>>;

MLHVec collectMallocLikeHandler(Module &m)
{
  LLVM_DEBUG(llvm::dbgs() << "#### " << __func__ << " BEGIN ####\n");
  std::vector<std::string> names{"malloc"};
  MLHVec tmp;

  for (const auto &name : names) {
    LLVM_DEBUG(llvm::dbgs() << "Searching " << name << " as ");
    // Mangle name to find the function
    std::string mangledName;
    llvm::raw_string_ostream mangledNameStream(mangledName);
    llvm::Mangler::getNameWithPrefix(mangledNameStream, name, m.getDataLayout());
    mangledNameStream.flush();
    LLVM_DEBUG(llvm::dbgs() << mangledName << "\n");

    // Search function
    auto fun = m.getFunction(mangledName);
    if (fun == nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "Not Found\n"
                              << "\n");
      continue;
    }

    // cycles Users of the function
    for (auto UF : fun->users()) {
      // cycles Users of return value of the function
      llvm::Type *type = nullptr;
      for (auto UC : UF->users()) {
        if (auto bitcast = llvm::dyn_cast<llvm::BitCastInst>(UC)) {
          LLVM_DEBUG(llvm::dbgs() << "Found bitcast from ");
          LLVM_DEBUG(UF->dump());
          LLVM_DEBUG(llvm::dbgs() << "to ");
          LLVM_DEBUG(bitcast->dump());
          if (type == nullptr) {

            type = bitcast->getType()->isPtrOrPtrVectorTy() ? bitcast->getType()->getPointerElementType() : bitcast->getType()->getScalarType();
            while (type->isPtrOrPtrVectorTy()) {
              type = type->getPointerElementType();
              LLVM_DEBUG(llvm::dbgs() << "type " << *type << "\n");
            }
            LLVM_DEBUG(llvm::dbgs() << "Scalar type ");
            LLVM_DEBUG(type->dump());
          } else {
            llvm::Type *type_tmp = nullptr;
            type_tmp = bitcast->getType()->isPtrOrPtrVectorTy() ? bitcast->getType()->getPointerElementType() : bitcast->getType()->getScalarType();
            while (type_tmp->isPtrOrPtrVectorTy()) {
              type_tmp = type_tmp->getPointerElementType();
            }
            LLVM_DEBUG(llvm::dbgs() << "Scalar type ");
            LLVM_DEBUG(type_tmp->dump());
            if (type->getScalarSizeInBits() < type_tmp->getScalarSizeInBits()) {
              type = type_tmp;
            }
          }
        }
      }
      LLVM_DEBUG(dbgs() << "added user " << *UF << "type ptr " << type << "\n");
      tmp.push_back({UF, type});
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "#### " << __func__ << " END ####\n");
  return tmp;
}


Value *flttofix::adjustBufferSize(Value *OrigSize, Type *OldTy, Type *NewTy, Instruction *IP, bool Tight)
{
  assert(IP && "adjustBufferSize requires a valid insertion pointer. Somebody must use this buffer size after all, right?");

  llvm::Type *RootOldTy = fullyUnwrapPointerOrArrayType(OldTy);
  llvm::Type *RootNewTy = fullyUnwrapPointerOrArrayType(NewTy);
  LLVM_DEBUG(dbgs() << "Adjusting buffer size " << OrigSize->getNameOrAsOperand() << ", type change from " << *OldTy << " to " << *NewTy << "\n");

  if (!Tight && RootOldTy->getScalarSizeInBits() >= RootNewTy->getScalarSizeInBits()) {
    LLVM_DEBUG(dbgs() << "Old type is larger or same size than new type, doing nothing\n");
    return OrigSize;
  }
  if (Tight)
    LLVM_DEBUG(dbgs() << "Tight flag is set, adjusting size even if it gets reduced\n");
  else
    LLVM_DEBUG(dbgs() << "Old type is smaller than new type, adjusting arguments\n");

  unsigned Num = RootNewTy->getScalarSizeInBits();
  unsigned Den = RootOldTy->getScalarSizeInBits();
  LLVM_DEBUG(dbgs() << "Ratio: " << Num << " / " << Den << "\n");

  ConstantInt *int_const = dyn_cast<ConstantInt>(OrigSize);
  Value *Res;
  if (int_const == nullptr) {
    IRBuilder<> builder(IP);
    Res = builder.CreateMul(OrigSize, ConstantInt::get(OrigSize->getType(), Num));
    Res = builder.CreateAdd(Res, ConstantInt::get(OrigSize->getType(), Den-1));
    Res = builder.CreateUDiv(Res, ConstantInt::get(OrigSize->getType(), Den));
  } else {
    Res = ConstantInt::get(OrigSize->getType(), (int_const->getUniqueInteger() * Num + (Den-1)).udiv(Den));
  }

  LLVM_DEBUG(dbgs() << "Buffer size adjusted to " << *Res << "\n");
  return Res;
}


void closeMallocLikeHandler(Module &m, const MLHVec &vec)
{
  LLVM_DEBUG(llvm::dbgs() << "#### " << __func__ << " BEGIN ####\n");
  auto tmp = collectMallocLikeHandler(m);
  llvm::IRBuilder<> builder(m.getContext());

  for (auto &V : vec) {
    for (auto &T : tmp) {
      if (V.first == T.first) {
        LLVM_DEBUG(dbgs() << "Processing malloc " << *(V.first) << "...\n");
        if (V.second == T.second && V.second == nullptr) {
          LLVM_DEBUG(llvm::dbgs() << " Both types are null? ok...\n");
          continue;
        }
        Value *OldBufSize = V.first->getOperand(0);
        Value *NewBufSize = adjustBufferSize(OldBufSize, V.second, T.second, dyn_cast<llvm::Instruction>(V.first));
        if (NewBufSize != OldBufSize) {
          V.first->setOperand(0, NewBufSize);
          LLVM_DEBUG(dbgs() << "Converted malloc transformed to " << *(V.first) << "\n");
        } else {
          LLVM_DEBUG(dbgs() << "Buffer size did not change; the malloc stays as it is\n");
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "#### " << __func__ << " END ####\n");
}


PreservedAnalyses FloatToFixed::run(Module &m, ModuleAnalysisManager &AM)
{
  MAM = &AM;
  ModuleDL = &(m.getDataLayout());

  SmallPtrSet<Value *, 32> local;
  SmallPtrSet<Value *, 32> global;
  readAllLocalMetadata(m, local);
  readGlobalMetadata(m, global);

  std::vector<Value *> vals(local.begin(), local.end());
  vals.insert(vals.begin(), global.begin(), global.end());
  MetadataCount = vals.size();

  sortQueue(vals);
  propagateCall(vals, global, m);
  LLVM_DEBUG(printConversionQueue(vals));
  ConversionCount = vals.size();

  auto mallocLikevec = collectMallocLikeHandler(m);
  performConversion(m, vals);
  closeMallocLikeHandler(m, mallocLikevec);
  closePhiLoops();
  cleanup(vals);

  convertIndirectCalls(m);

  cleanUpOpenCLKernelTrampolines(&m);

  return PreservedAnalyses::none();
}


int FloatToFixed::getLoopNestingLevelOfValue(Value *v)
{
  Instruction *inst = dyn_cast<Instruction>(v);
  if (!inst)
    return 0;

  Function *fun = inst->getFunction();
  FunctionAnalysisManager &FAM = MAM->getResult<FunctionAnalysisManagerModuleProxy>(*(fun->getParent())).getManager();
  LoopInfo &li = FAM.getResult<LoopAnalysis>(*fun);
  BasicBlock *bb = inst->getParent();
  return li.getLoopDepth(bb);
}


void FloatToFixed::openPhiLoop(PHINode *phi)
{
  PHIInfo info;

  if (phi->materialized_use_empty()) {
    LLVM_DEBUG(dbgs() << "phi" << *phi << " not currently used by anything; skipping placeholder creation\n");
    return;
  }

  info.placeh_noconv = createPlaceholder(phi->getType(), phi->getParent(), "phi_noconv");
  *(newValueInfo(info.placeh_noconv)) = *(valueInfo(phi));
  phi->replaceAllUsesWith(info.placeh_noconv);
  cpMetaData(info.placeh_noconv, phi);
  if (isFloatingPointToConvert(phi)) {
    Type *convt = getLLVMFixedPointTypeForFloatType(phi->getType(), fixPType(phi));
    info.placeh_conv = createPlaceholder(convt, phi->getParent(), "phi_conv");
    *(newValueInfo(info.placeh_conv)) = *(valueInfo(phi));
    cpMetaData(info.placeh_conv, phi);
  } else {
    info.placeh_conv = info.placeh_noconv;
  }
  operandPool[info.placeh_noconv] = info.placeh_conv;

  LLVM_DEBUG(dbgs() << "created placeholder (non-converted=[" << *info.placeh_noconv << "], converted=["
                    << *info.placeh_conv << "]) for phi " << *phi << "\n");

  phiReplacementData[phi] = info;
}


void FloatToFixed::closePhiLoops()
{
  LLVM_DEBUG(dbgs() << __PRETTY_FUNCTION__ << " begin\n");

  for (auto data : phiReplacementData) {
    PHINode *origphi = data.first;
    PHIInfo &info = data.second;
    Value *substphi = operandPool[origphi];

    LLVM_DEBUG(dbgs() << "restoring data flow of phi " << *origphi << "\n");
    if (info.placeh_noconv != info.placeh_conv)
      info.placeh_noconv->replaceAllUsesWith(origphi);
    if (!substphi) {
      LLVM_DEBUG(dbgs() << "phi " << *origphi << "could not be converted! Trying last resort conversion\n");
      substphi = translateOrMatchAnyOperandAndType(origphi, fixPType(origphi));
      assert(substphi && "phi conversion has failed");
    }

    info.placeh_conv->replaceAllUsesWith(substphi);
    LLVM_DEBUG(
        dbgs() << "restored data flow of original phi " << *origphi << " to new value " << *substphi << "\n");
  }

  LLVM_DEBUG(dbgs() << __PRETTY_FUNCTION__ << " end\n");
}


bool FloatToFixed::isKnownConvertibleWithIncompleteMetadata(Value *V)
{
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    CallBase *Call = dyn_cast<CallBase>(I);
    if (!Call)
      return false;
    Function *F = Call->getCalledFunction();
    if (isSupportedOpenCLFunction(F))
      return true;
  }
  return false;
}


void FloatToFixed::sortQueue(std::vector<Value *> &vals)
{
  size_t next = 0;
  while (next < vals.size()) {
    Value *v = vals.at(next);
    LLVM_DEBUG(dbgs() << "[V] " << *v << "\n");
    SmallPtrSet<Value *, 5> roots;
    for (Value *oldroot : valueInfo(v)->roots) {
      if (valueInfo(oldroot)->roots.empty())
        roots.insert(oldroot);
    }
    valueInfo(v)->roots.clear();
    valueInfo(v)->roots.insert(roots.begin(), roots.end());
    if (roots.empty()) {
      roots.insert(v);
    }

    if (PHINode *phi = dyn_cast<PHINode>(v))
      openPhiLoop(phi);

    for (auto *u : v->users()) {
      if (Instruction *i = dyn_cast<Instruction>(u)) {
        if (functionPool.find(i->getFunction()) != functionPool.end()) {
          LLVM_DEBUG(dbgs() << "old function: skipped " << *u << "\n");
          continue;
        }
      }

      /* Insert u at the end of the queue.
       * If u exists already in the queue, *move* it to the end instead. */
      for (size_t i = 0; i < vals.size();) {
        if (vals[i] == u) {
          vals.erase(vals.begin() + i);
          if (i < next)
            next--;
        } else {
          i++;
        }
      }

      if (!hasInfo(u)) {
        LLVM_DEBUG(dbgs() << "[WARNING] Value " << *u << " will not be converted because it has no metadata\n");
        newValueInfo(u)->noTypeConversion = true;
        valueInfo(u)->origType = u->getType();
      }

      LLVM_DEBUG(dbgs() << "[U] " << *u << "\n");
      vals.push_back(u);
      if (PHINode *phi = dyn_cast<PHINode>(u))
        openPhiLoop(phi);
      valueInfo(u)->roots.insert(roots.begin(), roots.end());
    }
    next++;
  }

  for (Value *v : vals) {
    assert(hasInfo(v) && "all values in the queue should have a valueInfo by now");
    if (fixPType(v).isInvalid() && !(v->getType()->isVoidTy() && !isa<ReturnInst>(v)) && !isKnownConvertibleWithIncompleteMetadata(v)) {
      LLVM_DEBUG(dbgs() << "[WARNING] Value " << *v
                        << " will not be converted because its metadata is incomplete\n");
      LLVM_DEBUG(dbgs() << " (Apparent type of the value: " << fixPType(v).toString() << ")\n");
      valueInfo(v)->noTypeConversion = true;
    }

    SmallPtrSetImpl<Value *> &roots = valueInfo(v)->roots;
    if (roots.empty()) {
      valueInfo(v)->isRoot = true;
      if (isa<Instruction>(v) && !isa<AllocaInst>(v)) {
        /* non-alloca roots must have been generated by backtracking */
        valueInfo(v)->isBacktrackingNode = true;
      }
      roots.insert(v);
    }
  }
}


bool potentiallyUsesMemory(Value *val)
{
  if (!isa<Instruction>(val))
    return false;
  if (isa<BitCastInst>(val))
    return false;
  if (CallInst *call = dyn_cast<CallInst>(val)) {
    Function *f = call->getCalledFunction();
    if (!f)
      return true;
    if (f->isIntrinsic()) {
      Intrinsic::ID fiid = f->getIntrinsicID();
      if (fiid == Intrinsic::lifetime_start ||
          fiid == Intrinsic::lifetime_end)
        return false;
    }
    return !f->doesNotAccessMemory();
  }
  return true;
}


void FloatToFixed::cleanup(const std::vector<Value *> &q)
{
  std::vector<Value *> roots;
  for (Value *v : q) {
    if (valueInfo(v)->isRoot == true)
      roots.push_back(v);
  }

  DenseMap<Value *, bool> isrootok;
  for (Value *root : roots)
    isrootok[root] = true;

  for (Value *qi : q) {
    Value *cqi = operandPool[qi];
    assert(cqi && "every value should have been processed at this point!!");
    if (cqi == ConversionError) {
      if (!potentiallyUsesMemory(qi)) {
        continue;
      }
      LLVM_DEBUG(qi->print(errs());
                 if (Instruction *i = dyn_cast<Instruction>(qi))
                     errs()
                 << " in function " << i->getFunction()->getName();
                 errs() << " not converted; invalidates roots ");
      const auto &rootsaffected = valueInfo(qi)->roots;
      for (Value *root : rootsaffected) {
        isrootok[root] = false;
        LLVM_DEBUG(root->print(errs()));
      }
      LLVM_DEBUG(errs() << '\n');
    }
  }

  std::vector<Instruction *> toErase;

  auto clear = [&](bool (*toDelete)(const Instruction &Y)) {
    for (Value *v : q) {
      Instruction *i = dyn_cast<Instruction>(v);
      if (!i || (!toDelete(*i)))
        continue;
      if (operandPool[v] == v) {
        LLVM_DEBUG(dbgs() << *i << " not deleted, as it was converted by self-mutation\n");
        continue;
      }
      const auto &roots = valueInfo(v)->roots;

      bool allok = true;
      for (Value *root : roots) {
        if (!isrootok[root]) {
          LLVM_DEBUG(i->print(errs());
                     errs() << " not deleted: involves root ";
                     root->print(errs());
                     errs() << '\n');
          allok = false;
          break;
        }
      }
      if (allok) {
        if (!i->use_empty())
          i->replaceAllUsesWith(UndefValue::get(i->getType()));
        toErase.push_back(i);
      }
    }
  };

  clear(isa<StoreInst>);

  /* remove calls manually because DCE does not do it as they may have
   * side effects */
  clear(isa<CallInst>);
  clear(isa<InvokeInst>);

  clear(isa<BranchInst>);

  /* remove old phis manually as DCE cannot remove values having a circular
   * dependence on a phi */
  phiReplacementData.clear();
  clear(isa<PHINode>);

  for (Instruction *v : toErase) {
    v->eraseFromParent();
  }
}


void FloatToFixed::propagateCall(std::vector<Value *> &vals, llvm::SmallPtrSetImpl<llvm::Value *> &global, Module &m)
{
  mdutils::MetadataManager &MM = mdutils::MetadataManager::getMetadataManager();
  SmallPtrSet<Function *, 16> oldFuncs;

  for (size_t i = 0; i < vals.size(); i++) {
    Value *valsi = vals[i];
    CallBase *call = dyn_cast<CallBase>(valsi);

    if (call == nullptr)
      continue;

    bool alreadyHandledNewF;
    Function *oldF = call->getCalledFunction();
    Function *newF = createFixFun(call, &alreadyHandledNewF);
    if (!newF) {
      LLVM_DEBUG(dbgs() << "Attempted to clone function " << oldF->getName() << " but failed\n");
      continue;
    }
    if (alreadyHandledNewF) {
      oldFuncs.insert(oldF);
      continue;
    }

    LLVM_DEBUG(dbgs() << "Converting function " << oldF->getName() << " : " << *oldF->getType()
                      << " into " << newF->getName() << " : " << *newF->getType() << "\n");

    ValueToValueMapTy origValToCloned; // Create Val2Val mapping and clone function
    Function::arg_iterator newIt = newF->arg_begin();
    Function::arg_iterator oldIt = oldF->arg_begin();
    for (; oldIt != oldF->arg_end(); oldIt++, newIt++) {
      newIt->setName(oldIt->getName());
      origValToCloned.insert(std::make_pair(oldIt, newIt));
    }
    SmallVector<ReturnInst *, 100> returns;
    CloneFunctionInto(newF, oldF, origValToCloned, CloneFunctionChangeType::GlobalChanges, returns);
    /* after CloneFunctionInto, valueMap maps all values from the oldF to the newF (not just the arguments) */

    /* CloneFunctionInto also fixes the attributes of the arguments.
     * This is not exactly what we want for OpenCL kernels because the alignment
     * after the conversion is not defined by us but by the OpenCL runtime.
     * So we need to compensate for this. */
    if (newF->getCallingConv() == CallingConv::SPIR_KERNEL || mdutils::MetadataManager::isCudaKernel(m, oldF)) {
      /* OpenCL spec says the alignment is equal to the size of the type */
      SmallVector<AttributeSet, 4> NewAttrs(newF->arg_size());
      AttributeList OldAttrs = newF->getAttributes();
      for (unsigned ArgId = 0; ArgId < newF->arg_size(); ArgId++) {
        Argument *Arg = newF->getArg(ArgId);
        if (!Arg->getType()->isPointerTy())
          continue;
        Type *ArgTy = fullyUnwrapPointerOrArrayType(Arg->getType());
        Align align(ArgTy->getScalarSizeInBits() / 8);
        AttributeSet OldArgAttrs = OldAttrs.getParamAttrs(ArgId);
        AttributeSet NewArgAttrs = OldArgAttrs.addAttributes(newF->getContext(), AttributeSet::get(newF->getContext(), {Attribute::getWithAlignment(newF->getContext(), align)}));
        NewAttrs[ArgId] = NewArgAttrs;
        LLVM_DEBUG(dbgs() << "Fixed align of arg " << ArgId << " (" << *Arg << ") to " << align.value() << "\n");
      }
      newF->setAttributes(AttributeList::get(newF->getContext(), OldAttrs.getFnAttrs(), OldAttrs.getRetAttrs(), NewAttrs));
      LLVM_DEBUG(dbgs() << "Set new attributes, hopefully without breaking anything\n");
    }
    LLVM_DEBUG(dbgs() << "After CloneFunctionInto, the function now looks like this:\n" << *newF << "\n");

    SmallVector<mdutils::MDInfo *, 4> ArgsII;
    MM.retrieveArgumentInputInfo(*oldF, ArgsII);

    std::vector<Value *> newVals; // propagate fixp conversion
    oldIt = oldF->arg_begin();
    newIt = newF->arg_begin();
    for (int i = 0; oldIt != oldF->arg_end(); oldIt++, newIt++, i++) {
      if (oldIt->getType() != newIt->getType()) {
        FixedPointType fixtype = valueInfo(oldIt)->fixpType;

        // append fixp info to arg name
        newIt->setName(newIt->getName() + "." + fixtype.toString());

        /* Create a fake value to maintain type consistency because
         * createFixFun has RAUWed all arguments
         * FIXME: is there a cleaner way to do this? */
        std::string name("placeholder");
        if (newIt->hasName()) {
          name += "." + newIt->getName().str();
        }
        Value *placehValue = createPlaceholder(oldIt->getType(), &newF->getEntryBlock(), name);
        /* Reimplement RAUW to defeat the same-type check (which is ironic because
         * we are attempting to fix a type mismatch here) */
        while (!newIt->materialized_use_empty()) {
          Use &U = *(newIt->uses().begin());
          U.set(placehValue);
        }
        *(newValueInfo(placehValue)) = *(valueInfo(oldIt));
        operandPool[placehValue] = newIt;

        valueInfo(placehValue)->isArgumentPlaceholder = true;
        newVals.push_back(placehValue);

        /* Copy input info to the placeholder because it's the only place where+
         * ranges are stored */
        mdutils::InputInfo *II = dyn_cast_or_null<mdutils::InputInfo>(ArgsII[i]);
        if (II) {
          MM.setInputInfoMetadata(*(dyn_cast<Instruction>(placehValue)), *II);
        }

        /* No need to mark the argument itself, readLocalMetadata will
         * do it in a bit as its metadata has been cloned as well */
      }
    }

    newVals.insert(newVals.end(), global.begin(), global.end());
    SmallPtrSet<Value *, 32> localFix;
    readLocalMetadata(*newF, localFix);
    newVals.insert(newVals.end(), localFix.begin(), localFix.end());

    /* Make sure that the new arguments have correct ValueInfo */
    oldIt = oldF->arg_begin();
    newIt = newF->arg_begin();
    for (; oldIt != oldF->arg_end(); oldIt++, newIt++) {
      if (oldIt->getType() != newIt->getType()) {
        *(valueInfo(newIt)) = *(valueInfo(oldIt));
      }
    }

    /* Copy the return type on the call instruction to all the return
     * instructions */
    for (ReturnInst *v : returns) {
      if (!hasInfo(call))
        continue;
      newVals.push_back(v);
      demandValueInfo(v)->fixpType = valueInfo(call)->fixpType;
      valueInfo(v)->origType = nullptr;
      valueInfo(v)->fixpTypeRootDistance = 0;
    }

    LLVM_DEBUG(dbgs() << "Sorting queue of new function " << newF->getName() << "\n");
    sortQueue(newVals);

    oldFuncs.insert(oldF);

    /* Put the instructions from the new function in */
    for (Value *val : newVals) {
      if (Instruction *inst = dyn_cast<Instruction>(val)) {
        if (inst->getFunction() == newF) {
          vals.push_back(val);
        }
      }
    }
  }

  /* Remove instructions of the old functions from the queue */
  size_t removei, removej;
  for (removei = 0, removej = 0; removej < vals.size(); removej++) {
    vals[removei] = vals[removej];
    Value *val = vals[removej];
    bool toDelete = false;
    if (Instruction *inst = dyn_cast<Instruction>(val)) {
      if (oldFuncs.count(inst->getFunction())) {
        toDelete = true;
        if (PHINode *phi = dyn_cast_or_null<PHINode>(inst)) {
          phiReplacementData.erase(phi);
        }
      }
    } else if (Argument *arg = dyn_cast<Argument>(val)) {
      if (oldFuncs.count(arg->getParent()))
        toDelete = true;
    }
    if (!toDelete)
      removei++;
  }
  vals.resize(removei);
}


Function *FloatToFixed::createFixFun(CallBase *call, bool *old)
{
  LLVM_DEBUG(dbgs() << "*********** " << __FUNCTION__ << "\n");
  Function *oldF = call->getCalledFunction();
  assert(oldF && "bitcasted function pointers and such not handled atm");
  if (isSpecialFunction(oldF))
    return nullptr;

  if (!oldF->getMetadata(SOURCE_FUN_METADATA)) {
    LLVM_DEBUG(dbgs() << "createFixFun: function " << oldF->getName() << " not a clone; ignoring\n");
    return nullptr;
  }

  std::vector<Type *> typeArgs;
  std::vector<std::pair<int, FixedPointType>> fixArgs; // for match already converted function

  std::string suffix;
  if (isFloatType(oldF->getReturnType())) { // ret value in signature
    FixedPointType retValType = valueInfo(call)->fixpType;
    suffix = retValType.toString();
    fixArgs.push_back(std::pair<int, FixedPointType>(-1, retValType));
  } else {
    suffix = "fixp";
  }

  int i = 0;
  for (auto arg = oldF->arg_begin(); arg != oldF->arg_end(); arg++, i++) {
    Value *v = dyn_cast<Value>(arg);
    Type *newTy;
    if (hasInfo(v)) {
      fixArgs.push_back(std::pair<int, FixedPointType>(i, valueInfo(v)->fixpType));
      newTy = getLLVMFixedPointTypeForFloatValue(v);
    } else {
      newTy = v->getType();
    }
    typeArgs.push_back(newTy);
  }

  Function *newF = functionPool[oldF]; // check if is previously converted
  if (newF) {
    LLVM_DEBUG(dbgs() << *(call) << " use already converted function : " << newF->getName() << " " << *newF->getType() << "\n";);
    if (old)
      *old = true;
    return newF;
  }
  if (old)
    *old = false;

  Type *retType = oldF->getReturnType();
  if (hasInfo(call)) {
    if (!valueInfo(call)->noTypeConversion)
      retType = getLLVMFixedPointTypeForFloatValue(call);
  }

  FunctionType *newFunTy = FunctionType::get(
      isFloatType(oldF->getReturnType()) ? retType : oldF->getReturnType(),
      typeArgs, oldF->isVarArg());

  LLVM_DEBUG({
    dbgs() << "creating function " << oldF->getName() << "_" << suffix << " with types ";
    for (auto pair : fixArgs) {
      dbgs() << "(" << pair.first << ", " << pair.second << ") ";
    }
    dbgs() << "\n";
  });

  newF = Function::Create(newFunTy, oldF->getLinkage(), oldF->getName() + "_" + suffix, oldF->getParent());
  LLVM_DEBUG(dbgs() << "created function\n" << *newF << "\n");
  functionPool[oldF] = newF; // add to pool
  FunctionCreated++;
  return newF;
}


void FloatToFixed::printConversionQueue(std::vector<Value *> vals)
{
  if (vals.size() > 1000) {
    LLVM_DEBUG(dbgs() << "not printing the conversion queue because it exceeds 1000 items\n";);
    return;
  }

  LLVM_DEBUG(dbgs() << "conversion queue:\n";);
  for (Value *val : vals) {
    LLVM_DEBUG(dbgs() << "bt=" << valueInfo(val)->isBacktrackingNode << " ";);
    LLVM_DEBUG(dbgs() << "noconv=" << valueInfo(val)->noTypeConversion << " ";);
    LLVM_DEBUG(dbgs() << "type=" << valueInfo(val)->fixpType << " ";);
    if (Instruction *i = dyn_cast<Instruction>(val)) {
      LLVM_DEBUG(dbgs() << " fun='" << i->getFunction()->getName() << "' ";);
    }

    LLVM_DEBUG(dbgs() << "roots=[";);
    for (Value *rootv : valueInfo(val)->roots) {
      LLVM_DEBUG(dbgs() << *rootv << ", ";);
    }
    LLVM_DEBUG(dbgs() << "] ";);

    LLVM_DEBUG(dbgs() << *val << "\n";);
  }
  LLVM_DEBUG(dbgs() << "\n\n";);
}
