#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>
#include <cmath>

using namespace llvm;
using namespace flttofix;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"


/* also inserts the new value in the basic blocks, alongside the old one */
Value *FloatToFixed::convertInstruction(Module &m, Instruction *val,
                                        FixedPointType &fixpt)
{
  Value *res = Unsupported;
  if (AllocaInst *alloca = dyn_cast<AllocaInst>(val)) {
    res = convertAlloca(alloca, fixpt);
  } else if (LoadInst *load = dyn_cast<LoadInst>(val)) {
    res = convertLoad(load, fixpt);
  } else if (StoreInst *store = dyn_cast<StoreInst>(val)) {
    res = convertStore(store);
  } else if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(val)) {
    res = convertGep(gep, fixpt);
  } else if (ExtractValueInst *ev = dyn_cast<ExtractValueInst>(val)) {
    res = convertExtractValue(ev, fixpt);
  } else if (InsertValueInst *iv = dyn_cast<InsertValueInst>(val)) {
    res = convertInsertValue(iv, fixpt);
  } else if (PHINode *phi = dyn_cast<PHINode>(val)) {
    res = convertPhi(phi, fixpt);
  } else if (SelectInst *select = dyn_cast<SelectInst>(val)) {
    res = convertSelect(select, fixpt);
  } else if (isa<CallInst>(val) || isa<InvokeInst>(val)) {
    res = convertCall(dyn_cast<CallBase>(val), fixpt);
  } else if (ReturnInst *ret = dyn_cast<ReturnInst>(val)) {
    res = convertRet(ret, fixpt);
  } else if (Instruction *instr = dyn_cast<Instruction>(
                 val)) { // llvm/include/llvm/IR/Instruction.def for more info
    if (instr->isBinaryOp()) {
      res = convertBinOp(instr, fixpt);
    } else if (CastInst *cast = dyn_cast<CastInst>(instr)) {
      res = convertCast(cast, fixpt);
    } else if (FCmpInst *fcmp = dyn_cast<FCmpInst>(val)) {
      res = convertCmp(fcmp);
    } else if (instr->isUnaryOp()) {

      res = convertUnaryOp(instr, fixpt);
    }
  }
  if (res == Unsupported) {
    res = fallback(dyn_cast<Instruction>(val), fixpt);
  }
  if (res && res != Unsupported && !(res->getType()->isVoidTy()) &&
      !hasInfo(res)) {
    if (isFloatType(val->getType()) && !valueInfo(val)->noTypeConversion) {
      std::string tmpstore;
      raw_string_ostream tmp(tmpstore);
      if (res->hasName())
        tmp << res->getName().str() << ".";
      else if (val->hasName())
        tmp << val->getName().str() << ".";
      tmp << fixpt;
      res->setName(tmp.str());
    } else if (valueInfo(val)->noTypeConversion) {
      std::string tmpstore;
      raw_string_ostream tmp(tmpstore);
      if (res->hasName())
        tmp << res->getName().str() << ".";
      else if (val->hasName())
        tmp << val->getName().str() << ".";
      tmp << "matchop";
      res->setName(tmp.str());
    }
  }

  if (res) {
    if (!isa<Instruction>(res)) {
      LLVM_DEBUG(dbgs() << "Conversion produced something that is not an instruction from an instruction...\n");
    }
    return res;
  } else {
    return ConversionError;
  }
}


Value *FloatToFixed::convertAlloca(AllocaInst *alloca,
                                   const FixedPointType &fixpt)
{
  if (valueInfo(alloca)->noTypeConversion)
    return alloca;
  Type *prevt = alloca->getAllocatedType();
  Type *newt = getLLVMFixedPointTypeForFloatType(prevt, fixpt);
  if (newt == prevt)
    return alloca;
  Value *as = alloca->getArraySize();
  Align align = alloca->getAlign();
  AllocaInst *newinst = new AllocaInst(
      newt, alloca->getType()->getPointerAddressSpace(), as, align);
  newinst->setUsedWithInAlloca(alloca->isUsedWithInAlloca());
  newinst->setSwiftError(alloca->isSwiftError());
  newinst->insertAfter(alloca);
  return newinst;
}


Value *FloatToFixed::convertLoad(LoadInst *load, FixedPointType &fixpt)
{
  Value *ptr = load->getPointerOperand();
  Value *newptr = operandPool[ptr];
  if (newptr == ConversionError)
    return nullptr;
  if (!newptr)
    return Unsupported;
  if (isConvertedFixedPoint(newptr)) {
    fixpt = fixPType(newptr);
    Align align = load->getAlign();
    LoadInst *newinst =
        new LoadInst(newptr->getType()->getPointerElementType(), newptr, Twine(), load->isVolatile(), align,
                     load->getOrdering(), load->getSyncScopeID());
    newinst->insertAfter(load);
    if (valueInfo(load)->noTypeConversion) {
      assert(newinst->getType()->isIntegerTy() &&
             "DTA bug; improperly tagged struct/pointer!");
      return genConvertFixToFloat(newinst, fixPType(newptr), load->getType());
    }
    return newinst;
  }
  return Unsupported;
}


Value *FloatToFixed::convertStore(StoreInst *store)
{
  Value *ptr = store->getPointerOperand();
  Value *val = store->getValueOperand();
  Value *newptr = matchOp(ptr);
  if (!newptr)
    return nullptr;
  Value *newval;
  Type *peltype = newptr->getType()->getPointerElementType();
  if (isFloatingPointToConvert(val)) {
    /* value is converted (thus we can match it) */
    if (isConvertedFixedPoint(newptr)) {
      FixedPointType valtype = fixPType(newptr);
      if (peltype->isPointerTy()) {
        /* store <value ptr> into <value ptr> pointer; both are converted
         * so everything is fine and there is nothing special to do.
         * Only logging type mismatches because we trust DTA has done its job */
        newval = matchOp(val);
        if (!(fixPType(newval) == valtype))
          LLVM_DEBUG(
              dbgs()
              << "unsolvable fixp type mismatch between store dest and src!\n");
      } else {
        /* best case: store <value> into <value> pointer */
        valtype = fixPType(newptr);
        newval = translateOrMatchOperandAndType(val, valtype, store);
      }
    } else {
      /* store fixp <value ptr?> into original <value ptr?> pointer
       * try to match the stored value if possible */
      newval = fallbackMatchValue(val, peltype);
      if (!newval)
        return Unsupported;
    }
  } else {
    /* pointer is converted, but value is not converted */
    if (isConvertedFixedPoint(newptr)) {
      FixedPointType valtype = fixPType(newptr);
      /* the value to store is not converted but the pointer is */
      /*Checking for the value to be a pointer in order to assert that is a
       * converted type is not sufficient anymore because of destination
       * datatype can be a float too. This raises a bug when storing constants
       * (in other cases should be ok)*/
      if (peltype->isIntegerTy() || !peltype->isPointerTy()) {
        /* value is not a pointer; we can convert it to fixed point */
        newval = genConvertFloatToFix(val, valtype);
      } else {
        /* value unconverted ptr; dest is converted ptr
         * would be an error; remove this as soon as it is not needed anymore */
        LLVM_DEBUG(
            dbgs()
            << "[Store] HACK: bitcasting operands of wrong type to new type\n");
        BitCastInst *bc = new BitCastInst(val, peltype);
        cpMetaData(bc, val);
        bc->insertBefore(store);
        newval = bc;
      }
    } else {
      /* nothing is converted, just matchop */
      newval = matchOp(val);
    }
  }
  if (!newval)
    return nullptr;
  Align align = store->getAlign();
  StoreInst *newinst =
      new StoreInst(newval, newptr, store->isVolatile(), align,
                    store->getOrdering(), store->getSyncScopeID());
  newinst->insertAfter(store);
  return newinst;
}


Value *FloatToFixed::convertGep(GetElementPtrInst *gep, FixedPointType &fixpt)
{
  LLVM_DEBUG(llvm::dbgs() << "### Convert GEP ###\n");
  IRBuilder<NoFolder> builder(gep);
  Value *newval = matchOp(gep->getPointerOperand());

  LLVM_DEBUG(llvm::dbgs() << *gep << "\nhas operand \n"
                          << *(gep->getPointerOperand()) << "\nmatchOp return \n"
                          << *newval << "\n");
  if (!newval)
    return valueInfo(gep)->noTypeConversion ? Unsupported : nullptr;
  if (!isConvertedFixedPoint(newval)) {
    /* just replace the arguments, they should stay the same type */
    return Unsupported;
  }
  FixedPointType tempFixpt = fixPType(newval);
  Type *type = gep->getPointerOperand()->getType();
  fixpt = tempFixpt.unwrapIndexList(type, gep->indices());
  /* if conversion is disabled, we can extract values that didn't get a type
   * change, but we cannot extract values that didn't */
  if (valueInfo(gep)->noTypeConversion && !fixpt.isRecursivelyInvalid())
    return Unsupported;
  std::vector<Value *> idxlist(gep->indices().begin(), gep->indices().end());
  return builder.CreateInBoundsGEP(newval->getType()->getPointerElementType(), newval, idxlist);
}


Value *FloatToFixed::convertExtractValue(ExtractValueInst *exv,
                                         FixedPointType &fixpt)
{
  if (valueInfo(exv)->noTypeConversion)
    return Unsupported;
  IRBuilder<NoFolder> builder(exv);
  Value *oldval = exv->getAggregateOperand();
  Value *newval = matchOp(oldval);
  if (!newval)
    return nullptr;
  FixedPointType baset =
      fixPType(newval).unwrapIndexList(oldval->getType(), exv->getIndices());
  std::vector<unsigned> idxlist(exv->indices().begin(), exv->indices().end());
  Value *newi = builder.CreateExtractValue(newval, idxlist);
  if (!baset.isInvalid() && newi->getType()->isIntegerTy())
    return genConvertFixedToFixed(newi, baset, fixpt);
  fixpt = baset;
  return newi;
}


Value *FloatToFixed::convertInsertValue(InsertValueInst *inv,
                                        FixedPointType &fixpt)
{
  if (valueInfo(inv)->noTypeConversion)
    return Unsupported;
  IRBuilder<NoFolder> builder(inv);
  Value *oldAggVal = inv->getAggregateOperand();
  Value *newAggVal = matchOp(oldAggVal);
  if (!newAggVal)
    return nullptr;
  FixedPointType baset = fixPType(newAggVal).unwrapIndexList(
      oldAggVal->getType(), inv->getIndices());
  Value *oldInsertVal = inv->getInsertedValueOperand();
  Value *newInsertVal;
  if (!baset.isInvalid())
    newInsertVal = translateOrMatchOperandAndType(oldInsertVal, baset);
  else
    newInsertVal = oldInsertVal;
  if (!newInsertVal)
    return nullptr;
  fixpt = fixPType(newAggVal);
  std::vector<unsigned> idxlist(inv->indices().begin(), inv->indices().end());
  return builder.CreateInsertValue(newAggVal, newInsertVal, idxlist);
}


Value *FloatToFixed::convertPhi(PHINode *phi, FixedPointType &fixpt)
{
  if (!phi->getType()->isFloatingPointTy() ||
      valueInfo(phi)->noTypeConversion) {
    /* in the conversion chain the floating point number was converted to
     * an int at some point; we just upgrade the incoming values in place */
    /* if all of our incoming values were not converted, we want to propagate
     * that information across the phi. If at least one of them was converted
     * the phi is converted as well; otherwise it is not. */
    bool donesomething = false;
    for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
      Value *thisval = phi->getIncomingValue(i);
      Value *newval = fallbackMatchValue(thisval, thisval->getType(), phi);
      if (newval && newval != ConversionError) {
        phi->setIncomingValue(i, newval);
        donesomething = true;
      }
    }
    return donesomething ? phi : nullptr;
  }
  /* if we have to do a type change, create a new phi node. The new type is for
   * sure that of a fixed point value; because the original type was a float
   * and thus all of its incoming values were floats */
  PHINode *newphi = PHINode::Create(fixpt.scalarToLLVMType(phi->getContext()),
                                    phi->getNumIncomingValues());
  for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
    Value *thisval = phi->getIncomingValue(i);
    BasicBlock *thisbb = phi->getIncomingBlock(i);
    Value *newval =
        translateOrMatchOperandAndType(thisval, fixpt, thisbb->getTerminator());
    if (!newval) {
      delete newphi;
      return nullptr;
    }
    Instruction *inst2 = dyn_cast<Instruction>(newval);
    if (inst2) {
      LLVM_DEBUG(dbgs() << "warning: new phi value " << *inst2
                        << " not coming from the obviously correct BB\n");
    }
    newphi->addIncoming(newval, thisbb);
  }
  newphi->insertAfter(phi);
  return newphi;
}


Value *FloatToFixed::convertSelect(SelectInst *sel, FixedPointType &fixpt)
{
  if (!isFloatingPointToConvert(sel))
    return Unsupported;
  /* the condition is always a bool (i1) or a vector of bools */
  Value *newcond = matchOp(sel->getCondition());
  /* otherwise create a new one */
  Value *newtruev =
      translateOrMatchAnyOperandAndType(sel->getTrueValue(), fixpt, sel);
  Value *newfalsev =
      translateOrMatchAnyOperandAndType(sel->getFalseValue(), fixpt, sel);
  if (!newtruev || !newfalsev || !newcond)
    return nullptr;
  SelectInst *newsel = SelectInst::Create(newcond, newtruev, newfalsev);
  newsel->insertAfter(sel);
  return newsel;
}


Value *FloatToFixed::convertCall(CallBase *call, FixedPointType &fixpt)
{
  /* If the function return a float the new return type will be a fix point of
   * type fixpt, otherwise the return type is left unchanged.*/
  Function *oldF = call->getCalledFunction();

  if (oldF->getName() == "clCreateBuffer") {
    /* hack! */
    LLVM_DEBUG(dbgs() << "clCreateBuffer detected, attempting to convert\n");
    Value *TheBuffer = call->getArgOperand(3);
    if (auto *BC = dyn_cast<BitCastOperator>(TheBuffer)) {
      TheBuffer = BC->getOperand(0);
    }
    Value *NewBuffer = matchOp(TheBuffer);
    if (!NewBuffer) {
      LLVM_DEBUG(dbgs() << "Arg to clCreateBuffer not converted; trying fallback.");
      return Unsupported;
    }
    LLVM_DEBUG(dbgs() << "Found converted buffer: " << *NewBuffer << "\n");
    LLVM_DEBUG(dbgs() << "clCreateBuffer buffer fixp type is: " << valueInfo(NewBuffer)->fixpType.toString() << "\n");
    Type *VoidPtrTy = Type::getInt8Ty(call->getContext())->getPointerTo();
    if (NewBuffer->getType() != VoidPtrTy) {
      NewBuffer = new BitCastInst(NewBuffer, VoidPtrTy, "", call);
    } 
    call->setArgOperand(3, NewBuffer);
    return call;
  }

  /* Special-case known math intrinsics */
  if (isSupportedMathIntrinsicFunction(oldF))
    return convertMathIntrinsicFunction(call, fixpt);
  /* Special case function prototypes and all other intrinsics */
  if (isSpecialFunction(oldF))
    return Unsupported;
  
  Function *newF = functionPool[oldF];
  if (!newF) {
    LLVM_DEBUG(dbgs() << "[Info] no function clone for instruction"
                      << *(call) << ", engaging fallback\n");
    return Unsupported;
  }
  LLVM_DEBUG(dbgs() << *(call)
                    << " will use converted function " << newF->getName() << " "
                    << *newF->getType() << "\n";);
  std::vector<Value *> convArgs;
  std::vector<Type *> typeArgs;
  std::vector<std::pair<int, FixedPointType>> fixArgs; // for match right function
  if (isFloatType(oldF->getReturnType())) {
    fixArgs.push_back(
        std::pair<int, FixedPointType>(-1, fixpt)); // ret value in signature
  }
  int i = 0;
  auto *call_arg = call->arg_begin();
  auto *f_arg = newF->arg_begin();
  while (call_arg != call->arg_end()) {
    Value *thisArgument;
    if (!hasInfo(f_arg)) {
      if (!hasInfo(*call_arg)) {
        thisArgument = *call_arg;
      } else {
        thisArgument = fallbackMatchValue(*call_arg, f_arg->getType());
      }
    } else if (hasInfo(*call_arg) &&
               valueInfo(*call_arg)->noTypeConversion == false) {
      FixedPointType argfpt, funfpt;
      argfpt = fixPType(*call_arg);
      funfpt = fixPType(&(*f_arg));
      if (!(argfpt == funfpt)) {
        LLVM_DEBUG(
            dbgs() << "CALL: fixed point type mismatch in actual argument " << i
                   << " (" << *f_arg << ") vs. formal argument\n");
        LLVM_DEBUG(dbgs() << "      (actual " << argfpt.toString()
                          << ", vs. formal " << funfpt.toString() << ")\n");
        LLVM_DEBUG(dbgs() << "      making an attempt to ignore the issue "
                             "because mem2reg can interfere\n");
      }
      thisArgument = translateOrMatchAnyOperandAndType(*call_arg, funfpt,
                                                       call);
      fixArgs.push_back(std::pair<int, FixedPointType>(i, funfpt));
    } else {
      FixedPointType funfpt;
      funfpt = fixPType(&(*f_arg));
      LLVM_DEBUG(dbgs() << "CALL: formal argument " << i << " (" << *f_arg
                        << ") converted but not actual argument\n");
      LLVM_DEBUG(dbgs() << "      making an attempt to ignore the issue "
                           "because mem2reg can interfere\n");
      if (call_arg->get()->getType()->isPointerTy()) {
        LLVM_DEBUG(dbgs() << "DANGER!! To make things worse, the problematic argument is a POINTER. Introducing a bitcast to try to salvage the mess.\n");
        Type *BCType = funfpt.toLLVMType(call_arg->get()->getType(), nullptr);
        thisArgument = new BitCastInst(call_arg->get(), BCType, call_arg->get()->getName() + ".salvaged", call);
      } else {
        thisArgument = translateOrMatchAnyOperandAndType(*call_arg, funfpt, call);
      }
    }
    if (!thisArgument) {
      LLVM_DEBUG(dbgs() << "CALL: match of argument " << i << " (" << *f_arg
                        << ") failed\n");
      return Unsupported;
    }
    convArgs.push_back(thisArgument);
    typeArgs.push_back(thisArgument->getType());
    if (convArgs[i]->getType() != f_arg->getType()) {
      LLVM_DEBUG(dbgs() << "CALL: type mismatch in actual argument " << i
                        << " (" << *f_arg << ") vs. formal argument\n");
      return nullptr;
    }
    i++;
    call_arg++;
    f_arg++;
  }
  if (isa<CallInst>(call)) {
    CallInst *newCall = CallInst::Create(newF, convArgs);
    newCall->setCallingConv(call->getCallingConv());
    newCall->insertBefore(call);
    return newCall;
  } else if (isa<InvokeInst>(call)) {
    InvokeInst *invk = dyn_cast<InvokeInst>(call);
    InvokeInst *newInvk = InvokeInst::Create(newF, invk->getNormalDest(),
                                             invk->getUnwindDest(), convArgs);
    newInvk->setCallingConv(call->getCallingConv());
    newInvk->insertBefore(invk);
    return newInvk;
  }
  assert(false && "Unknown CallBase type");
  return Unsupported;
}


Value *FloatToFixed::convertRet(ReturnInst *ret, FixedPointType &fixpt)
{
  Value *oldv = ret->getReturnValue();
  if (!oldv) // AKA return void
    return ret;
  if (!isFloatingPointToConvert(ret) || valueInfo(ret)->noTypeConversion) {
    // if return an int we shouldn't return a fix point, go into fallback
    return Unsupported;
  }
  Function *f = dyn_cast<Function>(ret->getParent()->getParent());
  Value *v = translateOrMatchAnyOperandAndType(oldv, fixpt);
  // check return type
  if (f->getReturnType() != v->getType())
    return nullptr;
  ret->setOperand(0, v);
  return ret;
}


Value *FloatToFixed::convertUnaryOp(Instruction *instr,
                                    const FixedPointType &fixpt)
{
  if (!instr->getType()->isFloatingPointTy() ||
      valueInfo(instr)->noTypeConversion)
    return Unsupported;

  unsigned int opc = instr->getOpcode();

  if (opc == Instruction::FNeg) {
    LLVM_DEBUG(instr->getOperand(0)->print(dbgs()););
    LLVM_DEBUG(dbgs() << "\n";);
    Value *val1 = translateOrMatchOperandAndType(instr->getOperand(0), fixpt, instr);
    if (!val1)
      return nullptr;
    IRBuilder<NoFolder> builder(instr);
    Value *fixop = nullptr;

    if (fixpt.isFixedPoint()) {
      fixop = builder.CreateNeg(val1);
    } else if (fixpt.isFloatingPoint()) {
      fixop = builder.CreateFNeg(val1);

    } else {
      llvm_unreachable("Unknown variable type. Are you trying to implement a "
                       "new datatype?");
    }
    cpMetaData(fixop, instr);
    updateConstTypeMetadata(fixop, 0U, fixpt);
    return fixop;
  }
  
  return Unsupported;
}


Value *FloatToFixed::convertBinOp(Instruction *instr,
                                  const FixedPointType &fixpt)
{
  /* Instruction::[Add,Sub,Mul,SDiv,UDiv,SRem,URem,Shl,LShr,AShr,And,Or,Xor]
   * are handled by the fallback function, not here */
  if (!instr->getType()->isFloatingPointTy() ||
      valueInfo(instr)->noTypeConversion)
    return Unsupported;

  int opc = instr->getOpcode();
  if (opc == Instruction::FAdd || opc == Instruction::FSub ||
      opc == Instruction::FRem) {
    LLVM_DEBUG(instr->getOperand(0)->print(dbgs()););
    LLVM_DEBUG(dbgs() << "\n";);
    LLVM_DEBUG(instr->getOperand(0)->print(dbgs()););
    LLVM_DEBUG(dbgs() << "\n";);
    Value *val1 =
        translateOrMatchOperandAndType(instr->getOperand(0), fixpt, instr);
    Value *val2 =
        translateOrMatchOperandAndType(instr->getOperand(1), fixpt, instr);
    if (!val1 || !val2)
      return nullptr;
    IRBuilder<NoFolder> builder(instr);
    Value *fixop;
    if (opc == Instruction::FAdd) {
      if (fixpt.isFixedPoint()) {
        fixop = builder.CreateBinOp(Instruction::Add, val1, val2);
      } else if (fixpt.isFloatingPoint()) {
        fixop = builder.CreateBinOp(Instruction::FAdd, val1, val2);
      } else {
        llvm_unreachable("Unknown variable type. Are you trying to implement a "
                         "new datatype?");
      }
    } else if (opc == Instruction::FSub) {
      // TODO: improve overflow resistance by shifting late
      LLVM_DEBUG(dbgs() << fixpt.toString() << "\n";);
      if (fixpt.isFixedPoint()) {
        fixop = builder.CreateBinOp(Instruction::Sub, val1, val2);
        LLVM_DEBUG(val1->print(dbgs()););
        LLVM_DEBUG(dbgs() << "\n";);
        LLVM_DEBUG(val2->print(dbgs()););
        LLVM_DEBUG(dbgs() << "\n";);
        LLVM_DEBUG(fixop->print(dbgs()););
        LLVM_DEBUG(dbgs() << "\n";);
      } else if (fixpt.isFloatingPoint()) {
        fixop = builder.CreateBinOp(Instruction::FSub, val1, val2);
      } else {
        llvm_unreachable("Unknown variable type. Are you trying to implement a "
                         "new datatype?");
      }
    } else /* if (opc == Instruction::FRem) */ {
      if (fixpt.isFixedPoint()) {
        if (fixpt.scalarIsSigned())
          fixop = builder.CreateBinOp(Instruction::SRem, val1, val2);
        else
          fixop = builder.CreateBinOp(Instruction::URem, val1, val2);
      } else if (fixpt.isFloatingPoint()) {
        fixop = builder.CreateBinOp(Instruction::FRem, val1, val2);
      } else {
        llvm_unreachable("Unknown variable type. Are you trying to implement a "
                         "new datatype?");
      }
    }
    updateConstTypeMetadata(fixop, 0U, fixpt);
    updateConstTypeMetadata(fixop, 1U, fixpt);
    return fixop;
  } else if (opc == Instruction::FMul) {
    FixedPointType intype1 = fixpt, intype2 = fixpt;
    if (fixpt.isFixedPoint()) {
      Value *val1 =
          translateOrMatchOperand(instr->getOperand(0), intype1, instr,
                                  TypeMatchPolicy::RangeOverHintMaxInt);
      Value *val2 =
          translateOrMatchOperand(instr->getOperand(1), intype2, instr,
                                  TypeMatchPolicy::RangeOverHintMaxInt);
      if (!val1 || !val2)
        return nullptr;
      FixedPointType intermtype(
          fixpt.scalarIsSigned(),
          intype1.scalarFracBitsAmt() + intype2.scalarFracBitsAmt(),
          intype1.scalarBitsAmt() + intype2.scalarBitsAmt());
      Type *dbfxt = intermtype.scalarToLLVMType(instr->getContext());
      IRBuilder<NoFolder> builder(instr);
      Value *ext1 = intype1.scalarIsSigned() ? builder.CreateSExt(val1, dbfxt)
                                             : builder.CreateZExt(val1, dbfxt);
      Value *ext2 = intype2.scalarIsSigned() ? builder.CreateSExt(val2, dbfxt)
                                             : builder.CreateZExt(val2, dbfxt);
      Value *fixop = builder.CreateMul(ext1, ext2);
      cpMetaData(ext1, val1);
      cpMetaData(ext2, val2);
      cpMetaData(fixop, instr);
      updateFPTypeMetadata(fixop, intermtype.scalarIsSigned(),
                           intermtype.scalarFracBitsAmt(),
                           intermtype.scalarBitsAmt());
      updateConstTypeMetadata(fixop, 0U, intype1);
      updateConstTypeMetadata(fixop, 1U, intype2);
      return genConvertFixedToFixed(fixop, intermtype, fixpt, instr);
    } else if (fixpt.isFloatingPoint()) {
      Value *val1 = translateOrMatchOperand(instr->getOperand(0), intype1,
                                            instr, TypeMatchPolicy::ForceHint);
      Value *val2 = translateOrMatchOperand(instr->getOperand(1), intype2,
                                            instr, TypeMatchPolicy::ForceHint);
      if (!val1 || !val2)
        return nullptr;
      IRBuilder<NoFolder> builder(instr);
      Value *fltop = builder.CreateFMul(val1, val2);
      return fltop;
    } else {
      llvm_unreachable(
          "Unknown variable type. Are you trying to implement a new datatype?");
    }
  } else if (opc == Instruction::FDiv) {
    // TODO: fix by using HintOverRange when it is actually implemented
    FixedPointType intype1 = fixpt, intype2 = fixpt;
    if (fixpt.isFixedPoint()) {
      Value *val1 = translateOrMatchOperand(instr->getOperand(0), intype1, instr, TypeMatchPolicy::RangeOverHintMaxFrac);
      Value *val2 = translateOrMatchOperand(instr->getOperand(1), intype2, instr, TypeMatchPolicy::RangeOverHintMaxInt);
      if (!val1 || !val2)
        return nullptr;
      LLVM_DEBUG(dbgs() << "fdiv val1 = " << *val1 << " type = " << intype1 << "\n");
      LLVM_DEBUG(dbgs() << "fdiv val2 = " << *val2 << " type = " << intype2 << "\n");

      /* Compute types of the intermediates */
      bool SignedRes = fixpt.scalarIsSigned();
      unsigned Ext2Exp = std::max(0, intype2.scalarFracBitsAmt() - (SignedRes && !intype2.scalarIsSigned() ? 1 : 0));
      unsigned Ext1Exp = fixpt.scalarFracBitsAmt() + Ext2Exp;
      unsigned Size = std::max(intype1.scalarBitsAmt(), intype2.scalarBitsAmt());
      if (Ext1Exp + intype1.scalarIntegerBitsAmt() > Size)
        Size = intype1.scalarBitsAmt() + intype2.scalarBitsAmt();

      /* Extend first operand */
      FixedPointType ext1type(SignedRes, Ext1Exp, Size);
      Value *ext1 = genConvertFixedToFixed(val1, intype1, ext1type, instr);

      /* Extend second operand */
      FixedPointType ext2type(SignedRes, Ext2Exp, Size);
      Value *ext2 = genConvertFixedToFixed(val2, intype2, ext2type, instr);

      /* Generate division */
      FixedPointType fixoptype(SignedRes, Ext1Exp - Ext2Exp, Size);
      IRBuilder<NoFolder> builder(instr);
      Value *fixop = fixpt.scalarIsSigned() ? builder.CreateSDiv(ext1, ext2)
                                            : builder.CreateUDiv(ext1, ext2);

      LLVM_DEBUG(dbgs() << "fdiv ext1 = " << *ext1 << " type = " << ext1type << "\n");
      LLVM_DEBUG(dbgs() << "fdiv ext2 = " << *ext2 << " type = " << ext2type << "\n");
      LLVM_DEBUG(dbgs() << "fdiv fixop = " << *fixop << "type = " << fixoptype << "\n");

      cpMetaData(ext1, val1);
      cpMetaData(ext2, val2);
      cpMetaData(fixop, instr);
      updateFPTypeMetadata(fixop, fixoptype.scalarIsSigned(),
                           fixoptype.scalarFracBitsAmt(),
                           fixoptype.scalarBitsAmt());
      updateConstTypeMetadata(fixop, 0U, ext1type);
      updateConstTypeMetadata(fixop, 1U, ext2type);
      return genConvertFixedToFixed(fixop, fixoptype, fixpt, instr);
    } else if (fixpt.isFloatingPoint()) {
      Value *val1 = translateOrMatchOperand(instr->getOperand(0), intype1,
                                            instr, TypeMatchPolicy::ForceHint);
      Value *val2 = translateOrMatchOperand(instr->getOperand(1), intype2,
                                            instr, TypeMatchPolicy::ForceHint);
      if (!val1 || !val2)
        return nullptr;
      IRBuilder<NoFolder> builder(instr);
      Value *fltop = builder.CreateFDiv(val1, val2);
      return fltop;
    } else {
      llvm_unreachable(
          "Unknown variable type. Are you trying to implement a new datatype?");
    }
  }
  return Unsupported;
}


Value *FloatToFixed::convertCmp(FCmpInst *fcmp)
{
  Value *op1 = fcmp->getOperand(0);
  Value *op2 = fcmp->getOperand(1);
  FixedPointType cmptype;
  FixedPointType t1, t2;
  bool hasinfo1 = hasInfo(op1), hasinfo2 = hasInfo(op2);
  bool isOneFloat = false;
  if (hasinfo1 && hasinfo2) {
    t1 = fixPType(op1);
    t2 = fixPType(op2);
    isOneFloat = t1.isFloatingPoint() || t2.isFloatingPoint();
  } else if (hasinfo1) {
    t1 = fixPType(op1);
    t2 = t1;
    t2.scalarIsSigned() = true;
    isOneFloat = t1.isFloatingPoint();
  } else if (hasinfo2) {
    t2 = fixPType(op2);
    t1 = t2;
    t1.scalarIsSigned() = true;
    isOneFloat = t2.isFloatingPoint();
  }
  if (!isOneFloat) {
    bool mixedsign = t1.scalarIsSigned() != t2.scalarIsSigned();
    int intpart1 = t1.scalarBitsAmt() - t1.scalarFracBitsAmt() +
                   (mixedsign ? t1.scalarIsSigned() : 0);
    int intpart2 = t2.scalarBitsAmt() - t2.scalarFracBitsAmt() +
                   (mixedsign ? t2.scalarIsSigned() : 0);
    cmptype.scalarIsSigned() = t1.scalarIsSigned() || t2.scalarIsSigned();
    cmptype.scalarFracBitsAmt() =
        std::max(t1.scalarFracBitsAmt(), t2.scalarFracBitsAmt());
    cmptype.scalarBitsAmt() =
        std::max(intpart1, intpart2) + cmptype.scalarFracBitsAmt();
    Value *val1 = translateOrMatchOperandAndType(op1, cmptype, fcmp);
    Value *val2 = translateOrMatchOperandAndType(op2, cmptype, fcmp);
    IRBuilder<NoFolder> builder(fcmp->getNextNode());
    CmpInst::Predicate ty;
    int pr = fcmp->getPredicate();
    bool swapped = false;
    // se unordered swappo, converto con la int, e poi mi ricordo di riswappare
    if (!CmpInst::isOrdered(fcmp->getPredicate())) {
      pr = fcmp->getInversePredicate();
      swapped = true;
    }
    if (pr == CmpInst::FCMP_OEQ) {
      ty = CmpInst::ICMP_EQ;
    } else if (pr == CmpInst::FCMP_ONE) {
      ty = CmpInst::ICMP_NE;
    } else if (pr == CmpInst::FCMP_OGT) {
      ty = cmptype.scalarIsSigned() ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT;
    } else if (pr == CmpInst::FCMP_OGE) {
      ty = cmptype.scalarIsSigned() ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE;
    } else if (pr == CmpInst::FCMP_OLE) {
      ty = cmptype.scalarIsSigned() ? CmpInst::ICMP_SLE : CmpInst::ICMP_ULE;
    } else if (pr == CmpInst::FCMP_OLT) {
      ty = cmptype.scalarIsSigned() ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
    } else if (pr == CmpInst::FCMP_ORD) {
      ;
      // TODO gestione NaN
    } else if (pr == CmpInst::FCMP_TRUE) {
      /* there is no integer-only always-true / always-false comparison
       * operator... so we roll out our own by producing a tautology */
      return builder.CreateICmpEQ(
          ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0),
          ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0));
    } else if (pr == CmpInst::FCMP_FALSE) {
      return builder.CreateICmpNE(
          ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0),
          ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0));
    }
    if (swapped) {
      ty = CmpInst::getInversePredicate(ty);
    }
    return val1 && val2 ? builder.CreateICmp(ty, val1, val2) : nullptr;
  } else {
    // Handling the presence of at least one float:
    // Converting all to the biggest float, then comparing as before
    if (t1.isFloatingPoint() && t2.isFloatingPoint()) {
      // take the biggest floating point
      if (t1.scalarToLLVMType(fcmp->getContext())->getPrimitiveSizeInBits() >
          t2.scalarToLLVMType(fcmp->getContext())->getPrimitiveSizeInBits()) {
        // t1 is "more precise"
        cmptype = t1;
      } else if (t1.scalarToLLVMType(fcmp->getContext())
                     ->getPrimitiveSizeInBits() <
                 t2.scalarToLLVMType(fcmp->getContext())
                     ->getPrimitiveSizeInBits()) {
        // t2 is "more precise"
        cmptype = t2;
      } else {
        // they are equal, yeah!
        cmptype = t1; // or t2, they are equal
        // FIXME: what if bfloat16 (for now unsupported) and half???
      }
    } else if (t1.isFloatingPoint()) {
      cmptype = t1;
    } else if (t2.isFloatingPoint()) {
      cmptype = t2;
    } else {
      llvm_unreachable("There should be at least one floating point.");
    }
    Value *val1 = translateOrMatchOperandAndType(op1, cmptype, fcmp);
    Value *val2 = translateOrMatchOperandAndType(op2, cmptype, fcmp);
    IRBuilder<NoFolder> builder(fcmp->getNextNode());
    return builder.CreateFCmp(fcmp->getPredicate(), // original predicate
                              val1, val2);
  }
}


Value *FloatToFixed::convertCast(CastInst *cast, const FixedPointType &fixpt)
{
  /* Instruction opcodes:
   * - [FPToSI,FPToUI,SIToFP,UIToFP] are handled here
   * - [Trunc,ZExt,SExt] are handled as a fallback case, not here
   * - [PtrToInt,IntToPtr,BitCast,AddrSpaceCast] might cause errors */

  IRBuilder<NoFolder> builder(cast->getNextNode());
  Value *operand = cast->getOperand(0);
  if (valueInfo(cast)->noTypeConversion)
    return Unsupported;
  if (BitCastInst *bc = dyn_cast<BitCastInst>(cast)) {
    Value *newOperand = operandPool[operand];
    Type *newType = getLLVMFixedPointTypeForFloatType(bc->getDestTy(), fixpt);
    if (newOperand && newOperand != ConversionError) {
      return builder.CreateBitCast(newOperand, newType);
    } else {
      return builder.CreateBitCast(operand, newType);
    }
  }
  if (operand->getType()->isFloatingPointTy()) {
    /* fptosi, fptoui, fptrunc, fpext */
    if (cast->getOpcode() == Instruction::FPToSI) {
      return translateOrMatchOperandAndType(
          operand, FixedPointType(cast->getType(), true), cast);
    } else if (cast->getOpcode() == Instruction::FPToUI) {
      return translateOrMatchOperandAndType(
          operand, FixedPointType(cast->getType(), false), cast);
    } else if (cast->getOpcode() == Instruction::FPTrunc ||
               cast->getOpcode() == Instruction::FPExt) {
      return translateOrMatchOperandAndType(operand, fixpt, cast);
    }
  } else {
    /* sitofp, uitofp */
    Value *val = matchOp(operand);
    if (cast->getOpcode() == Instruction::SIToFP) {
      return genConvertFixedToFixed(val, FixedPointType(val->getType(), true),
                                    fixpt, cast);
    } else if (cast->getOpcode() == Instruction::UIToFP) {
      return genConvertFixedToFixed(val, FixedPointType(val->getType(), false),
                                    fixpt, cast);
    }
  }
  return Unsupported;
}


Value *FloatToFixed::fallback(Instruction *unsupp, FixedPointType &fixpt)
{
  Value *fallval;
  Value *fixval;
  std::vector<Value *> newops;
  LLVM_DEBUG(dbgs() << "[Fallback] attempt to wrap not supported operation:\n"
                    << *unsupp << "\n");
  FallbackCount++;
  for (int i = 0, n = unsupp->getNumOperands(); i < n; i++) {
    fallval = unsupp->getOperand(i);
    fixval = fallbackMatchValue(fallval, fallval->getType(), unsupp);
    if (fixval) {
      LLVM_DEBUG(dbgs() << "  Substituted operand number : " << i + 1 << " of "
                        << n << "\n");
      newops.push_back(fixval);
    } else {
      newops.push_back(fallval);
    }
  }
  Instruction *tmp;
  if (valueInfo(unsupp)->noTypeConversion == false && !unsupp->isTerminator()) {
    tmp = unsupp->clone();
    if (!tmp->getType()->isVoidTy())
      tmp->setName(unsupp->getName() + ".flt");
    tmp->insertAfter(unsupp);
  } else {
    tmp = unsupp;
  }
  for (int i = 0, n = tmp->getNumOperands(); i < n; i++) {
    tmp->setOperand(i, newops[i]);
  }
  LLVM_DEBUG(dbgs() << "  mutated operands to:\n"
                    << *tmp << "\n");
  if (tmp->getType()->isFloatingPointTy() &&
      valueInfo(unsupp)->noTypeConversion == false) {
    Value *fallbackv =
        genConvertFloatToFix(tmp, fixpt, getFirstInsertionPointAfter(tmp));
    if (tmp->hasName())
      fallbackv->setName(tmp->getName() + ".fallback");
    return fallbackv;
  }
  return tmp;
}
