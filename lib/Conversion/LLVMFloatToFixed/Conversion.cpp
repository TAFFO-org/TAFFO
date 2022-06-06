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
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cmath>

using namespace llvm;
using namespace flttofix;
using namespace taffo;


Value *ConversionError = (Value *)(&ConversionError);
Value *Unsupported = (Value *)(&Unsupported);


void FloatToFixed::performConversion(
    Module &m,
    std::vector<Value *> &q)
{

  for (auto i = q.begin(); i != q.end();) {
    Value *v = *i;

    // Removes all taffo-related annotations from ll file
    if (CallInst *anno = dyn_cast<CallInst>(v)) {
      if (anno->getCalledFunction()) {
        if (anno->getCalledFunction()->getName() == "llvm.var.annotation") {
          anno->eraseFromParent();
          i = q.erase(i);
          continue;
        }
      }
    }

    LLVM_DEBUG(dbgs()
               << "\n-------------------------------------* performConversion *-------------------------------------\n");
    LLVM_DEBUG(dbgs() << "  [no conv ] " << valueInfo(v)->noTypeConversion << "\n");
    LLVM_DEBUG(dbgs() << "  [value   ] " << *v << "\n");
    if (Instruction *i = dyn_cast<Instruction>(v))
      LLVM_DEBUG(dbgs() << "  [function] " << i->getFunction()->getName() << "\n");

    Value *newv = convertSingleValue(m, v, valueInfo(v)->fixpType);
    if (newv) {
      operandPool[v] = newv;
    }

    if (newv && newv != ConversionError) {
      LLVM_DEBUG(dbgs() << "  [output  ] " << *newv << "\n");
      if (newv != v && isa<Instruction>(newv) && isa<Instruction>(v)) {
        Instruction *newinst = dyn_cast<Instruction>(newv);
        Instruction *oldinst = dyn_cast<Instruction>(v);
        newinst->setDebugLoc(oldinst->getDebugLoc());
      }
      cpMetaData(newv, v);
      if (newv != v) {
        if (hasInfo(newv)) {
          LLVM_DEBUG(dbgs() << "warning: output has valueInfo already from a previous conversion\n");
        } else {
          *newValueInfo(newv) = *valueInfo(v);
        }
      }
    } else {
      LLVM_DEBUG(dbgs() << "  [output  ] CONVERSION ERROR\n");
    }
    i++;
  }
}


Value *FloatToFixed::createPlaceholder(Type *type, BasicBlock *where, StringRef name)
{
  IRBuilder<> builder(where, where->getFirstInsertionPt());
  AllocaInst *alloca = builder.CreateAlloca(type);
  return builder.CreateLoad(type, alloca, name);
}


/* also inserts the new value in the basic blocks, alongside the old one */
Value *FloatToFixed::convertSingleValue(Module &m, Value *val, FixedPointType &fixpt)
{
  Value *res = Unsupported;

  if (valueInfo(val)->isArgumentPlaceholder) {
    return matchOp(val);
  } else if (Constant *con = dyn_cast<Constant>(val)) {
    /* Since constants never change, there is never anything to substitute
     * in them */
    if (!valueInfo(con)->noTypeConversion)
      res = convertConstant(con, fixpt, TypeMatchPolicy::RangeOverHintMaxFrac);
    else
      res = con;
  } else if (Instruction *instr = dyn_cast<Instruction>(val)) {
    res = convertInstruction(m, instr, fixpt);
  } else if (Argument *argument = dyn_cast<Argument>(val)) {
    if (isFloatType(argument->getType()))
      res = translateOrMatchOperand(val, fixpt, nullptr);
    else
      res = val;
  }

  return res ? res : ConversionError;
}


/* do not use on pointer operands */
/* In iofixpt there is also the source type*/
Value *
FloatToFixed::translateOrMatchOperand(Value *val, FixedPointType &iofixpt, Instruction *ip, TypeMatchPolicy typepol, bool wasHintForced)
{
  // FIXME: handle all the cases, we need more info about destination!
  if (typepol == TypeMatchPolicy::ForceHint) {
    LLVM_DEBUG(dbgs() << "Forcing hint!\n");
    FixedPointType origfixpt = iofixpt;
    llvm::Value *tmp = translateOrMatchOperand(val, iofixpt, ip, TypeMatchPolicy::RangeOverHintMaxFrac, true);
    return genConvertFixedToFixed(tmp, iofixpt, origfixpt, ip);
  }

  assert(val->getType()->getNumContainedTypes() == 0 && "translateOrMatchOperand val is not a scalar value");
  Value *res = operandPool[val];
  if (res) { // this means it has been converted, but can also be a floating point!
    if (res == ConversionError)
      /* the value should have been converted but it hasn't; bail out */
      return nullptr;

    // The value has to be converted into a floating point value, convert it, full stop.
    if (iofixpt.isFloatingPoint()) {
      return genConvertFixedToFixed(res, valueInfo(res)->fixpType, iofixpt, ip);
    }

    // Converting Floating point to whatever
    if (valueInfo(res)->fixpType.isFloatingPoint()) {
      LLVM_DEBUG(dbgs() << "Converting floating point to whatever.\n";);
      LLVM_DEBUG(dbgs() << "Is floating, calling subroutine, " << valueInfo(res)->fixpType.toString() << " --> " << iofixpt.toString() << "\n";);
      LLVM_DEBUG(dbgs() << "This value will be converted to fixpoint: ";);
      LLVM_DEBUG(val->print(dbgs()););
      LLVM_DEBUG(dbgs() << "\n";);
      // The conversion is not forced to a specific type, so let's choose the best type that contains our value
      // Which might NOT be the destination type
      // In fact, if we use the destination type, we risk screwing everything up causing overflows
      if (iofixpt.isFixedPoint()) {
        if (!wasHintForced) {
          // In this case we try to choose the best fixed point type!
          auto info = getInputInfo(val);
          if (!info || !info->IRange) {
            LLVM_DEBUG(dbgs() << "No metadata found, rolling with the suggested type and hoping for the best!\n");
          } else {
            associateFixFormat(info, iofixpt);
            LLVM_DEBUG(dbgs() << "We have a new fixed point suggested type: " << iofixpt.toString() << "\n";);
          }
        } else {
          LLVM_DEBUG(dbgs() << "Not associating better fixed point data type as the datatype was originally forced!\n";);
        }
      }

      return genConvertFixedToFixed(res, valueInfo(res)->fixpType, iofixpt, ip);
    }

    if (!valueInfo(val)->noTypeConversion) {
      /* the value has been successfully converted to fixed point in a previous step */
      iofixpt = fixPType(res);
      return res;
    }

    /* The value has changed but may not a fixed point */
    if (!res->getType()->isFloatingPointTy())
      /* Don't attempt to convert ints/pointers to fixed point */
      return res;
    /* Otherwise convert to fixed point the value */
    val = res;
  }

  assert(val->getType()->isFloatingPointTy());

  /* try the easy cases first
   *   this is essentially duplicated from genConvertFloatToFix because once we
   * enter that function iofixpt cannot change anymore
   *   in other words, by duplicating this logic here we potentially avoid a loss
   * of range if the suggested iofixpt is not enough for the value */
  if (Constant *c = dyn_cast<Constant>(val)) {
    Value *res = convertConstant(c, iofixpt, typepol);
    return res;
  } else if (iofixpt.isFixedPoint()) {
    // Only try to exclude conversion if we are trying to convert a float variable that has been converted
    if (SIToFPInst *instr = dyn_cast<SIToFPInst>(val)) {
      Value *intparam = instr->getOperand(0);
      iofixpt = FixedPointType(intparam->getType(), true);
      return intparam;
    } else if (UIToFPInst *instr = dyn_cast<UIToFPInst>(val)) {
      Value *intparam = instr->getOperand(0);
      iofixpt = FixedPointType(intparam->getType(), true);
      return intparam;
    }
  }

  /* not an easy case; check if the value has a range metadata
   * from VRA before giving up and using the suggested type */
  // Do this hack only if the final wanted type is a fixed point, otherwise we can go ahead
  if (iofixpt.isFixedPoint()) {
    mdutils::MDInfo *mdi = mdutils::MetadataManager::getMetadataManager().retrieveMDInfo(val);
    if (mdutils::InputInfo *ii = dyn_cast_or_null<mdutils::InputInfo>(mdi)) {
      if (ii->IRange) {
        FixedPointTypeGenError err;
        mdutils::FPType fpt = taffo::fixedPointTypeFromRange(*(ii->IRange), &err, iofixpt.scalarBitsAmt());
        if (err != FixedPointTypeGenError::InvalidRange)
          iofixpt = FixedPointType(&fpt);
      }
    }
  }

  return genConvertFloatToFix(val, iofixpt, ip);
}

bool FloatToFixed::associateFixFormat(mdutils::InputInfo *II, FixedPointType &iofixpt)
{
  mdutils::Range *rng = II->IRange.get();
  assert(rng && "No range info!");

  FixedPointTypeGenError fpgerr;
  // Using default parameters of DTA
  mdutils::FPType res = fixedPointTypeFromRange(*rng, &fpgerr, 32, 3, 64, 32);
  assert(fpgerr != FixedPointTypeGenError::InvalidRange && "Cannot assign a fixed point type!");

  iofixpt = FixedPointType(res.isSigned(), res.getPointPos(), res.getWidth());

  return true;
}


Value *FloatToFixed::genConvertFloatToFix(Value *flt, const FixedPointType &fixpt, Instruction *ip)
{
  assert(flt->getType()->isFloatingPointTy() && "genConvertFloatToFixed called on a non-float scalar");
  LLVM_DEBUG(dbgs() << "Called floatToFixed\n";);

  if (Constant *c = dyn_cast<Constant>(flt)) {
    FixedPointType fixptcopy = fixpt;
    Value *res = convertConstant(c, fixptcopy, TypeMatchPolicy::ForceHint);
    assert(fixptcopy == fixpt && "why is there a pointer here?");
    return res;
  }

  if (Instruction *i = dyn_cast<Instruction>(flt)) {
    if (!ip)
      ip = getFirstInsertionPointAfter(i);
  } else if (Argument *arg = dyn_cast<Argument>(flt)) {
    Function *fun = arg->getParent();
    BasicBlock &firstbb = fun->getEntryBlock();
    ip = &(*firstbb.getFirstInsertionPt());
  }
  assert(ip && "ip is mandatory if not passing an instruction/constant value");

  FloatToFixCount++;
  FloatToFixWeight += std::pow(2, std::min((int)(sizeof(int) * 8 - 1), this->getLoopNestingLevelOfValue(flt)));

  IRBuilder<> builder(ip);
  Type *destt = getLLVMFixedPointTypeForFloatType(flt->getType(), fixpt);

  /* insert new instructions before ip */
  if (!destt->isFloatingPointTy()) {
    if (SIToFPInst *instr = dyn_cast<SIToFPInst>(flt)) {
      Value *intparam = instr->getOperand(0);
      return cpMetaData(builder.CreateShl(
                            cpMetaData(builder.CreateIntCast(intparam, destt, true), flt, ip),
                            fixpt.scalarFracBitsAmt()),
                        flt, ip);
    } else if (UIToFPInst *instr = dyn_cast<UIToFPInst>(flt)) {
      Value *intparam = instr->getOperand(0);
      return cpMetaData(builder.CreateShl(
                            cpMetaData(builder.CreateIntCast(intparam, destt, false), flt, ip),
                            fixpt.scalarFracBitsAmt()),
                        flt, ip);
    } else {
      double twoebits = pow(2.0, fixpt.scalarFracBitsAmt());
      Value *interm = cpMetaData(builder.CreateFMul(
                                     cpMetaData(ConstantFP::get(flt->getType(), twoebits), flt, ip),
                                     flt),
                                 flt, ip);
      if (fixpt.scalarIsSigned()) {
        return cpMetaData(builder.CreateFPToSI(interm, destt), flt, ip);
      } else {
        return cpMetaData(builder.CreateFPToUI(interm, destt), flt, ip);
      }
    }
  } else {
    LLVM_DEBUG(dbgs() << "[genConvertFloatToFix] converting a floating point to a floating point\n");
    int startingBit = flt->getType()->getPrimitiveSizeInBits();
    int destinationBit = destt->getPrimitiveSizeInBits();
    if (startingBit == destinationBit) {
      // No casting is actually needed
      assert((flt->dump(), destt->dump(), 1) && flt->getType() == destt && "Floating types having same bits but differents types");
      LLVM_DEBUG(dbgs() << "[genConvertFloatToFix] no casting needed.\n");
      return flt;
    } else if (startingBit < destinationBit) {
      // Extension needed
      return cpMetaData(builder.CreateFPExt(flt, destt), flt, ip);
    } else {
      // Truncation needed
      return cpMetaData(builder.CreateFPTrunc(flt, destt), flt, ip);
    }
  }
}


Value *FloatToFixed::genConvertFixedToFixed(Value *fix, const FixedPointType &srct, const FixedPointType &destt,
                                            Instruction *ip)
{
  if (srct == destt)
    return fix;

  LLVM_DEBUG(dbgs() << "Called fixedToFixed\n";);

  Instruction *fixinst = dyn_cast<Instruction>(fix);
  if (!ip && fixinst)
    ip = getFirstInsertionPointAfter(fixinst);
  assert(ip && "ip required when converted value not an instruction");

  Type *llvmsrct = fix->getType();
  Type *llvmdestt = destt.scalarToLLVMType(fix->getContext());

  // Source and destination are both float
  if (llvmsrct->isFloatingPointTy() &&
      llvmdestt->isFloatingPointTy()) {
    IRBuilder<> builder(ip);

    LLVM_DEBUG(dbgs() << "[genConvertFloatToFix] converting a floating point to a floating point\n");
    int startingBit = llvmsrct->getPrimitiveSizeInBits();
    int destinationBit = llvmdestt->getPrimitiveSizeInBits();

    if (startingBit == destinationBit) {
      // No casting is actually needed
      assert((llvmsrct->dump(), llvmdestt->dump(), 1) && llvmsrct == llvmdestt && "Floating types having same bits but differents types");
      LLVM_DEBUG(dbgs() << "[genConvertFloatToFix] no casting needed.\n");
      return fix;
    } else if (startingBit < destinationBit) {
      // Extension needed
      return cpMetaData(builder.CreateFPExt(fix, llvmdestt), fix, ip);
    } else {
      // Truncation needed
      return cpMetaData(builder.CreateFPTrunc(fix, llvmdestt), fix, ip);
    }
  }

  /*We should never have these case in general, but when using mixed precision this can (and will) happen*/
  if (srct.isFloatingPoint() &&
      destt.isFixedPoint()) {
    return genConvertFloatToFix(fix, destt, ip);
  }

  if (srct.isFixedPoint() &&
      destt.isFloatingPoint()) {
    return genConvertFixToFloat(fix, srct, llvmdestt);
  }

  assert(llvmsrct->isSingleValueType() && "cannot change type of a pointer");
  assert(llvmsrct->isIntegerTy() && "cannot change type of a float");

  IRBuilder<> builder(ip);

  auto genSizeChange = [&](Value *fix) -> Value * {
    if (srct.scalarIsSigned()) {
      return cpMetaData(builder.CreateSExtOrTrunc(fix, llvmdestt), fix);
    } else {
      return cpMetaData(builder.CreateZExtOrTrunc(fix, llvmdestt), fix);
    }
  };

  auto genPointMovement = [&](Value *fix) -> Value * {
    int deltab = destt.scalarFracBitsAmt() - srct.scalarFracBitsAmt();
    if (deltab > 0) {
      return cpMetaData(builder.CreateShl(fix, deltab), fix);
    } else if (deltab < 0) {
      if (srct.scalarIsSigned()) {
        return cpMetaData(builder.CreateAShr(fix, -deltab), fix);
      } else {
        return cpMetaData(builder.CreateLShr(fix, -deltab), fix);
      }
    }
    return fix;
  };

  if (destt.scalarBitsAmt() > srct.scalarBitsAmt())
    return genPointMovement(genSizeChange(fix));
  return genSizeChange(genPointMovement(fix));
}


Value *FloatToFixed::genConvertFixToFloat(Value *fix, const FixedPointType &fixpt, Type *destt)
{
  LLVM_DEBUG(dbgs() << "******** trace: genConvertFixToFloat ";
             fix->print(dbgs());
             dbgs() << " -> ";
             destt->print(dbgs());
             dbgs() << "\n";);


  if (fix->getType()->isFloatingPointTy()) {
    if (isa<Instruction>(fix) || isa<Argument>(fix)) {
      Instruction *ip = nullptr;
      if (Instruction *i = dyn_cast<Instruction>(fix)) {
        ip = getFirstInsertionPointAfter(i);
      } else if (Argument *arg = dyn_cast<Argument>(fix)) {
        ip = &(*(arg->getParent()->getEntryBlock().getFirstInsertionPt()));
      }
      IRBuilder<> builder(ip);

      LLVM_DEBUG(dbgs() << "[genConvertFixToFloat] converting a floating point to a floating point\n");
      int startingBit = fix->getType()->getPrimitiveSizeInBits();
      int destinationBit = destt->getPrimitiveSizeInBits();

      if (startingBit == destinationBit) {
        // No casting is actually needed
        assert((fix->dump(), destt->dump(), 1) && fix->getType() == destt && "Floating types having same bits but differents types");
        LLVM_DEBUG(dbgs() << "[genConvertFixToFloat] no casting needed.\n");
        return fix;
      } else if (startingBit < destinationBit) {
        // Extension needed
        return cpMetaData(builder.CreateFPExt(fix, destt), fix, ip);
      } else {
        // Truncation needed
        return cpMetaData(builder.CreateFPTrunc(fix, destt), fix, ip);
      }
    } else if (Constant *cst = dyn_cast<Constant>(fix)) {

      int startingBit = fix->getType()->getPrimitiveSizeInBits();
      int destinationBit = destt->getPrimitiveSizeInBits();

      if (startingBit == destinationBit) {
        // No casting is actually needed
        assert((fix->dump(), destt->dump(), 1) && fix->getType() == destt && "Floating types having same bits but differents types");
        LLVM_DEBUG(dbgs() << "[genConvertFixToFloat] no casting needed.\n");
        return fix;
      } else if (startingBit < destinationBit) {
        // Extension needed
        return ConstantExpr::getFPExtend(cst, destt);
      } else {
        // Truncation needed
        return ConstantExpr::getFPTrunc(cst, destt);
      }


      Constant *floattmp = fixpt.scalarIsSigned() ? ConstantExpr::getSIToFP(cst, destt) : ConstantExpr::getUIToFP(cst, destt);
      double twoebits = pow(2.0, fixpt.scalarFracBitsAmt());
      return ConstantExpr::getFDiv(floattmp, ConstantFP::get(destt, twoebits));
    }
  }

  if (!fix->getType()->isIntegerTy()) {
    LLVM_DEBUG(errs() << "can't wrap-convert to flt non integer value ";
               fix->print(errs());
               errs() << "\n");
    return nullptr;
  }


  FixToFloatCount++;
  FixToFloatWeight += std::pow(2, std::min((int)(sizeof(int) * 8 - 1), this->getLoopNestingLevelOfValue(fix)));

  if (isa<Instruction>(fix) || isa<Argument>(fix)) {
    Instruction *ip = nullptr;
    if (Instruction *i = dyn_cast<Instruction>(fix)) {
      ip = getFirstInsertionPointAfter(i);
    } else if (Argument *arg = dyn_cast<Argument>(fix)) {
      ip = &(*(arg->getParent()->getEntryBlock().getFirstInsertionPt()));
    }
    IRBuilder<> builder(ip);

    Value *floattmp = fixpt.scalarIsSigned() ? builder.CreateSIToFP(fix, destt) : builder.CreateUIToFP(fix, destt);
    cpMetaData(floattmp, fix);
    double twoebits = pow(2.0, fixpt.scalarFracBitsAmt());

    if (twoebits != 1.0) {

      return cpMetaData(builder.CreateFDiv(floattmp,
                                           cpMetaData(ConstantFP::get(destt, twoebits), fix)),
                        fix);
    } else {
      LLVM_DEBUG(dbgs() << "Optimizing conversion removing division by one!\n";);
      return floattmp;
    }

  } else if (Constant *cst = dyn_cast<Constant>(fix)) {
    Constant *floattmp = fixpt.scalarIsSigned() ? ConstantExpr::getSIToFP(cst, destt) : ConstantExpr::getUIToFP(cst, destt);
    double twoebits = pow(2.0, fixpt.scalarFracBitsAmt());
    return ConstantExpr::getFDiv(floattmp, ConstantFP::get(destt, twoebits));
  }

  llvm_unreachable("unrecognized value type passed to genConvertFixToFloat");
}


Type *FloatToFixed::getLLVMFixedPointTypeForFloatType(Type *srct, const FixedPointType &baset, bool *hasfloats)
{
  if (srct->isPointerTy()) {
    Type *enc = getLLVMFixedPointTypeForFloatType(srct->getPointerElementType(), baset, hasfloats);
    if (enc)
      return enc->getPointerTo();
    return nullptr;

  } else if (srct->isArrayTy()) {
    int nel = srct->getArrayNumElements();
    Type *enc = getLLVMFixedPointTypeForFloatType(srct->getArrayElementType(), baset, hasfloats);
    if (enc)
      return ArrayType::get(enc, nel);
    return nullptr;

  } else if (srct->isStructTy()) {
    SmallVector<Type *, 2> elems;
    bool allinvalid = true;
    for (int i = 0; i < srct->getStructNumElements(); i++) {
      const FixedPointType &fpelemt = baset.structItem(i);
      Type *baseelemt = srct->getStructElementType(i);
      Type *newelemt;
      if (!fpelemt.isInvalid()) {
        allinvalid = false;
        newelemt = getLLVMFixedPointTypeForFloatType(baseelemt, fpelemt, hasfloats);
      } else {
        newelemt = baseelemt;
      }
      elems.push_back(newelemt);
    }
    if (!allinvalid)
      return StructType::get(srct->getContext(), elems, dyn_cast<StructType>(srct)->isPacked());
    return srct;

  } else if (srct->isFloatingPointTy()) {
    if (hasfloats)
      *hasfloats = true;
    return baset.scalarToLLVMType(srct->getContext());
  }
  LLVM_DEBUG(
      dbgs() << "getLLVMFixedPointTypeForFloatType given unexpected non-float type ";
      srct->print(dbgs());
      dbgs() << "\n";);
  if (hasfloats)
    *hasfloats = false;
  return srct;
}


Type *FloatToFixed::getLLVMFixedPointTypeForFloatValue(Value *val)
{
  FixedPointType &fpt = fixPType(val);
  return getLLVMFixedPointTypeForFloatType(val->getType(), fpt);
}
