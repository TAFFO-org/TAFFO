#include "LLVMFloatToFixedPass.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <cassert>
#include <cmath>

#include "Debug/Logger.hpp"

using namespace llvm;
using namespace flttofix;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"


Value *ConversionError = (Value *)(&ConversionError);
Value *Unsupported = (Value *)(&Unsupported);


void FloatToFixed::performConversion(Module &m, std::vector<Value*> &q) {
  for (auto iter = q.begin(); iter != q.end();) {
    Value *v = *iter;

    // Removes all taffo-related annotations from ll file
    if (CallInst *anno = dyn_cast<CallInst>(v)) {
      if (anno->getCalledFunction()) {
        if (anno->getCalledFunction()->getName().starts_with("llvm.var.annotation")) {
          anno->eraseFromParent();
          TaffoInfo::getInstance().eraseValue(*anno);
          iter = q.erase(iter);
          continue;
        }
      }
    }

    LLVM_DEBUG(dbgs() << "\n-------------------------------------* performConversion *-------------------------------------\n");
    LLVM_DEBUG(dbgs() << "  [no conv ] " << getConversionInfo(v)->noTypeConversion << "\n");
    LLVM_DEBUG(dbgs() << "  [value   ] " << *v << "\n");
    if (Instruction *i = dyn_cast<Instruction>(v))
      LLVM_DEBUG(dbgs() << "  [function] " << i->getFunction()->getName() << "\n");

    std::shared_ptr<FixedPointType> newType = getFixpType(v);
    LLVM_DEBUG(dbgs() << "  [req. ty.] " << *newType << "\n");
    Value *newv = convertSingleValue(m, v, newType);
    getConversionInfo(v)->fixpType = newType;
    if (newv) {
      operandPool[v] = newv;
      LLVM_DEBUG(dbgs() << "  [out  ty.] " << *newType << "\n");
    }

    if (newv && newv != ConversionError) {
      LLVM_DEBUG(dbgs() << "  [output  ] " << *newv << "\n");

      if (newv != v && isa<Instruction>(newv) && isa<Instruction>(v)) {
        Instruction *newinst = dyn_cast<Instruction>(newv);
        Instruction *oldinst = dyn_cast<Instruction>(v);
        newinst->setDebugLoc(oldinst->getDebugLoc());
      }
      if (newv != v) {
        std::shared_ptr<TransparentType> oldTransparentType = TaffoInfo::getInstance().getOrCreateTransparentType(*v);
        std::shared_ptr<TransparentType> newTransparentType;
        if (!newType->isInvalid())
          newTransparentType = newType->toTransparentType(oldTransparentType, nullptr);
        else
          newTransparentType = oldTransparentType;
        cpMetaData(newv, v, nullptr, newTransparentType);

        if (hasConversionInfo(newv)) {
          LLVM_DEBUG(dbgs() << "warning: output has valueInfo already from a previous conversion (type " << *getFixpType(newv) << ")\n");
          if (*getFixpType(newv) != *getFixpType(v)) {
            Logger &logger = Logger::getInstance();
            logger.logln("FATAL ERROR: SAME VALUE INSTANCE HAS TWO DIFFERENT SEMANTICS!", raw_ostream::Colors::RED);
            logger.log("New type: ", raw_ostream::Colors::RED);
            logger.log(getFixpType(newv), raw_ostream::Colors::RED);
            logger.log(", old type: ", raw_ostream::Colors::RED);
            logger.logln(getFixpType(v), raw_ostream::Colors::RED);
            abort();
          }
        }
        else
          *newConversionInfo(newv) = *getConversionInfo(v);
      }
    } else {
      LLVM_DEBUG(dbgs() << "  [output  ] CONVERSION ERROR\n");
    }
    iter++;
  }
}


Value *FloatToFixed::createPlaceholder(Type *type, BasicBlock *where, StringRef name)
{
  IRBuilder<NoFolder> builder(where, where->getFirstInsertionPt());
  AllocaInst *alloca = builder.CreateAlloca(type);
  return builder.CreateLoad(type, alloca, name);
}


/* also inserts the new value in the basic blocks, alongside the old one */
Value *FloatToFixed::convertSingleValue(Module &m, Value *val, std::shared_ptr<FixedPointType> &fixpt) {
  Value *res = Unsupported;

  if (getConversionInfo(val)->isArgumentPlaceholder) {
    return matchOp(val);
  } else if (Constant *con = dyn_cast<Constant>(val)) {
    /* Since constants never change, there is never anything to substitute in them */
    if (!getConversionInfo(con)->noTypeConversion)
      res = convertConstant(con, fixpt, TypeMatchPolicy::RangeOverHintMaxFrac);
    else
      res = con;
  } else if (Instruction *instr = dyn_cast<Instruction>(val)) {
    res = convertInstruction(m, instr, fixpt);
  } else if (Argument *argument = dyn_cast<Argument>(val)) {
    if (getUnwrappedType(argument)->isFloatTy())
      res = translateOrMatchOperand(val, fixpt, nullptr);
    else
      res = val;
  }

  return res ? res : ConversionError;
}


/* do not use on pointer operands */
/* In iofixpt there is also the source type*/
Value *
FloatToFixed::translateOrMatchOperand(Value *val, std::shared_ptr<FixedPointType> &iofixpt, Instruction *ip, TypeMatchPolicy typepol, bool wasHintForced) {
  LLVM_DEBUG(dbgs() << "translateOrMatchOperand of " << *val << "\n");
  
  // FIXME: handle all the cases, we need more info about destination!
  if (typepol == TypeMatchPolicy::ForceHint) {
    LLVM_DEBUG(dbgs() << "translateOrMatchOperand: forcing hint as requested!\n");
    std::shared_ptr<FixedPointType> origfixpt = iofixpt->clone();
    Value *tmp = translateOrMatchOperand(val, iofixpt, ip, TypeMatchPolicy::RangeOverHintMaxFrac, true);
    return genConvertFixedToFixed(tmp, std::static_ptr_cast<FixedPointScalarType>(iofixpt), std::static_ptr_cast<FixedPointScalarType>(origfixpt), ip);
  }

  assert(val->getType()->getNumContainedTypes() == 0 && "translateOrMatchOperand val is not a scalar value");
  Value *res = operandPool[val];
  if (res) { // this means it has been converted, but can also be a floating point!
    if (res == ConversionError)
      /* the value should have been converted but it hasn't; bail out */
      return nullptr;

    // The value has to be converted into a floating point value, convert it, full stop.
    if (iofixpt->isFloatingPoint()) {
      LLVM_DEBUG(dbgs() << "translateOrMatchOperand: converting converted value to floating point\n");
      return genConvertFixedToFixed(res, std::static_ptr_cast<FixedPointScalarType>(getFixpType(res)), std::static_ptr_cast<FixedPointScalarType>(iofixpt), ip);
    }

    // Converting Floating point to whatever
    if (getFixpType(res)->isFloatingPoint()) {
      LLVM_DEBUG(dbgs() << "translateOrMatchOperand: Converting floating point to whatever.\n";);
      LLVM_DEBUG(dbgs() << "Is floating, calling subroutine, " << *getFixpType(res) << " --> " << *iofixpt << "\n";);
      LLVM_DEBUG(dbgs() << "This value will be converted to fixpoint: ";);
      LLVM_DEBUG(val->print(dbgs()););
      LLVM_DEBUG(dbgs() << "\n";);
      // The conversion is not forced to a specific type, so let's choose the best type that contains our value
      // Which might NOT be the destination type
      // In fact, if we use the destination type, we risk screwing everything up causing overflows
      if (iofixpt->isFixedPoint()) {
        if (!wasHintForced) {
          // In this case we try to choose the best fixed point type!
          std::shared_ptr<ValueInfo> valueInfo = TaffoInfo::getInstance().getValueInfo(*val);
          std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast_or_null<ScalarInfo>(valueInfo);
          if (!scalarInfo || !scalarInfo->range) {
            LLVM_DEBUG(dbgs() << "No metadata found, rolling with the suggested type and hoping for the best!\n");
          } else {
            associateFixFormat(scalarInfo, iofixpt);
            LLVM_DEBUG(dbgs() << "We have a new fixed point suggested type: " << *iofixpt << "\n";);
          }
        } else {
          LLVM_DEBUG(dbgs() << "Not associating better fixed point data type as the datatype was originally forced!\n";);
        }
      }

      return genConvertFixedToFixed(res, std::static_ptr_cast<FixedPointScalarType>(getFixpType(res)), std::static_ptr_cast<FixedPointScalarType>(iofixpt), ip);
    }

    if (!getConversionInfo(val)->noTypeConversion) {
      /* the value has been successfully converted to fixed point in a previous step */
      LLVM_DEBUG(dbgs() << "translateOrMatchOperand: value has been converted in the past to \n"; 
          dbgs().indent(18)  << "Value: " << *res << "\n";
          dbgs().indent(16) << "FixType: " << *iofixpt << "\n");
      iofixpt = getFixpType(res);
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

  LLVM_DEBUG(dbgs() << "translateOrMatchOperand: non-converted value, converting now\n");

  /* try the easy cases first
   *   this is essentially duplicated from genConvertFloatToFix because once we
   * enter that function iofixpt cannot change anymore
   *   in other words, by duplicating this logic here we potentially avoid a loss
   * of range if the suggested iofixpt is not enough for the value */
  if (Constant *c = dyn_cast<Constant>(val)) {
    Value *res = convertConstant(c, iofixpt, typepol);
    return res;
  } else if (iofixpt->isFixedPoint()) {
    // Only try to exclude conversion if we are trying to convert a float variable that has been converted
    if (SIToFPInst *instr = dyn_cast<SIToFPInst>(val)) {
      Value *intparam = instr->getOperand(0);
      iofixpt = std::make_shared<FixedPointScalarType>(intparam->getType(), true);
      return intparam;
    } else if (UIToFPInst *instr = dyn_cast<UIToFPInst>(val)) {
      Value *intparam = instr->getOperand(0);
      iofixpt = std::make_shared<FixedPointScalarType>(intparam->getType(), true);
      return intparam;
    }
  }

  /* not an easy case; check if the value has a range metadata
   * from VRA before giving up and using the suggested type */
  // Do this hack only if the final wanted type is a fixed point, otherwise we can go ahead
  if (iofixpt->isFixedPoint()) {
    std::shared_ptr<ValueInfo> mdi = TaffoInfo::getInstance().getValueInfo(*val);
    if (std::shared_ptr<ScalarInfo> ii = std::dynamic_ptr_cast_or_null<ScalarInfo>(mdi)) {
      if (ii->range) {
        FixedPointTypeGenError err;
        FixpType fpt = fixedPointTypeFromRange(*ii->range, &err, std::static_ptr_cast<FixedPointScalarType>(iofixpt)->getBits());
        if (err != FixedPointTypeGenError::InvalidRange)
          iofixpt = std::make_shared<FixedPointScalarType>(&fpt);
      }
    }
  }

  return genConvertFloatToFix(val, std::static_ptr_cast<FixedPointScalarType>(iofixpt), ip);
}

bool FloatToFixed::associateFixFormat(const std::shared_ptr<ScalarInfo> &II, std::shared_ptr<FixedPointType> &iofixpt) {
  Range *rng = II->range.get();
  assert(rng && "No range info!");

  FixedPointTypeGenError fpgerr;
  // Using default parameters of DTA
  FixpType res = fixedPointTypeFromRange(*rng, &fpgerr, 32, 3, 64, 32);
  assert(fpgerr != FixedPointTypeGenError::InvalidRange && "Cannot assign a fixed point type!");

  iofixpt = std::make_shared<FixedPointScalarType>(res.isSigned(), res.getWidth(), res.getPointPos());

  return true;
}


// TODO: rewrite this mess!
Value *FloatToFixed::genConvertFloatToFix(Value *flt, const std::shared_ptr<FixedPointScalarType> &fixpt, Instruction *ip)
{
  assert(flt->getType()->isFloatingPointTy() && "genConvertFloatToFixed called on a non-float scalar");
  LLVM_DEBUG(dbgs() << "Called floatToFixed\n";);

  if (Constant *c = dyn_cast<Constant>(flt)) {
    std::shared_ptr<FixedPointType> fixptcopy = fixpt->clone();
    Value *res = convertConstant(c, fixptcopy, TypeMatchPolicy::ForceHint);
    assert(*fixptcopy == *fixpt && "why is there a pointer here?");
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

  IRBuilder<NoFolder> builder(ip);
  Type *SrcTy = flt->getType();
  Type *destt = getLLVMFixedPointTypeForFloatType(TaffoInfo::getInstance().getOrCreateTransparentType(*flt), fixpt);

  /* insert new instructions before ip */
  if (!destt->isFloatingPointTy()) {
    if (SIToFPInst *instr = dyn_cast<SIToFPInst>(flt)) {
      Value *intparam = instr->getOperand(0);
      return cpMetaData(builder.CreateShl(
                            cpMetaData(builder.CreateIntCast(intparam, destt, true), flt, ip),
                            fixpt->getFractionalBits()),
                        flt, ip);
    } else if (UIToFPInst *instr = dyn_cast<UIToFPInst>(flt)) {
      Value *intparam = instr->getOperand(0);
      return cpMetaData(builder.CreateShl(
                            cpMetaData(builder.CreateIntCast(intparam, destt, false), flt, ip),
                            fixpt->getFractionalBits()),
                        flt, ip);
    } else {
      double twoebits = pow(2.0, fixpt->getFractionalBits());
      const fltSemantics &FltSema = SrcTy->getFltSemantics();
      double MaxSrc = APFloat::getLargest(FltSema).convertToDouble();
      double MaxDest = pow(2.0, fixpt->getBits());
      Value *SanitizedFloat = flt;
      if (MaxSrc < MaxDest || MaxSrc < twoebits) {
        LLVM_DEBUG(dbgs() << "floatToFixed: Extending " << *SrcTy << " to float because dest integer is too large\n");
        SanitizedFloat = builder.CreateFPCast(flt, Type::getFloatTy(flt->getContext()));
        cpMetaData(SanitizedFloat, flt, ip);
      }
      Type *IntermType = SanitizedFloat->getType();
      Value *interm = cpMetaData(builder.CreateFMul(
                                     cpMetaData(ConstantFP::get(IntermType, twoebits), SanitizedFloat, ip),
                                     SanitizedFloat),
                                 SanitizedFloat, ip);
      if (fixpt->isSigned()) {
        return cpMetaData(builder.CreateFPToSI(interm, destt), SanitizedFloat, ip);
      } else {
        return cpMetaData(builder.CreateFPToUI(interm, destt), SanitizedFloat, ip);
      }
    }
  } else {
    LLVM_DEBUG(dbgs() << "[genConvertFloatToFix] converting a floating point to a floating point\n");
    int startingBit = flt->getType()->getPrimitiveSizeInBits();
    int destinationBit = destt->getPrimitiveSizeInBits();
    if (startingBit == destinationBit) {
      // No casting is actually needed
      LLVM_DEBUG(assert((flt->dump(), destt->dump(), 1) && flt->getType() == destt && "Floating types having same bits but differents types"));
      LLVM_DEBUG(dbgs() << "[genConvertFloatToFix] no casting needed.\n");
      assert(flt->getType() == destt && "Floating types having same bits but differents types");
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


// TODO: rewrite this mess!
Value *FloatToFixed::genConvertFixedToFixed(
  Value *fix, const std::shared_ptr<FixedPointScalarType> &srct, const std::shared_ptr<FixedPointScalarType> &destt, Instruction *ip)
{
  auto& log = Logger::getInstance();
  if (*srct == *destt)
    return fix;

  LLVM_DEBUG(
      log.log("Called fixedToFixed with src ");
      log.log(*srct, llvm::raw_ostream::Colors::BLUE);
      log.log(" to dst ");
      log.logln(*destt, llvm::raw_ostream::Colors::BLUE);
      );


  Instruction *fixinst = dyn_cast<Instruction>(fix);
  if (!ip && fixinst)
    ip = getFirstInsertionPointAfter(fixinst);
  assert(ip && "ip required when converted value not an instruction");

  Type *llvmsrct = fix->getType();
  Type *llvmdestt = destt->scalarToLLVMType(fix->getContext());

  // Source and destination are both float
  if (llvmsrct->isFloatingPointTy() &&
      llvmdestt->isFloatingPointTy()) {
    IRBuilder<NoFolder> builder(ip);

    LLVM_DEBUG(dbgs() << "[genConvertFloatToFix] converting a floating point to a floating point\n");
    int startingBit = llvmsrct->getPrimitiveSizeInBits();
    int destinationBit = llvmdestt->getPrimitiveSizeInBits();

    if (startingBit == destinationBit) {
      // No casting is actually needed
      LLVM_DEBUG(assert((llvmsrct->dump(), llvmdestt->dump(), 1) && llvmsrct == llvmdestt && "Floating types having same bits but differents types"));
      assert( llvmsrct == llvmdestt && "Floating types having same bits but differents types");
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
  if (srct->isFloatingPoint() &&
      destt->isFixedPoint()) {
    return genConvertFloatToFix(fix, destt, ip);
  }

  if (srct->isFixedPoint() &&
      destt->isFloatingPoint()) {
    return genConvertFixToFloat(fix, srct, llvmdestt);
  }

  assert(llvmsrct->isSingleValueType() && "cannot change type of a pointer");
  assert(llvmsrct->isIntegerTy() && "cannot change type of a float");

  IRBuilder<NoFolder> builder(ip);

  auto genSizeChange = [&](Value *fix) -> Value * {
    if (srct->isSigned()) {
      return cpMetaData(builder.CreateSExtOrTrunc(fix, llvmdestt), fix);
    } else {
      return cpMetaData(builder.CreateZExtOrTrunc(fix, llvmdestt), fix);
    }
  };

  auto genPointMovement = [&](Value *fix) -> Value * {
    int deltab = destt->getFractionalBits() - srct->getFractionalBits();
    if (deltab > 0) {
      return cpMetaData(builder.CreateShl(fix, deltab), fix);
    } else if (deltab < 0) {
      if (srct->isSigned()) {
        return cpMetaData(builder.CreateAShr(fix, -deltab), fix);
      } else {
        return cpMetaData(builder.CreateLShr(fix, -deltab), fix);
      }
    }
    return fix;
  };

  if (destt->getBits() > srct->getBits())
    return genPointMovement(genSizeChange(fix));
  return genSizeChange(genPointMovement(fix));
}


// TODO: rewrite this mess!
Value *FloatToFixed::genConvertFixToFloat(Value *fix, const std::shared_ptr<FixedPointType> &fixpt, Type *destt)
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
      IRBuilder<NoFolder> builder(ip);

      LLVM_DEBUG(dbgs() << "[genConvertFixToFloat] converting a floating point to a floating point\n");
      int startingBit = fix->getType()->getPrimitiveSizeInBits();
      int destinationBit = destt->getPrimitiveSizeInBits();

      if (startingBit == destinationBit) {
        // No casting is actually needed
        LLVM_DEBUG(assert((fix->dump(), destt->dump(), 1) && fix->getType() == destt && "Floating types having same bits but differents types"));
        assert(fix->getType() == destt && "Floating types having same bits but differents types");
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
        LLVM_DEBUG(assert((fix->dump(), destt->dump(), 1) && fix->getType() == destt && "Floating types having same bits but differents types"));
        assert(fix->getType() == destt && "Floating types having same bits but differents types");
        LLVM_DEBUG(dbgs() << "[genConvertFixToFloat] no casting needed.\n");
        return fix;
      } else if (startingBit < destinationBit) {
        // Extension needed
        return ConstantExpr::getCast(Instruction::FPExt, cst, destt);
      } else {
        // Truncation needed
        return ConstantExpr::getCast(Instruction::FPTrunc, cst, destt);
      }

      std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);
      // Always convert to double then to the destination type
      // No need to worry about efficiency, as everything will be constant folded
      Type *TmpTy = Type::getDoubleTy(cst->getContext());
      Constant *floattmp = scalarFixpt->isSigned()
                               ? ConstantExpr::getCast(Instruction::SIToFP, cst, TmpTy)
                               : ConstantExpr::getCast(Instruction::UIToFP, cst, TmpTy);
      double twoebits = pow(2.0, scalarFixpt->getFractionalBits());
      Constant *DblRes = ConstantFoldBinaryOpOperands(Instruction::FDiv, floattmp, ConstantFP::get(TmpTy, twoebits), *ModuleDL);
      assert(DblRes && "Constant folding failed...");
      LLVM_DEBUG(dbgs() << "ConstantFoldBinaryOpOperands returned " << *DblRes << "\n");
      Constant *Res = ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, destt, *ModuleDL);
      assert(Res && "Constant folding failed...");
      LLVM_DEBUG(dbgs() << "ConstantFoldCastInstruction returned " << *Res << "\n");
      return Res;
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
    IRBuilder<NoFolder> builder(ip);

    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);

    double twoebits = pow(2.0, scalarFixpt->getFractionalBits());
    if (twoebits == 1.0) {
      LLVM_DEBUG(dbgs() << "Optimizing conversion removing division by one!\n");
      Value *floattmp = scalarFixpt->isSigned() ? builder.CreateSIToFP(fix, destt) : builder.CreateUIToFP(fix, destt);
      return floattmp;
    }

    const fltSemantics &FltSema = destt->getFltSemantics();
    double MaxDest = APFloat::getLargest(FltSema).convertToDouble();
    double MaxSrc = pow(2.0, scalarFixpt->getBits());
    if (MaxDest < MaxSrc || MaxDest < twoebits) {
      LLVM_DEBUG(dbgs() << "fixToFloat: Extending " << *destt << " to float because source integer is too small\n");
      Type *TmpTy = Type::getFloatTy(fix->getContext());
      Value *floattmp = scalarFixpt->isSigned() ? builder.CreateSIToFP(fix, TmpTy) : builder.CreateUIToFP(fix, TmpTy);
      cpMetaData(floattmp, fix);
      return cpMetaData(
          builder.CreateFPTrunc(
            builder.CreateFDiv(floattmp,
              cpMetaData(ConstantFP::get(TmpTy, twoebits), fix)),
          destt),
        fix);
    } else {
      Value *floattmp = scalarFixpt->isSigned() ? builder.CreateSIToFP(fix, destt) : builder.CreateUIToFP(fix, destt);
      cpMetaData(floattmp, fix);
      return cpMetaData(builder.CreateFDiv(floattmp,cpMetaData(ConstantFP::get(destt, twoebits), fix)),
                        fix);
    }

  } else if (Constant *cst = dyn_cast<Constant>(fix)) {
    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);
    // Always convert to double then to the destination type
    // No need to worry about efficiency, as everything will be constant folded
    Type *TmpTy = Type::getDoubleTy(cst->getContext());
    Constant *floattmp = scalarFixpt->isSigned()
                             ? ConstantExpr::getCast(Instruction::SIToFP, cst, TmpTy)
                             : ConstantExpr::getCast(Instruction::UIToFP, cst, TmpTy);
    double twoebits = pow(2.0, scalarFixpt->getFractionalBits());
    Constant *DblRes = ConstantFoldBinaryOpOperands(Instruction::FDiv, floattmp, ConstantFP::get(TmpTy, twoebits), *ModuleDL);
    assert(DblRes && "ConstantFoldBinaryOpOperands failed...");
    LLVM_DEBUG(dbgs() << "ConstantFoldBinaryOpOperands returned " << *DblRes << "\n");
    Constant *Res = ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, destt, *ModuleDL);
    assert(Res && "Constant folding failed...");
    LLVM_DEBUG(dbgs() << "ConstantFoldCastInstruction returned " << *Res << "\n");
    return Res;
  }

  llvm_unreachable("unrecognized value type passed to genConvertFixToFloat");
  return nullptr;
}


Type *FloatToFixed::getLLVMFixedPointTypeForFloatType(const std::shared_ptr<TransparentType> &srcType, const std::shared_ptr<FixedPointType> &baset, bool *hasfloats)
{
  return baset->toTransparentType(srcType, hasfloats)->toLLVMType();
}


Type *FloatToFixed::getLLVMFixedPointTypeForFloatValue(Value *val)
{
  std::shared_ptr<FixedPointType> fpt = getFixpType(val);
  return getLLVMFixedPointTypeForFloatType(TaffoInfo::getInstance().getOrCreateTransparentType(*val), fpt);
}
