#include "ConversionPass.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TransparentType.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <cmath>
#include <memory>

using namespace llvm;
using namespace taffo;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

Value* ConversionError = (Value*) (&ConversionError);
Value* Unsupported = (Value*) (&Unsupported);

void FloatToFixed::performConversion(Module& m, std::vector<Value*>& q) {
  Logger& logger = log();

  for (auto iter = q.begin(); iter != q.end();) {
    Value* v = *iter;
    std::shared_ptr<FixedPointType> newType = getFixpType(v);

    auto indenter = logger.getIndenter();
    LLVM_DEBUG(
      logger << raw_ostream::Colors::BLUE << repeatString("▀▄▀▄", 10) << "[Perform conversion]"
             << repeatString("▄▀▄▀", 10) << raw_ostream::Colors::RESET << "\n";
      indenter.increaseIndent();
      logger.log("Value: ").logValueln(v);
      logger << "To convert: " << getConversionInfo(v)->noTypeConversion << "\n";
      logger << "Requested type: " << *newType << "\n";);

    Value* newv = convertSingleValue(m, v, newType);
    getConversionInfo(v)->fixpType = newType;
    if (newv) {
      convertedValues[v] = newv;
      LLVM_DEBUG(logger.log("Result type: ").logln(*newType, raw_ostream::Colors::CYAN));
    }

    if (newv && newv != ConversionError) {
      LLVM_DEBUG(log().log("Result:      ").logln(*newv, raw_ostream::Colors::CYAN));

      if (newv != v && isa<Instruction>(newv) && isa<Instruction>(v)) {
        Instruction* newinst = dyn_cast<Instruction>(newv);
        Instruction* oldinst = dyn_cast<Instruction>(v);
        newinst->setDebugLoc(oldinst->getDebugLoc());
      }
      if (newv != v) {
        std::shared_ptr<TransparentType> oldTransparentType = TaffoInfo::getInstance().getOrCreateTransparentType(*v);
        std::shared_ptr<TransparentType> newTransparentType;
        if (!newType->isInvalid())
          newTransparentType = newType->toTransparentType(oldTransparentType, nullptr);
        else
          newTransparentType = oldTransparentType;
        copyValueInfo(newv, v, newTransparentType);

        if (hasConversionInfo(newv)) {
          LLVM_DEBUG(
            logger.log("warning: output has valueInfo already from a previous conversion", raw_ostream::Colors::YELLOW)
            << " (type " << *getFixpType(newv) << ")\n");
          if (*getFixpType(newv) != *getFixpType(v)) {
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
    }
    else
      LLVM_DEBUG(logger.log("Result:      ").logln("CONVERSION ERROR", raw_ostream::Colors::RED));
    LLVM_DEBUG(logger << "\n");
    iter++;
  }
}

Value* FloatToFixed::createPlaceholder(Type* type, BasicBlock* where, StringRef name) {
  IRBuilder<NoFolder> builder(where, where->getFirstInsertionPt());
  AllocaInst* alloca = builder.CreateAlloca(type);
  return builder.CreateLoad(type, alloca, name);
}

/* also inserts the new value in the basic blocks, alongside the old one */
Value* FloatToFixed::convertSingleValue(Module& m, Value* val, std::shared_ptr<FixedPointType>& fixpt) {
  auto& taffoInfo = TaffoInfo::getInstance();
  Value* res = Unsupported;

  if (getConversionInfo(val)->isArgumentPlaceholder) {
    return matchOp(val);
  }
  else if (Constant* con = dyn_cast<Constant>(val)) {
    /* Since constants never change, there is never anything to substitute in them */
    if (!getConversionInfo(con)->noTypeConversion) {
      res = convertConstant(con, fixpt, TypeMatchPolicy::RangeOverHintMaxFrac);
      taffoInfo.setTransparentType(*res, TransparentTypeFactory::create(res->getType()));
    }
    else
      res = con;
  }
  else if (Instruction* instr = dyn_cast<Instruction>(val)) {
    res = convertInstruction(m, instr, fixpt);
    LLVM_DEBUG(
      // Check that all operands are valid
      if (User* user = dyn_cast<User>(res))
        for (auto& operand : user->operands())
          assert(operand.get() != nullptr);
      // Check that the transparent type of res is coherent
      if (!res->getType()->isPointerTy()) {
        auto type = taffoInfo.getTransparentType(*res);
        auto expectedType = TransparentTypeFactory::create(res->getType());
        assert(*type == *expectedType);
      });
  }
  else if (Argument* argument = dyn_cast<Argument>(val)) {
    if (getUnwrappedType(argument)->isFloatTy())
      res = translateOrMatchOperand(val, fixpt, nullptr);
    else
      res = val;
  }

  return res ? res : ConversionError;
}

/* do not use on pointer operands */
/* In iofixpt there is also the source type*/
Value* FloatToFixed::translateOrMatchOperand(
  Value* val, std::shared_ptr<FixedPointType>& iofixpt, Instruction* ip, TypeMatchPolicy typepol, bool wasHintForced) {
  auto& taffoInfo = TaffoInfo::getInstance();

  Logger& logger = log();
  auto indenter = logger.getIndenter();

  LLVM_DEBUG(
    logger.log("[TranslateOrMatchOperand of]", raw_ostream::Colors::BLACK) << *val << "\n";
    indenter.increaseIndent(););

  // FIXME: handle all the cases, we need more info about destination!
  if (typepol == TypeMatchPolicy::ForceHint) {
    std::shared_ptr<FixedPointType> origfixpt = iofixpt->clone();
    Value* tmp = translateOrMatchOperand(val, iofixpt, ip, TypeMatchPolicy::RangeOverHintMaxFrac, true);
    LLVM_DEBUG(log() << "forcing hint as requested!\n");
    return genConvertFixedToFixed(tmp,
                                  std::static_ptr_cast<FixedPointScalarType>(iofixpt),
                                  std::static_ptr_cast<FixedPointScalarType>(origfixpt),
                                  ip);
  }

  assert(val->getType()->getNumContainedTypes() == 0 && "val is not scalar");
  Value* res = convertedValues[val];
  if (res) { // this means it has been converted, but can also be a floating point!
    if (res == ConversionError)
      /* the value should have been converted but it hasn't; bail out */
      return nullptr;

    // The value has to be converted into a floating point value, convert it, full stop.
    if (iofixpt->isFloatingPoint()) {
      LLVM_DEBUG(log() << "converting converted value to floating point\n");
      return genConvertFixedToFixed(res,
                                    std::static_ptr_cast<FixedPointScalarType>(getFixpType(res)),
                                    std::static_ptr_cast<FixedPointScalarType>(iofixpt),
                                    ip);
    }

    // Converting Floating point to whatever
    if (getFixpType(res)->isFloatingPoint()) {
      LLVM_DEBUG(
        Logger& logger = log();
        logger << "converting floating point to whatever\n";
        logger << "Is floating, calling subroutine, " << *getFixpType(res) << " --> " << *iofixpt << "\n";
        logger << "This value will be converted to fixpoint: ";
        logger.logln(val););
      // The conversion is not forced to a specific type, so let's choose the best type that contains our value
      // Which might NOT be the destination type
      // In fact, if we use the destination type, we risk screwing everything up causing overflows
      if (iofixpt->isFixedPoint()) {
        if (!wasHintForced) {
          // In this case we try to choose the best fixed point type!
          std::shared_ptr<ValueInfo> valueInfo = TaffoInfo::getInstance().getValueInfo(*val);
          std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast_or_null<ScalarInfo>(valueInfo);
          if (!scalarInfo || !scalarInfo->range) {
            LLVM_DEBUG(log() << "no metadata found, rolling with the suggested type and hoping for the best!\n");
          }
          else {
            associateFixFormat(scalarInfo, iofixpt);
            LLVM_DEBUG(log() << "we have a new fixed point suggested type: " << *iofixpt << "\n";);
          }
        }
        else {
          LLVM_DEBUG(log() << "not associating better fixed point data type as the datatype was originally forced!\n";);
        }
      }

      return genConvertFixedToFixed(res,
                                    std::static_ptr_cast<FixedPointScalarType>(getFixpType(res)),
                                    std::static_ptr_cast<FixedPointScalarType>(iofixpt),
                                    ip);
    }

    if (!getConversionInfo(val)->noTypeConversion) {
      /* the value has been successfully converted to fixed point in a previous step */
      iofixpt = getFixpType(res);
      LLVM_DEBUG(
        logger << "value has been converted in the past to: \n";
        indenter.increaseIndent();
        logger << "Value: " << *res << "\n";
        logger << "FixType: " << *iofixpt << "\n";
        indenter.decreaseIndent(););
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

  LLVM_DEBUG(log() << "translateOrMatchOperand: non-converted value, converting now\n");

  /* try the easy cases first
   *   this is essentially duplicated from genConvertFloatToFix because once we
   * enter that function iofixpt cannot change anymore
   *   in other words, by duplicating this logic here we potentially avoid a loss
   * of range if the suggested iofixpt is not enough for the value */
  if (Constant* c = dyn_cast<Constant>(val)) {
    Value* res = convertConstant(c, iofixpt, typepol);
    taffoInfo.setTransparentType(*res, TransparentTypeFactory::create(res->getType()));
    return res;
  }
  else if (iofixpt->isFixedPoint()) {
    // Only try to exclude conversion if we are trying to convert a float variable that has been converted
    if (SIToFPInst* instr = dyn_cast<SIToFPInst>(val)) {
      Value* intparam = instr->getOperand(0);
      iofixpt = std::make_shared<FixedPointScalarType>(intparam->getType(), true);
      return intparam;
    }
    else if (UIToFPInst* instr = dyn_cast<UIToFPInst>(val)) {
      Value* intparam = instr->getOperand(0);
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
        FixedPointInfo fpt =
          fixedPointTypeFromRange(*ii->range, &err, std::static_ptr_cast<FixedPointScalarType>(iofixpt)->getBits());
        if (err != FixedPointTypeGenError::InvalidRange)
          iofixpt = std::make_shared<FixedPointScalarType>(&fpt);
      }
    }
  }

  return genConvertFloatToFix(val, std::static_ptr_cast<FixedPointScalarType>(iofixpt), ip);
}

bool FloatToFixed::associateFixFormat(const std::shared_ptr<ScalarInfo>& II, std::shared_ptr<FixedPointType>& iofixpt) {
  Range* rng = II->range.get();
  assert(rng && "No range info!");

  FixedPointTypeGenError fpgerr;
  // Using default parameters of DTA
  FixedPointInfo res = fixedPointTypeFromRange(*rng, &fpgerr, 32, 3, 64, 32);
  assert(fpgerr != FixedPointTypeGenError::InvalidRange && "Cannot assign a fixed point type!");

  iofixpt = std::make_shared<FixedPointScalarType>(res.isSigned(), res.getBits(), res.getFractionalBits());

  return true;
}

// TODO: rewrite this mess!
Value*
FloatToFixed::genConvertFloatToFix(Value* flt, const std::shared_ptr<FixedPointScalarType>& fixpt, Instruction* ip) {
  auto& taffoInfo = TaffoInfo::getInstance();
  assert(flt->getType()->isFloatingPointTy() && "genConvertFloatToFixed called on a non-float scalar");
  LLVM_DEBUG(
    Logger& logger = log();
    logger << "Called floatToFixed with src ";
    logger.log(flt->getType(), raw_ostream::Colors::CYAN);
    logger.log(" to ").logln(*fixpt, raw_ostream::Colors::CYAN););

  if (Constant* c = dyn_cast<Constant>(flt)) {
    std::shared_ptr<FixedPointType> fixptcopy = fixpt->clone();
    Value* res = convertConstant(c, fixptcopy, TypeMatchPolicy::ForceHint);
    taffoInfo.setTransparentType(*res, TransparentTypeFactory::create(res->getType()));
    assert(*fixptcopy == *fixpt && "why is there a pointer here?");
    return res;
  }

  if (Instruction* i = dyn_cast<Instruction>(flt)) {
    if (!ip)
      ip = getFirstInsertionPointAfter(i);
  }
  else if (Argument* arg = dyn_cast<Argument>(flt)) {
    Function* fun = arg->getParent();
    BasicBlock& firstbb = fun->getEntryBlock();
    ip = &(*firstbb.getFirstInsertionPt());
  }
  assert(ip && "ip is mandatory if not passing an instruction/constant value");

  FloatToFixCount++;
  FloatToFixWeight += std::pow(2, std::min((int) (sizeof(int) * 8 - 1), this->getLoopNestingLevelOfValue(flt)));

  IRBuilder<NoFolder> builder(ip);
  Type* SrcTy = flt->getType();
  Type* destt = getLLVMFixedPointTypeForFloatType(TaffoInfo::getInstance().getOrCreateTransparentType(*flt), fixpt);

  /* insert new instructions before ip */
  if (!destt->isFloatingPointTy()) {
    if (SIToFPInst* instr = dyn_cast<SIToFPInst>(flt)) {
      Value* intparam = instr->getOperand(0);
      return copyValueInfo(
        builder.CreateShl(copyValueInfo(builder.CreateIntCast(intparam, destt, true), flt), fixpt->getFractionalBits()),
        flt);
    }
    else if (UIToFPInst* instr = dyn_cast<UIToFPInst>(flt)) {
      Value* intparam = instr->getOperand(0);
      return copyValueInfo(builder.CreateShl(copyValueInfo(builder.CreateIntCast(intparam, destt, false), flt),
                                             fixpt->getFractionalBits()),
                           flt);
    }
    else {
      double twoebits = pow(2.0, fixpt->getFractionalBits());
      const fltSemantics& FltSema = SrcTy->getFltSemantics();
      double MaxSrc = APFloat::getLargest(FltSema).convertToDouble();
      double MaxDest = pow(2.0, fixpt->getBits());
      Value* SanitizedFloat = flt;
      if (MaxSrc < MaxDest || MaxSrc < twoebits) {
        LLVM_DEBUG(log() << "floatToFixed: Extending " << *SrcTy << " to float because dest integer is too large\n");
        SanitizedFloat = builder.CreateFPCast(flt, Type::getFloatTy(flt->getContext()));
        copyValueInfo(SanitizedFloat, flt);
      }
      Type* IntermType = SanitizedFloat->getType();
      Value* interm = copyValueInfo(
        builder.CreateFMul(copyValueInfo(ConstantFP::get(IntermType, twoebits), SanitizedFloat), SanitizedFloat),
        SanitizedFloat);
      if (fixpt->isSigned())
        return copyValueInfo(builder.CreateFPToSI(interm, destt), SanitizedFloat);
      else
        return copyValueInfo(builder.CreateFPToUI(interm, destt), SanitizedFloat);
    }
  }
  else {
    LLVM_DEBUG(log() << "[genConvertFloatToFix] converting a floating point to a floating point\n");
    int startingBit = flt->getType()->getPrimitiveSizeInBits();
    int destinationBit = destt->getPrimitiveSizeInBits();
    if (startingBit == destinationBit) {
      // No casting is actually needed
      LLVM_DEBUG(assert((flt->dump(), destt->dump(), 1) && flt->getType() == destt
                        && "Floating types having same bits but differents types"));
      LLVM_DEBUG(log() << "[genConvertFloatToFix] no casting needed.\n");
      assert(flt->getType() == destt && "Floating types having same bits but differents types");
      return flt;
    }
    else if (startingBit < destinationBit) {
      // Extension needed
      return copyValueInfo(builder.CreateFPExt(flt, destt), flt);
    }
    else {
      // Truncation needed
      return copyValueInfo(builder.CreateFPTrunc(flt, destt), flt);
    }
  }
}

// TODO: rewrite this mess!
Value* FloatToFixed::genConvertFixedToFixed(Value* fix,
                                            const std::shared_ptr<FixedPointScalarType>& srcFixedType,
                                            const std::shared_ptr<FixedPointScalarType>& dstFixedType,
                                            Instruction* ip) {
  auto& taffoInfo = TaffoInfo::getInstance();
  if (*srcFixedType == *dstFixedType)
    return fix;

  LLVM_DEBUG(
    Logger& logger = log();
    logger.log("Called fixedToFixed with src ");
    logger.log(*srcFixedType, llvm::raw_ostream::Colors::CYAN);
    logger.log(" to dst ");
    logger.logln(*dstFixedType, llvm::raw_ostream::Colors::CYAN););

  Instruction* fixInst = dyn_cast<Instruction>(fix);
  if (!ip && fixInst)
    ip = getFirstInsertionPointAfter(fixInst);
  assert(ip && "ip required when converted value not an instruction");

  std::shared_ptr<TransparentType> srcType = taffoInfo.getTransparentType(*fix);
  std::shared_ptr<TransparentType> dstType = dstFixedType->toTransparentType(srcType);
  Type* srcLLVMType = srcType->toLLVMType();
  Type* dstLLVMType = dstType->toLLVMType();

  // Source and destination are both float
  if (srcType->isFloatingPointType() && dstType->isFloatingPointType()) {
    IRBuilder<NoFolder> builder(ip);

    LLVM_DEBUG(log() << "[genConvertFloatToFix] converting a floating point to a floating point\n");
    int startingBit = srcLLVMType->getPrimitiveSizeInBits();
    int destinationBit = dstLLVMType->getPrimitiveSizeInBits();

    if (startingBit == destinationBit) {
      // No casting is actually needed
      assert(*srcType == *dstType && "Floating types having same bits but differents types");
      LLVM_DEBUG(log() << "[genConvertFloatToFix] no casting needed.\n");
      return fix;
    }
    else if (startingBit < destinationBit) {
      // Extension needed
      return copyValueInfo(builder.CreateFPExt(fix, dstLLVMType), fix);
    }
    else {
      // Truncation needed
      return copyValueInfo(builder.CreateFPTrunc(fix, dstLLVMType), fix);
    }
  }

  /*We should never have these case in general, but when using mixed precision this can (and will) happen*/
  if (srcFixedType->isFloatingPoint() && dstFixedType->isFixedPoint())
    return genConvertFloatToFix(fix, dstFixedType, ip);

  if (srcFixedType->isFixedPoint() && dstFixedType->isFloatingPoint())
    return genConvertFixToFloat(fix, srcFixedType, dstType);

  assert(srcLLVMType->isSingleValueType() && "cannot be a struct or an array");
  assert(srcLLVMType->isIntegerTy() && "src must be an integer");

  IRBuilder<NoFolder> builder(ip);

  auto genSizeChange = [&](Value* fix) -> Value* {
    if (srcFixedType->isSigned())
      return copyValueInfo(builder.CreateSExtOrTrunc(fix, dstLLVMType), fix);
    else
      return copyValueInfo(builder.CreateZExtOrTrunc(fix, dstLLVMType), fix);
  };

  auto genPointMovement = [&](Value* fix) -> Value* {
    int deltaBits = dstFixedType->getFractionalBits() - srcFixedType->getFractionalBits();
    if (deltaBits > 0) {
      return copyValueInfo(builder.CreateShl(fix, deltaBits), fix);
    }
    else if (deltaBits < 0) {
      if (srcFixedType->isSigned())
        return copyValueInfo(builder.CreateAShr(fix, -deltaBits), fix);
      else
        return copyValueInfo(builder.CreateLShr(fix, -deltaBits), fix);
    }
    return fix;
  };

  if (dstFixedType->getBits() > srcFixedType->getBits())
    return genPointMovement(genSizeChange(fix));
  return genSizeChange(genPointMovement(fix));
}

// TODO: rewrite this mess!
Value* FloatToFixed::genConvertFixToFloat(Value* fixValue,
                                          const std::shared_ptr<FixedPointType>& fixpt,
                                          const std::shared_ptr<TransparentType>& dstType) {
  Logger& logger = log();
  auto& taffoInfo = TaffoInfo::getInstance();
  Type* dstLLVMType = dstType->toLLVMType();

  LLVM_DEBUG(
    logger << "******** trace: genConvertFixToFloat ";
    logger.logValue(fixValue);
    logger << " -> ";
    logger.logln(dstType););

  auto fixValueType = taffoInfo.getTransparentType(*fixValue);
  if (fixValueType->isFloatingPointType()) {
    if (isa<Instruction>(fixValue) || isa<Argument>(fixValue)) {
      Instruction* ip = nullptr;
      if (Instruction* i = dyn_cast<Instruction>(fixValue))
        ip = getFirstInsertionPointAfter(i);
      else if (Argument* arg = dyn_cast<Argument>(fixValue))
        ip = &(*(arg->getParent()->getEntryBlock().getFirstInsertionPt()));
      IRBuilder<NoFolder> builder(ip);

      LLVM_DEBUG(logger << "[genConvertFixToFloat] converting a floating point to a floating point\n");
      int startingBit = fixValueType->toLLVMType()->getPrimitiveSizeInBits();
      int destinationBit = dstLLVMType->getPrimitiveSizeInBits();
      assert(!fixValueType->isPointerType() && !dstType->isPointerType()
             && "In FixToFloat src and dst cannot be pointers");

      if (startingBit == destinationBit) {
        // No casting is actually needed
        assert(*fixValueType == *dstType && "Floating types having same bits but differents types");
        LLVM_DEBUG(logger << "[genConvertFixToFloat] no casting needed.\n");
        return fixValue;
      }
      else if (startingBit < destinationBit) {
        // Extension needed
        return copyValueInfo(builder.CreateFPExt(fixValue, dstLLVMType), fixValue);
      }
      else {
        // Truncation needed
        return copyValueInfo(builder.CreateFPTrunc(fixValue, dstLLVMType), fixValue);
      }
    }
    else if (Constant* cst = dyn_cast<Constant>(fixValue)) {

      int startingBit = fixValueType->toLLVMType()->getPrimitiveSizeInBits();
      int destinationBit = dstLLVMType->getPrimitiveSizeInBits();

      if (startingBit == destinationBit) {
        // No casting is actually needed
        assert(*fixValueType == *dstType && "Floating types having same bits but differents types");
        LLVM_DEBUG(logger << "[genConvertFixToFloat] no casting needed.\n");
        return fixValue;
      }
      else if (startingBit < destinationBit) {
        // Extension needed
        return ConstantExpr::getCast(Instruction::FPExt, cst, dstLLVMType);
      }
      else {
        // Truncation needed
        return ConstantExpr::getCast(Instruction::FPTrunc, cst, dstLLVMType);
      }

      std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);
      // Always convert to double then to the destination type
      // No need to worry about efficiency, as everything will be constant folded
      Type* TmpTy = Type::getDoubleTy(cst->getContext());
      Constant* floattmp = scalarFixpt->isSigned() ? ConstantExpr::getCast(Instruction::SIToFP, cst, TmpTy)
                                                   : ConstantExpr::getCast(Instruction::UIToFP, cst, TmpTy);
      double twoebits = pow(2.0, scalarFixpt->getFractionalBits());
      Constant* DblRes =
        ConstantFoldBinaryOpOperands(Instruction::FDiv, floattmp, ConstantFP::get(TmpTy, twoebits), *ModuleDL);
      assert(DblRes && "Constant folding failed...");
      LLVM_DEBUG(log() << "ConstantFoldBinaryOpOperands returned " << *DblRes << "\n");
      Constant* Res = ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, dstLLVMType, *ModuleDL);
      assert(Res && "Constant folding failed...");
      LLVM_DEBUG(log() << "ConstantFoldCastInstruction returned " << *Res << "\n");
      return Res;
    }
  }

  if (!fixValue->getType()->isIntegerTy()) {
    LLVM_DEBUG(
      errs() << "can't wrap-convert to flt non integer value ";
      fixValue->print(errs());
      errs() << "\n");
    return nullptr;
  }

  FixToFloatCount++;
  FixToFloatWeight += std::pow(2, std::min((int) (sizeof(int) * 8 - 1), this->getLoopNestingLevelOfValue(fixValue)));

  if (isa<Instruction>(fixValue) || isa<Argument>(fixValue)) {
    Instruction* ip = nullptr;
    if (Instruction* i = dyn_cast<Instruction>(fixValue))
      ip = getFirstInsertionPointAfter(i);
    else if (Argument* arg = dyn_cast<Argument>(fixValue))
      ip = &(*(arg->getParent()->getEntryBlock().getFirstInsertionPt()));
    IRBuilder<NoFolder> builder(ip);

    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);

    double twoebits = pow(2.0, scalarFixpt->getFractionalBits());
    if (twoebits == 1.0) {
      LLVM_DEBUG(log() << "Optimizing conversion removing division by one!\n");
      Value* floattmp = scalarFixpt->isSigned() ? builder.CreateSIToFP(fixValue, dstLLVMType)
                                                : builder.CreateUIToFP(fixValue, dstLLVMType);
      return floattmp;
    }

    const fltSemantics& FltSema = dstLLVMType->getFltSemantics();
    double MaxDest = APFloat::getLargest(FltSema).convertToDouble();
    double MaxSrc = pow(2.0, scalarFixpt->getBits());
    if (MaxDest < MaxSrc || MaxDest < twoebits) {
      LLVM_DEBUG(log() << "fixToFloat: Extending " << *dstType << " to float because source integer is too small\n");
      Type* TmpTy = Type::getFloatTy(fixValue->getContext());
      Value* floattmp =
        scalarFixpt->isSigned() ? builder.CreateSIToFP(fixValue, TmpTy) : builder.CreateUIToFP(fixValue, TmpTy);
      copyValueInfo(floattmp, fixValue);
      return copyValueInfo(
        builder.CreateFPTrunc(builder.CreateFDiv(floattmp, copyValueInfo(ConstantFP::get(TmpTy, twoebits), fixValue)),
                              dstLLVMType),
        fixValue);
    }
    else {
      Value* floattmp = scalarFixpt->isSigned() ? builder.CreateSIToFP(fixValue, dstLLVMType)
                                                : builder.CreateUIToFP(fixValue, dstLLVMType);
      copyValueInfo(floattmp, fixValue);
      return copyValueInfo(
        builder.CreateFDiv(floattmp, copyValueInfo(ConstantFP::get(dstLLVMType, twoebits), fixValue)), fixValue);
    }
  }
  else if (Constant* cst = dyn_cast<Constant>(fixValue)) {
    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);
    // Always convert to double then to the destination type
    // No need to worry about efficiency, as everything will be constant folded
    Type* TmpTy = Type::getDoubleTy(cst->getContext());
    Constant* floattmp = scalarFixpt->isSigned() ? ConstantExpr::getCast(Instruction::SIToFP, cst, TmpTy)
                                                 : ConstantExpr::getCast(Instruction::UIToFP, cst, TmpTy);
    double twoebits = pow(2.0, scalarFixpt->getFractionalBits());
    Constant* DblRes =
      ConstantFoldBinaryOpOperands(Instruction::FDiv, floattmp, ConstantFP::get(TmpTy, twoebits), *ModuleDL);
    assert(DblRes && "ConstantFoldBinaryOpOperands failed...");
    LLVM_DEBUG(log() << "ConstantFoldBinaryOpOperands returned " << *DblRes << "\n");
    Constant* Res = ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, dstLLVMType, *ModuleDL);
    assert(Res && "Constant folding failed...");
    LLVM_DEBUG(log() << "ConstantFoldCastInstruction returned " << *Res << "\n");
    return Res;
  }

  llvm_unreachable("unrecognized value type passed to genConvertFixToFloat");
  return nullptr;
}

Type* FloatToFixed::getLLVMFixedPointTypeForFloatType(const std::shared_ptr<TransparentType>& srcType,
                                                      const std::shared_ptr<FixedPointType>& baset,
                                                      bool* hasfloats) {
  return baset->toTransparentType(srcType, hasfloats)->toLLVMType();
}

Type* FloatToFixed::getLLVMFixedPointTypeForFloatValue(Value* val) {
  std::shared_ptr<FixedPointType> fpt = getFixpType(val);
  return getLLVMFixedPointTypeForFloatType(TaffoInfo::getInstance().getOrCreateTransparentType(*val), fpt);
}
