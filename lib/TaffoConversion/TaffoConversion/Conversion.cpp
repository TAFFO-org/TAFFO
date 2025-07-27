#include "ConversionPass.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <cmath>
#include <memory>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

Value* ConversionError = (Value*) &ConversionError;
Value* Unsupported = (Value*) &Unsupported;

void ConversionPass::performConversion(Module& m, std::vector<Value*>& queue) {
  Logger& logger = log();

  for (auto iter = queue.begin(); iter != queue.end();) {
    Value* value = *iter;
    std::shared_ptr<FixedPointType> newType = getFixpType(value);

    auto indenter = logger.getIndenter();
    LLVM_DEBUG(
      logger << Logger::Blue << repeatString("▀▄▀▄", 10) << "[Perform conversion]" << repeatString("▄▀▄▀", 10)
             << Logger::Reset << "\n";
      logger.log("[Value] ", Logger::Bold).logValueln(value);
      logger << "to convert: " << !getConversionInfo(value)->isConversionDisabled << "\n";
      logger << "requested type: " << *newType << "\n";);

    Value* newValue = convertSingleValue(m, value, newType);
    getConversionInfo(value)->fixpType = newType;
    if (newValue) {
      convertedValues[value] = newValue;
      LLVM_DEBUG(logger.log("result type: ", Logger::Green).logln(*newType));
    }

    if (newValue && newValue != ConversionError) {
      LLVM_DEBUG(log().log("result:      ", Logger::Green).logln(*newValue));

      if (newValue != value && isa<Instruction>(newValue) && isa<Instruction>(value)) {
        auto* newInst = dyn_cast<Instruction>(newValue);
        auto* oldInst = dyn_cast<Instruction>(value);
        newInst->setDebugLoc(oldInst->getDebugLoc());
      }
      if (newValue != value) {
        std::shared_ptr<TransparentType> oldTransparentType = taffoInfo.getTransparentType(*value);
        std::shared_ptr<TransparentType> newTransparentType;
        if (!newType->isInvalid() && newType->isFixedPoint() && !isa<CmpInst>(newValue))
          newTransparentType = newType->toTransparentType(oldTransparentType, nullptr);
        else
          newTransparentType = oldTransparentType;
        copyValueInfo(newValue, value, newTransparentType);

        LLVM_DEBUG(
          // Check that the transparent type of newv is coherent
          if (!newValue->getType()->isPointerTy()) {
            auto type = taffoInfo.getTransparentType(*newValue);
            auto expectedType = TransparentTypeFactory::create(newValue->getType());
            assert(*type == *expectedType);
          });

        if (hasConversionInfo(newValue)) {
          LLVM_DEBUG(logger.log("warning: output has valueInfo already from a previous conversion", Logger::Yellow)
                     << " (type " << *getFixpType(newValue) << ")\n");
          if (*getFixpType(newValue) != *getFixpType(value)) {
            logger.logln("FATAL ERROR: SAME VALUE INSTANCE HAS TWO DIFFERENT SEMANTICS!", Logger::Red);
            logger.log("New type: ", Logger::Red);
            logger.log(getFixpType(newValue), Logger::Red);
            logger.log(", old type: ", Logger::Red);
            logger.logln(getFixpType(value), Logger::Red);
            abort();
          }
        }
        else
          *newConversionInfo(newValue) = *getConversionInfo(value);
      }
    }
    else
      LLVM_DEBUG(logger.log("Result:      ").logln("CONVERSION ERROR", Logger::Red));
    LLVM_DEBUG(logger << "\n");
    iter++;
  }
}

Value* ConversionPass::createPlaceholder(Type* type, BasicBlock* where, StringRef name) {
  IRBuilder<NoFolder> builder(where, where->getFirstInsertionPt());
  AllocaInst* alloca = builder.CreateAlloca(type);
  return builder.CreateLoad(type, alloca, name);
}

/* also inserts the new value in the basic blocks, alongside the old one */
Value* ConversionPass::convertSingleValue(Module& m, Value* val, std::shared_ptr<FixedPointType>& fixpt) {
  Value* res = Unsupported;
  if (getConversionInfo(val)->isArgumentPlaceholder)
    return matchOp(val);
  if (auto* con = dyn_cast<Constant>(val)) {
    /* Since constants never change, there is never anything to substitute in them */
    if (!getConversionInfo(con)->isConversionDisabled) {
      res = convertConstant(con, fixpt, TypeMatchPolicy::RangeOverHintMaxFrac);
      taffoInfo.setTransparentType(*res, TransparentTypeFactory::create(res->getType()));
    }
    else
      res = con;
  }
  else if (auto* instr = dyn_cast<Instruction>(val)) {
    res = convertInstruction(m, instr, fixpt);
    LLVM_DEBUG(
      // Check that all operands are valid
      if (User* user = dyn_cast<User>(res))
        for (auto& operand : user->operands())
          assert(operand.get() != nullptr););
  }
  else if (auto* argument = dyn_cast<Argument>(val)) {
    if (getFullyUnwrappedType(argument)->isFloatingPointTy())
      res = translateOrMatchOperand(val, fixpt, nullptr);
    else
      res = val;
  }

  return res ? res : ConversionError;
}

/* do not use on pointer operands */
/* In iofixpt there is also the source type*/
Value* ConversionPass::translateOrMatchOperand(Value* value,
                                               std::shared_ptr<FixedPointType>& iofixpt,
                                               Instruction* insertionPoint,
                                               TypeMatchPolicy policy,
                                               bool wasHintForced) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();

  LLVM_DEBUG(
    logger.log("[TranslateOrMatchOperand of] ", Logger::Bold) << *value << "\n";
    indenter.increaseIndent(););

  // FIXME: handle all the cases, we need more info about destination!
  if (policy == TypeMatchPolicy::ForceHint) {
    std::shared_ptr<FixedPointType> origfixpt = iofixpt->clone();
    Value* tmp = translateOrMatchOperand(value, iofixpt, insertionPoint, TypeMatchPolicy::RangeOverHintMaxFrac, true);
    LLVM_DEBUG(log() << "forcing hint as requested!\n");
    return genConvertFixedToFixed(tmp,
                                  std::static_ptr_cast<FixedPointScalarType>(iofixpt),
                                  std::static_ptr_cast<FixedPointScalarType>(origfixpt),
                                  insertionPoint);
  }

  assert(value->getType()->getNumContainedTypes() == 0 && "val is not scalar");
  Value* res = matchOp(value);
  std::shared_ptr<TransparentType> valueType = taffoInfo.getTransparentType(*value);
  if (res != value) { // this means it has been converted, but can also be a floating point!
    if (res == ConversionError)
      /* the value should have been converted but it hasn't; bail out */
      return nullptr;

    // The value has to be converted into a floating point value, convert it, full stop.
    if (iofixpt->isFloatingPoint()) {
      LLVM_DEBUG(log() << "converting converted value to floating point\n");
      return genConvertFixedToFixed(res,
                                    std::static_ptr_cast<FixedPointScalarType>(getFixpType(res)),
                                    std::static_ptr_cast<FixedPointScalarType>(iofixpt),
                                    insertionPoint);
    }

    // Converting Floating point to whatever
    if (getFixpType(res)->isFloatingPoint()) {
      LLVM_DEBUG(
        logger << "converting floating point to whatever\n";
        logger << "Is floating, calling subroutine, " << *getFixpType(res) << " --> " << *iofixpt << "\n";
        logger << "This value will be converted to fixpoint: ";
        logger.logln(value););
      // The conversion is not forced to a specific type, so let's choose the best type that contains our value
      // Which might NOT be the destination type
      // In fact, if we use the destination type, we risk screwing everything up causing overflows
      if (iofixpt->isFixedPoint()) {
        if (!wasHintForced) {
          // In this case we try to choose the best fixed point type!
          std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*value);
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
                                    insertionPoint);
    }

    if (!getConversionInfo(value)->isConversionDisabled) {
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
    value = res;
  }
  else if (std::static_ptr_cast<FixedPointScalarType>(iofixpt)->isInvalid()
           || *valueType == *iofixpt->toTransparentType(valueType)) {
    // value doesn't need conversion
    return res;
  }

  assert(value->getType()->isFloatingPointTy());

  LLVM_DEBUG(log() << "translateOrMatchOperand: non-converted value, converting now\n");

  /* try the easy cases first
   *   this is essentially duplicated from genConvertFloatToFix because once we
   * enter that function iofixpt cannot change anymore
   *   in other words, by duplicating this logic here we potentially avoid a loss
   * of range if the suggested iofixpt is not enough for the value */
  if (auto* c = dyn_cast<Constant>(value)) {
    Value* res = convertConstant(c, iofixpt, policy);
    taffoInfo.setTransparentType(*res, TransparentTypeFactory::create(res->getType()));
    return res;
  }
  else if (iofixpt->isFixedPoint()) {
    // Only try to exclude conversion if we are trying to convert a float variable that has been converted
    if (auto* inst = dyn_cast<SIToFPInst>(value)) {
      Value* intparam = inst->getOperand(0);
      iofixpt = std::make_shared<FixedPointScalarType>(intparam->getType(), true);
      return intparam;
    }
    else if (auto* inst = dyn_cast<UIToFPInst>(value)) {
      Value* intparam = inst->getOperand(0);
      iofixpt = std::make_shared<FixedPointScalarType>(intparam->getType(), true);
      return intparam;
    }
  }

  /* not an easy case; check if the value has a range metadata
   * from VRA before giving up and using the suggested type */
  // Do this hack only if the final wanted type is a fixed point, otherwise we can go ahead
  if (iofixpt->isFixedPoint() && taffoInfo.hasValueInfo(*value)) {
    std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*value);
    if (std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo)) {
      if (scalarInfo->range) {
        FixedPointTypeGenError err;
        FixedPointInfo fpt = fixedPointTypeFromRange(
          *scalarInfo->range, &err, std::static_ptr_cast<FixedPointScalarType>(iofixpt)->getBits());
        if (err != FixedPointTypeGenError::InvalidRange)
          iofixpt = std::make_shared<FixedPointScalarType>(&fpt);
      }
    }
  }

  return genConvertFloatToFix(value, std::static_ptr_cast<FixedPointScalarType>(iofixpt), insertionPoint);
}

bool ConversionPass::associateFixFormat(const std::shared_ptr<ScalarInfo>& II,
                                        std::shared_ptr<FixedPointType>& iofixpt) {
  Range* range = II->range.get();
  assert(range && "No range info!");

  FixedPointTypeGenError genError;
  // Using default parameters of DTA
  FixedPointInfo res = fixedPointTypeFromRange(*range, &genError, 32, 3, 64, 32);
  assert(genError != FixedPointTypeGenError::InvalidRange && "Cannot assign a fixed point type!");

  iofixpt = std::make_shared<FixedPointScalarType>(res.isSigned(), res.getBits(), res.getFractionalBits());

  return true;
}

// TODO: rewrite this mess!
Value*
ConversionPass::genConvertFloatToFix(Value* flt, const std::shared_ptr<FixedPointScalarType>& fixpt, Instruction* ip) {
  assert(flt->getType()->isFloatingPointTy() && "genConvertFloatToFixed called on a non-float scalar");
  LLVM_DEBUG(
    Logger& logger = log();
    logger << "called floatToFixed with src ";
    logger.log(flt->getType(), Logger::Cyan);
    logger.log(" to ").logln(*fixpt, Logger::Cyan););

  if (auto* c = dyn_cast<Constant>(flt)) {
    std::shared_ptr<FixedPointType> fixptcopy = fixpt->clone();
    Value* res = convertConstant(c, fixptcopy, TypeMatchPolicy::ForceHint);
    taffoInfo.setTransparentType(*res, TransparentTypeFactory::create(res->getType()));
    assert(*fixptcopy == *fixpt && "why is there a pointer here?");
    return res;
  }

  if (auto* i = dyn_cast<Instruction>(flt)) {
    if (!ip)
      ip = getFirstInsertionPointAfter(i);
  }
  else if (auto* arg = dyn_cast<Argument>(flt)) {
    Function* fun = arg->getParent();
    BasicBlock& firstbb = fun->getEntryBlock();
    ip = &(*firstbb.getFirstInsertionPt());
  }
  assert(ip && "ip is mandatory if not passing an instruction/constant value");

  FloatToFixCount++;

  IRBuilder<NoFolder> builder(ip);
  Type* SrcTy = flt->getType();
  Type* destt = getLLVMFixedPointTypeForFloatType(taffoInfo.getOrCreateTransparentType(*flt), fixpt);

  /* insert new instructions before ip */
  if (!destt->isFloatingPointTy()) {
    if (auto* instr = dyn_cast<SIToFPInst>(flt)) {
      Value* intparam = instr->getOperand(0);
      return copyValueInfo(
        builder.CreateShl(copyValueInfo(builder.CreateIntCast(intparam, destt, true), flt), fixpt->getFractionalBits()),
        flt);
    }
    else if (auto* instr = dyn_cast<UIToFPInst>(flt)) {
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
Value* ConversionPass::genConvertFixedToFixed(Value* fix,
                                              const std::shared_ptr<FixedPointScalarType>& srcFixedType,
                                              const std::shared_ptr<FixedPointScalarType>& dstFixedType,
                                              Instruction* ip) {
  if (*srcFixedType == *dstFixedType)
    return fix;

  LLVM_DEBUG(
    Logger& logger = log();
    logger.log("called fixedToFixed with src ");
    logger.log(*srcFixedType, Logger::Cyan);
    logger.log(" to dst ");
    logger.logln(*dstFixedType, Logger::Cyan););

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
Value* ConversionPass::genConvertFixToFloat(Value* fixValue,
                                            const std::shared_ptr<FixedPointType>& fixpt,
                                            const std::shared_ptr<TransparentType>& dstType) {
  Logger& logger = log();
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
        ConstantFoldBinaryOpOperands(Instruction::FDiv, floattmp, ConstantFP::get(TmpTy, twoebits), *dataLayout);
      assert(DblRes && "Constant folding failed...");
      LLVM_DEBUG(log() << "ConstantFoldBinaryOpOperands returned " << *DblRes << "\n");
      Constant* Res = ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, dstLLVMType, *dataLayout);
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
      ConstantFoldBinaryOpOperands(Instruction::FDiv, floattmp, ConstantFP::get(TmpTy, twoebits), *dataLayout);
    assert(DblRes && "ConstantFoldBinaryOpOperands failed...");
    LLVM_DEBUG(log() << "ConstantFoldBinaryOpOperands returned " << *DblRes << "\n");
    Constant* Res = ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, dstLLVMType, *dataLayout);
    assert(Res && "Constant folding failed...");
    LLVM_DEBUG(log() << "ConstantFoldCastInstruction returned " << *Res << "\n");
    return Res;
  }

  llvm_unreachable("unrecognized value type passed to genConvertFixToFloat");
  return nullptr;
}

Type* ConversionPass::getLLVMFixedPointTypeForFloatType(const std::shared_ptr<TransparentType>& srcType,
                                                        const std::shared_ptr<FixedPointType>& baset,
                                                        bool* hasfloats) {
  return baset->toTransparentType(srcType, hasfloats)->toLLVMType();
}

Type* ConversionPass::getLLVMFixedPointTypeForFloatValue(Value* val) {
  std::shared_ptr<FixedPointType> fpt = getFixpType(val);
  return getLLVMFixedPointTypeForFloatType(taffoInfo.getOrCreateTransparentType(*val), fpt);
}
