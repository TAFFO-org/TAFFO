#include "ConversionPass.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>

#include <cassert>
#include <cmath>
#include <memory>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conv"

Value* unsupported = (Value*) &unsupported;

void ConversionPass::performConversion(const std::vector<Value*>& queue) {
  Logger& logger = log();
  for (Value* value : queue) {
    ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(value);
    ConversionType* convType = taffoConvInfo.getNewType(value);

    auto indenter = logger.getIndenter();
    LLVM_DEBUG(
      logger << Logger::Blue << repeatString("▀▄▀▄", 10) << "[Perform conversion]" << repeatString("▄▀▄▀", 10)
             << Logger::Reset << "\n";
      logger.log("[Value] ", Logger::Bold).logValueln(value);
      if (valueConvInfo->isConversionDisabled())
        logger.logln("conversion disabled", Logger::Yellow);
      if (convType)
        logger << "requested conv type: " << *convType << "\n";);

    std::unique_ptr<ConversionType> resConvType = nullptr;
    Value* res = convert(value, &resConvType);
    convertedValues[value] = res;

    LLVM_DEBUG(
      if (resConvType)
        logger.log("result type: ", Logger::Green).logln(*resConvType);
      else if (ConversionType* newType = taffoConvInfo.getNewType(res))
        logger.log("result type: ", Logger::Green).logln(*newType);
      logger.log("result:      ", Logger::Green).logValueln(res) << "\n");

    if (res != value && isa<Instruction>(res) && isa<Instruction>(value)) {
      auto* newInst = dyn_cast<Instruction>(res);
      auto* oldInst = dyn_cast<Instruction>(value);
      newInst->setDebugLoc(oldInst->getDebugLoc());
    }
  }
}

Value* ConversionPass::createPlaceholder(Type* type, BasicBlock* where, StringRef name) {
  IRBuilder<NoFolder> builder(where, where->getFirstInsertionPt());
  AllocaInst* alloca = builder.CreateAlloca(type);
  return builder.CreateLoad(type, alloca, name);
}

Value* ConversionPass::convert(Value* value, std::unique_ptr<ConversionType>* resConvType) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(value);
  if (valueConvInfo->isArgumentPlaceholder)
    return convertedValues.at(value);

  if (auto* constant = dyn_cast<Constant>(value))
    return convertConstant(constant, *valueConvInfo->getNewType(), ConvTypePolicy::RangeOverHint, resConvType);

  if (auto* inst = dyn_cast<Instruction>(value))
    return convertInstruction(inst);

  if (auto* arg = dyn_cast<Argument>(value)) {
    if (getFullyUnwrappedType(arg)->isFloatingPointTy())
      return getConvertedOperand(value, *taffoConvInfo.getNewType(arg));
    return arg;
  }

  llvm_unreachable("Conversion failed");
}

Value* ConversionPass::getConvertedOperand(Value* value,
                                           const ConversionType& convType,
                                           Instruction* insertionPoint,
                                           const ConvTypePolicy& policy,
                                           std::unique_ptr<ConversionType>* resConvType) {
  ConversionType* currentConvType = taffoConvInfo.getOrCreateCurrentType(value);

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << "[" << __FUNCTION__ << "] ";
    logger.logValueln(value, false);
    indenter.increaseIndent();
    logger << "requested type: " << convType << "\n";);

  auto iter = convertedValues.find(value);
  if (iter != convertedValues.end()) {
    Value* convertedValue = iter->second;
    ConversionType* convertedValueConvType = taffoConvInfo.getCurrentType(convertedValue);
    LLVM_DEBUG(
      logger << "found converted value of type " << *convertedValueConvType << "\n";
      logger.log("converted operand: ", Logger::Green) << *convertedValue << "\n";);
    // We can return if we found a suitable converted value
    bool suitableValue = true;
    if (policy == ConvTypePolicy::ForceHint && *convertedValueConvType != convType)
      suitableValue = false;
    if (convType.isFixedPoint() != convertedValueConvType->isFixedPoint())
      suitableValue = false;
    if (convType.isFloatingPoint() != convertedValueConvType->isFloatingPoint())
      suitableValue = false;
    if (suitableValue) {
      if (resConvType)
        *resConvType = convertedValueConvType->clone();
      return convertedValue;
    }
    // Not suitable converted value: we need to convert further
    value = convertedValue;
    currentConvType = convertedValueConvType;
  }
  else if (*currentConvType == convType) {
    // We didn't find any converted value and value doesn't need conversion
    LLVM_DEBUG(logger.logln("operand did not need conversion", Logger::Green));
    if (resConvType)
      *resConvType = currentConvType->clone();
    return value;
  }

  LLVM_DEBUG(log() << "value must be converted: converting now\n");

  if (auto* constant = dyn_cast<Constant>(value))
    return convertConstant(constant, convType, policy, resConvType);

  auto* scalarCurrentConvType = cast<ConversionScalarType>(currentConvType);
  auto& scalarConvType = cast<ConversionScalarType>(convType);
  Value* res = genConvertConvToConv(value, *scalarCurrentConvType, scalarConvType, policy, insertionPoint);
  const auto* newConvType = taffoConvInfo.getNewType<ConversionScalarType>(res);
  if (*newConvType != scalarConvType && policy == ConvTypePolicy::ForceHint) {
    LLVM_DEBUG(log() << "forcing hint type\n");
    res = genConvertConvToConv(res, *newConvType, scalarConvType, ConvTypePolicy::ForceHint, insertionPoint);
  }
  LLVM_DEBUG(logger.log("converted operand: ", Logger::Green) << *res << "\n";);
  if (resConvType)
    *resConvType = newConvType->clone();
  return res;
}

Value* ConversionPass::genConvertConvToConv(Value* src,
                                            const ConversionScalarType& srcConvType,
                                            const ConversionScalarType& dstConvType,
                                            const ConvTypePolicy& policy,
                                            Instruction* insertionPoint) {
  if (srcConvType == dstConvType)
    return src;

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << "[" << __FUNCTION__ << "] (";
    logger.log(policy, Logger::Cyan) << ") ";
    logger.log(srcConvType, Logger::Cyan);
    logger << " -> ";
    logger.logln(dstConvType, Logger::Cyan);
    indenter.increaseIndent(););

  auto* inst = dyn_cast<Instruction>(src);
  if (!insertionPoint && inst)
    insertionPoint = getFirstInsertionPointAfter(inst);
  assert(insertionPoint && "insertionPoint required");

  TransparentType* srcType = srcConvType.toTransparentType();
  TransparentType* dstType = dstConvType.toTransparentType();
  Type* srcLLVMType = srcType->toLLVMType();
  Type* dstLLVMType = dstType->toLLVMType();
  IRBuilder<NoFolder> builder(insertionPoint);

  if (srcConvType.isFixedPoint() && dstConvType.isFloatingPoint())
    return genConvertConvToFloat(src, srcConvType, dstConvType);

  // Source and destination are both float
  if (srcType->isFloatingPointTyOrPtrTo() && dstType->isFloatingPointTyOrPtrTo()) {
    LLVM_DEBUG(logger << "converting float to float\n");

    unsigned srcBits = srcLLVMType->getPrimitiveSizeInBits();
    unsigned dstBits = dstLLVMType->getPrimitiveSizeInBits();

    if (srcBits == dstBits) {
      assert(*srcType == *dstType && "src and dst have same bits but different types");
      LLVM_DEBUG(logger << "no casting needed.\n");
      return src;
    }

    Value* res;
    if (srcBits < dstBits) // Extension needed
      res = builder.CreateFPExt(src, dstLLVMType);
    else                   // Truncation needed
      res = builder.CreateFPTrunc(src, dstLLVMType);

    LLVM_DEBUG(logger << res << "\n");
    setConversionResultInfo(res, src, &dstConvType);
    return res;
  }

  ConversionScalarType resConvType = dstConvType;
  if (policy == ConvTypePolicy::RangeOverHint)
    if (taffoInfo.hasValueInfo(*src))
      if (auto scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*src)))
        if (std::shared_ptr<Range> range = scalarInfo->range) {
          FixedPointTypeGenError err;
          FixedPointInfo fixedPointInfo = fixedPointInfoFromRange(*range, &err);
          if (err == FixedPointTypeGenError::NoError)
            resConvType = ConversionScalarType(*dstConvType.toTransparentType(), &fixedPointInfo);
        }

  if (srcConvType.isFloatingPoint() && resConvType.isFixedPoint())
    return genConvertFloatToConv(src, resConvType, insertionPoint);

  unsigned srcBits = srcConvType.getBits();
  unsigned dstBits = resConvType.getBits();

  auto genSizeChange = [&](Value* value) -> Value* {
    Value* res;
    if (srcConvType.isSigned())
      res = builder.CreateSExtOrTrunc(value, dstLLVMType);
    else
      res = builder.CreateZExtOrTrunc(value, dstLLVMType);
    LLVM_DEBUG(
      if (res != value)
        logger << "changed size from " << srcBits << " to " << dstBits << " bits\n";);
    return res;
  };

  auto genPointMovement = [&](Value* value) -> Value* {
    int deltaBits = resConvType.getFractionalBits() - srcConvType.getFractionalBits();
    Value* res;
    if (deltaBits > 0) {
      res = builder.CreateShl(value, deltaBits);
      LLVM_DEBUG(logger << "shifted left of " << deltaBits << " bits\n");
      return res;
    }
    if (deltaBits < 0) {
      if (srcConvType.isSigned())
        res = builder.CreateAShr(value, -deltaBits);
      else
        res = builder.CreateLShr(value, -deltaBits);
      LLVM_DEBUG(logger << "shifted right of " << -deltaBits << " bits\n");
      return res;
    }
    return value;
  };

  Value* res;
  if (dstBits > srcBits)
    res = genPointMovement(genSizeChange(src));
  else
    res = genSizeChange(genPointMovement(src));

  setConversionResultInfo(res, src, &resConvType);
  return res;
}

Value* ConversionPass::genConvertFloatToConv(Value* src,
                                             const ConversionScalarType& dstConvType,
                                             Instruction* insertionPoint) {
  assert(src->getType()->isFloatingPointTy() && "src must be a float scalar");
  ConversionType* srcConvType = taffoConvInfo.getOrCreateCurrentType(src);
  TransparentType* srcType = srcConvType->toTransparentType();

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << "[" << __FUNCTION__ << "] ";
    logger.log(*srcType, Logger::Cyan);
    logger << " -> ";
    logger.logln(dstConvType, Logger::Cyan);
    indenter.increaseIndent(););

  if (auto* constant = dyn_cast<Constant>(src))
    return convertConstant(constant, dstConvType, ConvTypePolicy::ForceHint);

  if (!insertionPoint)
    insertionPoint = getFirstInsertionPointAfter(src);
  assert(insertionPoint && "Missing insertion point");
  floatToFixCount++;

  IRBuilder<NoFolder> builder(insertionPoint);
  Type* srcLLVMType = src->getType();
  TransparentType* dstType = dstConvType.toTransparentType();
  Type* destLLVMType = dstType->toLLVMType();

  if (dstConvType.isFixedPoint()) {
    Value* res;
    if (auto* siToFpInst = dyn_cast<SIToFPInst>(src)) {
      Value* intOperand = siToFpInst->getOperand(0);
      res = builder.CreateShl(builder.CreateIntCast(intOperand, destLLVMType, true), dstConvType.getFractionalBits());
    }
    else if (auto* uiToFpInst = dyn_cast<UIToFPInst>(src)) {
      Value* intOperand = uiToFpInst->getOperand(0);
      res = builder.CreateShl(builder.CreateIntCast(intOperand, destLLVMType, false), dstConvType.getFractionalBits());
    }
    else {
      double exp = pow(2.0, dstConvType.getFractionalBits());
      const fltSemantics& fltSema = srcLLVMType->getFltSemantics();
      double maxSrc = APFloat::getLargest(fltSema).convertToDouble();
      double maxDest = pow(2.0, dstConvType.getBits());
      Value* sanitizedFloat = src;
      if (maxSrc < maxDest || maxSrc < exp) {
        LLVM_DEBUG(log() << "extending " << *srcLLVMType << " to float because dest integer is too large\n");
        sanitizedFloat = builder.CreateFPCast(src, Type::getFloatTy(src->getContext()));
      }
      Value* intermediateValue = builder.CreateFMul(ConstantFP::get(sanitizedFloat->getType(), exp), sanitizedFloat);
      if (dstConvType.isSigned())
        res = builder.CreateFPToSI(intermediateValue, destLLVMType);
      else
        res = builder.CreateFPToUI(intermediateValue, destLLVMType);
    }
    setConversionResultInfo(res, src, &dstConvType);
    return res;
  }

  if (dstConvType.isFloatingPoint()) {
    unsigned srcBits = src->getType()->getPrimitiveSizeInBits();
    unsigned dstBits = destLLVMType->getPrimitiveSizeInBits();
    if (srcBits == dstBits) {
      assert(*srcType == *dstType && "src and dst have same bits but different types");
      LLVM_DEBUG(log() << "no casting needed\n");
      return src;
    }

    Value* res;
    if (srcBits < dstBits) // Extension needed
      res = builder.CreateFPExt(src, destLLVMType);
    else                   // Truncation needed
      res = builder.CreateFPTrunc(src, destLLVMType);
    setConversionResultInfo(res, src, &dstConvType);
    return res;
  }

  llvm_unreachable("Unrecognized value type");
}

Value* ConversionPass::genConvertConvToFloat(Value* src,
                                             const ConversionScalarType& srcConvType,
                                             const ConversionScalarType& dstConvType) {
  auto* srcType = srcConvType.toTransparentType();
  auto* dstType = dstConvType.toTransparentType();
  Type* dstLLVMType = dstType->toLLVMType();
  assert(!srcType->isPointerTy() && !dstType->isPointerTy() && "src and dst cannot be pointers");

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << "[" << __FUNCTION__ << "] ";
    logger.log(srcConvType, Logger::Cyan);
    logger << " -> ";
    logger.logln(*dstType, Logger::Cyan);
    indenter.increaseIndent(););

  if (srcType->isFloatingPointTyOrPtrTo()) {

    unsigned srcBits = srcType->toLLVMType()->getPrimitiveSizeInBits();
    unsigned dstBits = dstLLVMType->getPrimitiveSizeInBits();

    if (isa<Instruction>(src) || isa<Argument>(src)) {
      if (srcBits == dstBits) {
        assert(*srcType == *dstType && "src and dst have same bits but different types");
        LLVM_DEBUG(logger << "no casting needed\n");
        return src;
      }

      IRBuilder<NoFolder> builder(getFirstInsertionPointAfter(src));
      Value* res;
      if (srcBits < dstBits) // Extension needed
        res = builder.CreateFPExt(src, dstLLVMType);
      else                   // Truncation needed
        res = builder.CreateFPTrunc(src, dstLLVMType);
      setConversionResultInfo(res, src, &dstConvType);
      return res;
    }
    if (auto* constant = dyn_cast<Constant>(src)) {
      if (srcBits == dstBits) {
        // No casting is actually needed
        assert(*srcType == *dstType && "src and dst have same bits but different types");
        LLVM_DEBUG(logger << "no casting needed.\n");
        return src;
      }

      Value* res;
      if (srcBits < dstBits) // Extension needed
        res = ConstantExpr::getCast(Instruction::FPExt, constant, dstLLVMType);
      else                   // Truncation needed
        res = ConstantExpr::getCast(Instruction::FPTrunc, constant, dstLLVMType);
      return res;

      /* TODO check this code and use constant folding also here? see constants below
      // Always convert to double then to the destination type
      // No need to worry about efficiency, as everything will be constant folded
      Type* TmpTy = Type::getDoubleTy(constant->getContext());
      Constant* floattmp = srcConvType.isSigned() ? ConstantExpr::getCast(Instruction::SIToFP, constant, TmpTy)
                                                  : ConstantExpr::getCast(Instruction::UIToFP, constant, TmpTy);
      double twoebits = pow(2.0, srcConvType.getFractionalBits());
      Constant* DblRes =
        ConstantFoldBinaryOpOperands(Instruction::FDiv, floattmp, ConstantFP::get(TmpTy, twoebits), *dataLayout);
      assert(DblRes && "Constant folding failed...");
      LLVM_DEBUG(log() << "ConstantFoldBinaryOpOperands returned " << *DblRes << "\n");
      Constant* Res = ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, dstLLVMType, *dataLayout);
      assert(Res && "Constant folding failed...");
      LLVM_DEBUG(log() << "ConstantFoldCastInstruction returned " << *Res << "\n");
      return Res;
      */
    }
  }

  if (!srcType->isIntegerTyOrPtrTo()) {
    LLVM_DEBUG(log() << "cannot wrap-convert to float non integer value\n");
    return nullptr;
  }
  fixToFloatCount++;

  if (isa<Instruction>(src) || isa<Argument>(src)) {
    IRBuilder<NoFolder> builder(getFirstInsertionPointAfter(src));

    double exp = pow(2.0, srcConvType.getFractionalBits());
    if (exp == 1.0) {
      LLVM_DEBUG(log() << "optimizing conversion removing division by one\n");
      Value* res =
        srcConvType.isSigned() ? builder.CreateSIToFP(src, dstLLVMType) : builder.CreateUIToFP(src, dstLLVMType);
      setConversionResultInfo(res, src, &dstConvType);
      return res;
    }

    const fltSemantics& fltSema = dstLLVMType->getFltSemantics();
    double maxDst = APFloat::getLargest(fltSema).convertToDouble();
    double maxSrc = pow(2.0, srcConvType.getBits());
    Value* res;
    if (maxDst < maxSrc || maxDst < exp) {
      LLVM_DEBUG(log() << "extending " << *dstType << " to float because source integer is too small\n");
      Type* tmpLLVMType = Type::getFloatTy(src->getContext());
      Value* floatTmp =
        srcConvType.isSigned() ? builder.CreateSIToFP(src, tmpLLVMType) : builder.CreateUIToFP(src, tmpLLVMType);
      res = builder.CreateFPTrunc(builder.CreateFDiv(floatTmp, ConstantFP::get(tmpLLVMType, exp)), dstLLVMType);
    }
    else {
      Value* floatTmp =
        srcConvType.isSigned() ? builder.CreateSIToFP(src, dstLLVMType) : builder.CreateUIToFP(src, dstLLVMType);
      res = builder.CreateFDiv(floatTmp, ConstantFP::get(dstLLVMType, exp));
    }
    setConversionResultInfo(res, src, &dstConvType);
    return res;
  }
  if (auto* constant = dyn_cast<Constant>(src)) {
    // Always convert to double then to the destination type
    // No need to worry about efficiency, as everything will be constant folded
    Type* tmpType = Type::getDoubleTy(constant->getContext());
    Constant* floatTmp = srcConvType.isSigned() ? ConstantExpr::getCast(Instruction::SIToFP, constant, tmpType)
                                                : ConstantExpr::getCast(Instruction::UIToFP, constant, tmpType);
    double exp = pow(2.0, srcConvType.getFractionalBits());
    Constant* doubleRes =
      ConstantFoldBinaryOpOperands(Instruction::FDiv, floatTmp, ConstantFP::get(tmpType, exp), *dataLayout);
    assert(doubleRes && "ConstantFoldBinaryOpOperands failed");
    LLVM_DEBUG(log() << "ConstantFoldBinaryOpOperands returned " << *doubleRes << "\n");
    Constant* res = ConstantFoldCastOperand(Instruction::FPTrunc, doubleRes, dstLLVMType, *dataLayout);
    assert(res && "Constant folding failed");
    LLVM_DEBUG(log() << "ConstantFoldCastInstruction returned " << *res << "\n");
    setConversionResultInfo(res, src, &dstConvType);
    return res;
  }
  llvm_unreachable("Unrecognized value type");
}
