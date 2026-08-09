#include "../ConversionPass.hpp"
#include "TaffoCommon/AliasUtils.hpp"
#include "TransparentType.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/NoFolder.h>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conv"

enum MathIntrinsicFamily : unsigned {
  Unrecognized = 0,
  FMA,
  FMulAdd,
  TANH,
  RELU,
  SIGMOID
};

static bool isExactFunctionOrClone(StringRef name, StringRef baseName) {
  if (name == baseName)
    return true;

  if (!name.consume_front(baseName))
    return false;

  if (!name.consume_front("_clone"))
    return false;

  if (name.empty())
    return false;

  for (char c : name)
    if (c < '0' || c > '9')
      return false;

  return true;
}

static MathIntrinsicFamily getMathIntrinsicFamily(Function* F) {
  if (!F)
    return Unrecognized;

  StringRef name = F->getName();

  if (isExactFunctionOrClone(name, "tanh") || isExactFunctionOrClone(name, "tanhf")
      || isExactFunctionOrClone(name, "tanhl") || name == "llvm.tanh.f32" || name == "llvm.tanh.f64")
    return TANH;

  if (isExactFunctionOrClone(name, "relu") || isExactFunctionOrClone(name, "reluf")
      || isExactFunctionOrClone(name, "relul"))
    return RELU;

  if (isExactFunctionOrClone(name, "sigmoid") || isExactFunctionOrClone(name, "sigmoidf")
      || isExactFunctionOrClone(name, "sigmoidl"))
    return SIGMOID;

  if (F->getIntrinsicID() == Intrinsic::fma)
    return FMA;

  if (name == "fma" || name == "fmal" || name == "fmaf")
    return FMA;

  if (F->getIntrinsicID() == Intrinsic::fmuladd)
    return FMulAdd;

  return Unrecognized;
}

bool ConversionPass::isSupportedMathIntrinsicFunction(Function* F) { return getMathIntrinsicFamily(F) != Unrecognized; }

Value* ConversionPass::convertMathIntrinsicFunction(CallBase* call) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(call);
  if (valueConvInfo->isConversionDisabled())
    return unsupported;

  TransparentType* type = taffoInfo.getOrCreateTransparentType(*call);
  auto* newConvType = valueConvInfo->getNewOrOldType<ConversionScalarType>();

  Function* fun = call->getCalledFunction();
  MathIntrinsicFamily family = getMathIntrinsicFamily(fun);

  if (family == FMA || family == FMulAdd) {
    Value* operand1 = call->getArgOperand(0);
    Value* operand2 = call->getArgOperand(1);
    Value* operand3 = call->getArgOperand(2);
    IRBuilder<NoFolder> builder(call);

    if (newConvType->isFixedPoint()) {
      auto* tmpFMul = cast<Instruction>(builder.CreateFMul(operand1, operand2));
      taffoInfo.setTransparentType(*tmpFMul, type->clone());
      auto* tmpFAdd = cast<Instruction>(builder.CreateFAdd(tmpFMul, operand3));
      taffoInfo.setTransparentType(*tmpFAdd, type->clone());
      convertedValues[tmpFMul] = convertFMul(tmpFMul, *newConvType);
      return convertFAdd(tmpFAdd, *newConvType);
    }
    if (newConvType->isFloatingPoint()) {
      Value* newOperand1 = getConvertedOperand(operand1, *newConvType, call, ConvTypePolicy::ForceHint);
      Value* newOperand2 = getConvertedOperand(operand2, *newConvType, call, ConvTypePolicy::ForceHint);
      Value* newOperand3 = getConvertedOperand(operand3, *newConvType, call, ConvTypePolicy::ForceHint);
      if (!newOperand1 || !newOperand2 || !newOperand3)
        return nullptr;

      Type* newLLVMType = newConvType->toScalarLLVMType(call->getContext());
      Function* newIntrinsic;
      if (family == FMA)
        newIntrinsic = aliasUtils::Intrinsic::getOrInsertDeclaration(call->getModule(), Intrinsic::fma, newLLVMType);
      else
        newIntrinsic =
          aliasUtils::Intrinsic::getOrInsertDeclaration(call->getModule(), Intrinsic::fmuladd, newLLVMType);
      Value* res =
        builder.CreateCall(newIntrinsic->getFunctionType(), newIntrinsic, {newOperand1, newOperand2, newOperand3});
      setConversionResultInfo(res, call, newConvType);
      return res;
    }
    llvm_unreachable("Unknown convType");
  }
  else if (family == TANH) {
    return createTanh(call);
  }
  else if (family == RELU) {
    return createReLU(call);
  }
  else if (family == SIGMOID) {
    return createSigmoid(call);
  }
  llvm_unreachable("Math intrinsic recognized but not handled");
}
