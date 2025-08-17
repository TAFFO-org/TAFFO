#include "ConversionPass.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <cassert>
#include <cmath>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

Constant* ConversionPass::convertConstant(Constant* constant, const ConversionType& convType, ConvTypePolicy policy) {
  if (isa<UndefValue>(constant)) {
    TransparentType* newType = convType.toTransparentType();
    auto* res = UndefValue::get(newType->toLLVMType());
    setConversionResultInfo(res, constant, &convType);
    return res;
  }
  if (isa<ConstantAggregateZero>(constant)) {
    TransparentType* newType = convType.toTransparentType();
    auto* res = ConstantAggregateZero::get(newType->toLLVMType());
    setConversionResultInfo(res, constant, &convType);
    return res;
  }

  if (auto* globalVariable = dyn_cast<GlobalVariable>(constant))
    return convertGlobalVariable(globalVariable, convType);
  if (auto* constantFloat = dyn_cast<ConstantFP>(constant))
    return convertConstantFloat(constantFloat, nullptr, cast<ConversionScalarType>(convType), policy);
  if (auto* constantAggregate = dyn_cast<ConstantAggregate>(constant))
    return convertConstantAggregate(constantAggregate, convType);
  if (auto* constantDataSequential = dyn_cast<ConstantDataSequential>(constant))
    return convertConstantDataSequential(constantDataSequential, cast<ConversionScalarType>(convType));
  if (auto* constantExpr = dyn_cast<ConstantExpr>(constant))
    return convertConstantExpr(constantExpr, convType, policy);

  return nullptr;
}

bool ConversionPass::convertAPFloat(APFloat floatValue,
                                    APSInt& fixedPointValue,
                                    const ConversionScalarType& convType,
                                    Instruction* inst) {
  bool precise = false;

  APFloat exp(pow(2.0, convType.getFractionalBits()));
  exp.convert(floatValue.getSemantics(), APFloat::rmTowardNegative, &precise);
  floatValue.multiply(exp, APFloat::rmTowardNegative);

  fixedPointValue = APSInt(convType.getBits(), !convType.isSigned());
  APFloat::opStatus res = floatValue.convertToInteger(fixedPointValue, APFloat::rmTowardNegative, &precise);

  if (res != APFloat::opStatus::opOK) {
    if (res != APFloat::opStatus::opInexact) {
      if (inst) {
        OptimizationRemarkEmitter emitter(inst->getFunction());
        emitter.emit(OptimizationRemark(DEBUG_TYPE, "ConstConversionFailed", inst)
                     << "impossible to convert constant " << toString(floatValue) << " to fixed point\n");
      }
      return false;
    }
    if (inst) {
      OptimizationRemarkEmitter emitter(inst->getFunction());
      emitter.emit(OptimizationRemark(DEBUG_TYPE, "ImpreciseConstConversion", inst)
                   << "fixed point conversion of constant " << toString(floatValue) << " is not precise\n");
    }
  }
  return true;
}

Constant* ConversionPass::convertGlobalVariable(GlobalVariable* globalVariable, const ConversionType& convType) {
  std::unique_ptr<TransparentType> newType = convType.toTransparentType()->getPointedType();
  Type* newLLVMType = newType->toLLVMType();

  Constant* initializer = globalVariable->getInitializer();
  Constant* newInitializer = nullptr;
  if (initializer && !initializer->isNullValue()) {
    std::unique_ptr<ConversionType> initializerConvType = convType.clone(*newType);
    newInitializer = convertConstant(initializer, *initializerConvType, ConvTypePolicy::ForceHint);
  }
  else
    newInitializer = Constant::getNullValue(newLLVMType);

  auto* res = new GlobalVariable(*(globalVariable->getParent()),
                                 newLLVMType,
                                 globalVariable->isConstant(),
                                 globalVariable->getLinkage(),
                                 newInitializer);
  res->setAlignment(MaybeAlign(globalVariable->getAlignment()));
  res->setName(globalVariable->getName() + ".fixp");
  setConversionResultInfo(res, globalVariable, &convType);
  return res;
}

Constant* ConversionPass::convertConstantFloat(ConstantFP* floatConst,
                                               Instruction* inst,
                                               const ConversionScalarType& convType,
                                               ConvTypePolicy policy) {
  APFloat floatConstantValue = floatConst->getValueAPF();

  if (convType.isFixedPoint()) {
    APSInt fixedPointConstantValue;
    ConversionScalarType resConvType = convType;

    if (policy != ConvTypePolicy::ForceHint) {
      bool precise = false;
      APFloat doubleConstantValue(floatConstantValue);
      doubleConstantValue.convert(APFloatBase::IEEEdouble(), APFloat::rmTowardNegative, &precise);
      double doubleValue = doubleConstantValue.convertToDouble();
      FixedPointInfo fixedPointInfo =
        fixedPointInfoFromRange(Range(doubleValue, doubleValue), nullptr, convType.getBits(), 0);
      resConvType = ConversionScalarType(*resConvType.toTransparentType(), &fixedPointInfo);
    }

    if (convertAPFloat(floatConstantValue, fixedPointConstantValue, resConvType, inst)) {
      TransparentType* newType = resConvType.toTransparentType();
      Type* newLLVMType = newType->toLLVMType();
      auto* res = ConstantInt::get(newLLVMType, fixedPointConstantValue);
      setConversionResultInfo(res, floatConst, &resConvType);
      return res;
    }
    return nullptr;
  }
  if (convType.isFloatingPoint()) {
    TransparentType* newType = convType.toTransparentType();
    Type* newLLVMType = newType->toLLVMType();
    bool loosesInfo;
    floatConstantValue.convert(newLLVMType->getFltSemantics(), APFloatBase::rmTowardPositive, &loosesInfo);
    auto* res = ConstantFP::get(newLLVMType, floatConstantValue);
    setConversionResultInfo(res, floatConst, &convType);
    return res;
  }
  llvm_unreachable("ConversionType not handled");
}

Constant* ConversionPass::convertConstantAggregate(ConstantAggregate* constantAggregate,
                                                   const ConversionType& convType) {
  unsigned numOperands = constantAggregate->getNumOperands();
  std::vector<Constant*> constants;
  constants.reserve(numOperands);
  for (unsigned i = 0; i < numOperands; i++) {
    Constant* oldConst = constantAggregate->getOperand(i);
    Constant* newConst = nullptr;
    if (getFullyUnwrappedType(oldConst)->isFloatingPointTy()) {
      TransparentType* oldType = taffoInfo.getOrCreateTransparentType(*oldConst);
      std::unique_ptr<ConversionType> constConvType = convType.clone(*oldType);
      newConst = convertConstant(constantAggregate->getOperand(i), *constConvType, ConvTypePolicy::ForceHint);
      if (!newConst)
        return nullptr;
    }
    else
      newConst = oldConst;
    constants.push_back(newConst);
  }

  Constant* res;
  if (isa<ConstantArray>(constantAggregate)) {
    auto* arrayType = ArrayType::get(constants[0]->getType(), constants.size());
    res = ConstantArray::get(arrayType, constants);
  }
  else if (isa<ConstantVector>(constantAggregate)) {
    res = ConstantVector::get(constants);
  }
  else if (isa<ConstantStruct>(constantAggregate)) {
    std::vector<Type*> types;
    types.reserve(constants.size());
    for (Constant* constant : constants)
      types.push_back(constant->getType());
    StructType* structType = StructType::get(constantAggregate->getContext(), types);
    res = ConstantStruct::get(structType, constants);
  }
  else
    llvm_unreachable("ConstantAggregate not handled");

  setConversionResultInfo(res, constantAggregate, &convType);
  return res;
}

Constant* ConversionPass::convertConstantDataSequential(ConstantDataSequential* cds,
                                                        const ConversionScalarType& convType) {
  if (!getFullyUnwrappedType(cds)->isFloatingPointTy())
    return cds;

  if (convType.isFixedPoint()) {
    if (convType.getBits() <= 8)
      return createConstantDataSequentialFixedPoint<uint8_t>(cds, convType);
    if (convType.getBits() <= 16)
      return createConstantDataSequentialFixedPoint<uint16_t>(cds, convType);
    if (convType.getBits() <= 32)
      return createConstantDataSequentialFixedPoint<uint32_t>(cds, convType);
    if (convType.getBits() <= 64)
      return createConstantDataSequentialFixedPoint<uint64_t>(cds, convType);
  }

  if (convType.isFloatingPoint()) {
    if (convType.getFloatStandard() == ConversionScalarType::Float_float)
      return createConstantDataSequentialFloat<float>(cds, convType);
    if (convType.getFloatStandard() == ConversionScalarType::Float_double)
      return createConstantDataSequentialFloat<double>(cds, convType);
    // As the sequential data does not accept anything different from float or double, we are doomed.
    // It's better to crash, so we see this kind of error. Maybe we can modify something at program source code level?
    llvm_unreachable("Only float or double supported in constantDataSequential");
  }

  LLVM_DEBUG(log() << convType << " too big for ConstantDataArray/Vector; 64 bit max\n");
  return nullptr;
}

template <class T>
Constant* ConversionPass::createConstantDataSequentialFixedPoint(ConstantDataSequential* cds,
                                                                 const ConversionScalarType& convType) {
  unsigned numElements = cds->getNumElements();
  std::vector<T> newElements;
  newElements.reserve(numElements);
  for (unsigned i = 0; i < numElements; i++) {
    APFloat oldElem = cds->getElementAsAPFloat(i);
    APSInt newElem;
    if (!convertAPFloat(oldElem, newElem, convType))
      return nullptr;
    newElements.push_back(newElem.getExtValue());
  }

  Constant* res = nullptr;
  if (isa<ConstantDataArray>(cds))
    res = ConstantDataArray::get(cds->getContext(), newElements);
  else if (isa<ConstantDataVector>(cds))
    res = ConstantDataVector::get(cds->getContext(), newElements);
  assert(res);
  setConversionResultInfo(res, cds, &convType);
  return res;
}

template <class T>
Constant* ConversionPass::createConstantDataSequentialFloat(ConstantDataSequential* cds,
                                                            const ConversionScalarType& convType) {
  unsigned numElements = cds->getNumElements();
  std::vector<T> newElements;
  newElements.reserve(numElements);
  for (unsigned i = 0; i < numElements; i++) {
    bool dontCare;
    APFloat elem = cds->getElementAsAPFloat(i);
    elem.convert(APFloat::IEEEdouble(), APFloatBase::rmTowardZero, &dontCare);
    newElements.push_back(elem.convertToDouble());
  }

  Constant* res = nullptr;
  if (isa<ConstantDataArray>(cds))
    res = ConstantDataArray::get(cds->getContext(), newElements);
  else if (isa<ConstantDataVector>(cds))
    res = ConstantDataVector::get(cds->getContext(), newElements);
  assert(res);
  setConversionResultInfo(res, cds, &convType);
  return res;
}

Constant*
ConversionPass::convertConstantExpr(ConstantExpr* constantExpr, const ConversionType& convType, ConvTypePolicy policy) {
  if (isa<GEPOperator>(constantExpr)) {
    Value* newValue = convertedValues.at(constantExpr->getOperand(0));

    auto* newConstant = dyn_cast<Constant>(newValue);
    if (!newConstant)
      return nullptr;

    std::unique_ptr<ConversionType> resConvType = convType.clone();
    if (policy == ConvTypePolicy::ForceHint)
      assert(*resConvType == *taffoConvInfo.getNewType(newValue) && "type adjustment forbidden");
    else
      *resConvType = *taffoConvInfo.getNewType(newValue);

    SmallVector<Constant*, 4> indices;
    for (unsigned i = 1; i < constantExpr->getNumOperands(); i++)
      indices.push_back(constantExpr->getOperand(i));

    TransparentType* newType = resConvType->toTransparentType();
    auto* res = ConstantExpr::getInBoundsGetElementPtr(newType->toLLVMType(), newConstant, indices);
    setConversionResultInfo(res, constantExpr, resConvType.get());
    return res;
  }

  LLVM_DEBUG(log() << Logger::Yellow << "ConstantExpr not handled: " << *constantExpr << "\n"
                   << Logger::Reset);
  return nullptr;
}
