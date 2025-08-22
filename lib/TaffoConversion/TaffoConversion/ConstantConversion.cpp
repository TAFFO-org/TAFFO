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

#define DEBUG_TYPE "taffo-conv"

Constant* ConversionPass::convertConstant(Constant* constant,
                                          const ConversionType& convType,
                                          const ConvTypePolicy& policy,
                                          std::unique_ptr<ConversionType>* resConvType) {
  if (isa<UndefValue>(constant)) {
    TransparentType* newType = convType.toTransparentType();
    auto* res = UndefValue::get(newType->toLLVMType());
    setConstantConversionResultInfo(res, constant, &convType, resConvType);
    return res;
  }
  if (isa<ConstantAggregateZero>(constant)) {
    TransparentType* newType = convType.toTransparentType();
    auto* res = ConstantAggregateZero::get(newType->toLLVMType());
    setConstantConversionResultInfo(res, constant, &convType, resConvType);
    return res;
  }

  if (auto* globalVariable = dyn_cast<GlobalVariable>(constant))
    return convertGlobalVariable(globalVariable, convType, resConvType);
  if (auto* constantFloat = dyn_cast<ConstantFP>(constant))
    return convertConstantFloat(constantFloat, cast<ConversionScalarType>(convType), policy, resConvType);
  if (auto* constantAggregate = dyn_cast<ConstantAggregate>(constant))
    return convertConstantAggregate(constantAggregate, convType, resConvType);
  if (auto* constantDataSequential = dyn_cast<ConstantDataSequential>(constant))
    return convertConstantDataSequential(constantDataSequential, cast<ConversionScalarType>(convType), resConvType);
  if (auto* constantExpr = dyn_cast<ConstantExpr>(constant))
    return convertConstantExpr(constantExpr, convType, policy, resConvType);
  llvm_unreachable("Constant conversion failed");
}

Constant* ConversionPass::convertGlobalVariable(GlobalVariable* globalVariable,
                                                const ConversionType& convType,
                                                std::unique_ptr<ConversionType>* resConvType) {
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
  setConstantConversionResultInfo(res, globalVariable, &convType, resConvType);
  return res;
}

Constant* ConversionPass::convertConstantFloat(ConstantFP* floatConst,
                                               const ConversionScalarType& convType,
                                               const ConvTypePolicy& policy,
                                               std::unique_ptr<ConversionType>* resConvType) {
  APFloat floatConstantValue = floatConst->getValueAPF();

  if (convType.isFixedPoint()) {
    APSInt fixedPointConstantValue;
    ConversionScalarType newConvType = convType;

    if (policy != ConvTypePolicy::ForceHint) {
      bool precise = false;
      APFloat doubleConstantValue(floatConstantValue);
      doubleConstantValue.convert(APFloatBase::IEEEdouble(), APFloat::rmTowardNegative, &precise);
      double doubleValue = doubleConstantValue.convertToDouble();
      FixedPointInfo fixedPointInfo =
        fixedPointInfoFromRange(Range(doubleValue, doubleValue), nullptr, convType.getBits(), 0);
      newConvType = ConversionScalarType(*newConvType.toTransparentType(), &fixedPointInfo);
    }

    convertAPFloat(floatConstantValue, fixedPointConstantValue, newConvType);
    TransparentType* newType = newConvType.toTransparentType();
    Type* newLLVMType = newType->toLLVMType();
    auto* res = ConstantInt::get(newLLVMType, fixedPointConstantValue);
    setConstantConversionResultInfo(res, floatConst, &newConvType, resConvType);
    return res;
  }
  if (convType.isFloatingPoint()) {
    TransparentType* newType = convType.toTransparentType();
    Type* newLLVMType = newType->toLLVMType();
    bool loosesInfo;
    floatConstantValue.convert(newLLVMType->getFltSemantics(), APFloatBase::rmTowardPositive, &loosesInfo);
    auto* res = ConstantFP::get(newLLVMType, floatConstantValue);
    setConstantConversionResultInfo(res, floatConst, &convType, resConvType);
    return res;
  }
  llvm_unreachable("ConversionType not handled");
}

Constant* ConversionPass::convertConstantAggregate(ConstantAggregate* constantAggregate,
                                                   const ConversionType& convType,
                                                   std::unique_ptr<ConversionType>* resConvType) {
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

  setConstantConversionResultInfo(res, constantAggregate, &convType, resConvType);
  return res;
}

Constant* ConversionPass::convertConstantDataSequential(ConstantDataSequential* cds,
                                                        const ConversionScalarType& convType,
                                                        std::unique_ptr<ConversionType>* resConvType) {
  if (!getFullyUnwrappedType(cds)->isFloatingPointTy())
    return cds;

  if (convType.isFixedPoint()) {
    if (convType.getBits() <= 8)
      return createConstantDataSequentialFixedPoint<uint8_t>(cds, convType, resConvType);
    if (convType.getBits() <= 16)
      return createConstantDataSequentialFixedPoint<uint16_t>(cds, convType, resConvType);
    if (convType.getBits() <= 32)
      return createConstantDataSequentialFixedPoint<uint32_t>(cds, convType, resConvType);
    if (convType.getBits() <= 64)
      return createConstantDataSequentialFixedPoint<uint64_t>(cds, convType, resConvType);
  }

  if (convType.isFloatingPoint()) {
    if (convType.getFloatStandard() == ConversionScalarType::Float_float)
      return createConstantDataSequentialFloat<float>(cds, convType, resConvType);
    if (convType.getFloatStandard() == ConversionScalarType::Float_double)
      return createConstantDataSequentialFloat<double>(cds, convType, resConvType);
    // As the sequential data does not accept anything different from float or double, we are doomed.
    // It's better to crash, so we see this kind of error. Maybe we can modify something at program source code level?
    llvm_unreachable("Only float or double supported in constantDataSequential");
  }

  LLVM_DEBUG(log() << Logger::Red << convType << " too big for ConstantDataArray/Vector; 64 bit max\n"
                   << Logger::Reset);
  llvm_unreachable("Constant conversion failed");
}

template <class T>
Constant* ConversionPass::createConstantDataSequentialFixedPoint(ConstantDataSequential* cds,
                                                                 const ConversionScalarType& convType,
                                                                 std::unique_ptr<ConversionType>* resConvType) {
  unsigned numElements = cds->getNumElements();
  std::vector<T> newElements;
  newElements.reserve(numElements);
  for (unsigned i = 0; i < numElements; i++) {
    APFloat oldElem = cds->getElementAsAPFloat(i);
    APSInt newElem;
    convertAPFloat(oldElem, newElem, convType);
    newElements.push_back(newElem.getExtValue());
  }

  Constant* res = nullptr;
  if (isa<ConstantDataArray>(cds))
    res = ConstantDataArray::get(cds->getContext(), newElements);
  else if (isa<ConstantDataVector>(cds))
    res = ConstantDataVector::get(cds->getContext(), newElements);
  assert(res);
  setConstantConversionResultInfo(res, cds, &convType, resConvType);
  return res;
}

template <class T>
Constant* ConversionPass::createConstantDataSequentialFloat(ConstantDataSequential* cds,
                                                            const ConversionScalarType& convType,
                                                            std::unique_ptr<ConversionType>* resConvType) {
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
  setConstantConversionResultInfo(res, cds, &convType, resConvType);
  return res;
}

Constant* ConversionPass::convertConstantExpr(ConstantExpr* constantExpr,
                                              const ConversionType& convType,
                                              const ConvTypePolicy& policy,
                                              std::unique_ptr<ConversionType>* resConvType) {
  if (isa<GEPOperator>(constantExpr)) {
    Value* newValue = convertedValues.at(constantExpr->getOperand(0));

    auto* newConstant = dyn_cast<Constant>(newValue);
    if (!newConstant)
      llvm_unreachable("ConstantExpr conversion failed");

    std::unique_ptr<ConversionType> newConvType = convType.clone();
    if (policy == ConvTypePolicy::ForceHint)
      assert(*newConvType == *taffoConvInfo.getNewType(newValue) && "Cannot force hint type");
    else
      *newConvType = *taffoConvInfo.getNewType(newValue);

    SmallVector<Constant*, 4> indices;
    for (unsigned i = 1; i < constantExpr->getNumOperands(); i++)
      indices.push_back(constantExpr->getOperand(i));

    TransparentType* newType = newConvType->toTransparentType();
    auto* res = ConstantExpr::getInBoundsGetElementPtr(newType->toLLVMType(), newConstant, indices);
    setConstantConversionResultInfo(res, constantExpr, newConvType.get(), resConvType);
    return res;
  }
  llvm_unreachable("ConstantExpr not handled");
}

void ConversionPass::convertAPFloat(APFloat floatValue, APSInt& fixedPointValue, const ConversionScalarType& convType) {
  bool precise = false;

  APFloat exp(pow(2.0, convType.getFractionalBits()));
  exp.convert(floatValue.getSemantics(), APFloat::rmTowardNegative, &precise);
  floatValue.multiply(exp, APFloat::rmTowardNegative);

  fixedPointValue = APSInt(convType.getBits(), !convType.isSigned());
  APFloat::opStatus res = floatValue.convertToInteger(fixedPointValue, APFloat::rmTowardNegative, &precise);

  if (res != APFloat::opStatus::opOK) {
    if (res != APFloat::opStatus::opInexact)
      llvm_unreachable("APFloat conversion failed");
    log() << Logger::Yellow << "imprecise conversion of APFloat\n" << Logger::Reset;
  }
}
