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

Constant*
ConversionPass::convertConstant(Constant* constant, std::shared_ptr<FixedPointType>& fixpt, TypeMatchPolicy policy) {
  if (dyn_cast<UndefValue>(constant))
    return UndefValue::get(getLLVMFixedPointTypeForFloatType(taffoInfo.getOrCreateTransparentType(*constant), fixpt));
  if (auto* globalVariable = dyn_cast<GlobalVariable>(constant))
    return convertGlobalVariable(globalVariable, fixpt);
  if (auto* constantFloat = dyn_cast<ConstantFP>(constant))
    return convertLiteral(constantFloat, nullptr, fixpt, policy);
  if (auto* constantAggregate = dyn_cast<ConstantAggregate>(constant))
    return convertConstantAggregate(constantAggregate, fixpt, policy);
  if (auto* constantDataSequential = dyn_cast<ConstantDataSequential>(constant))
    return convertConstantDataSequential(constantDataSequential, std::static_ptr_cast<FixedPointScalarType>(fixpt));
  if (dyn_cast<ConstantAggregateZero>(constant)) {
    Type* newt = getLLVMFixedPointTypeForFloatType(taffoInfo.getOrCreateTransparentType(*constant), fixpt);
    return ConstantAggregateZero::get(newt);
  }
  if (auto* constantExpr = dyn_cast<ConstantExpr>(constant))
    return convertConstantExpr(constantExpr, fixpt, policy);
  return nullptr;
}

Constant* ConversionPass::convertConstantExpr(ConstantExpr* constantExpr,
                                              std::shared_ptr<FixedPointType>& fixpt,
                                              TypeMatchPolicy policy) {
  if (isa<GEPOperator>(constantExpr)) {
    Value* newValue = convertedValues.at(constantExpr->getOperand(0));
    if (!newValue) {
      LLVM_DEBUG(log() << "[Warning] Operand of constant GEP not found in operandPool!\n");
      return nullptr;
    }
    auto* newConstant = dyn_cast<Constant>(newValue);
    if (!newConstant)
      return nullptr;

    if (policy == TypeMatchPolicy::ForceHint)
      assert(fixpt == getFixpType(newValue) && "type adjustment forbidden...");
    else
      fixpt = getFixpType(newValue);

    std::vector<Constant*> values;
    for (unsigned i = 1; i < constantExpr->getNumOperands(); i++)
      values.push_back(constantExpr->getOperand(i));

    ArrayRef idxlist(values);
    return ConstantExpr::getInBoundsGetElementPtr(nullptr, newConstant, idxlist);
  }
  LLVM_DEBUG(log() << "constant expression " << *constantExpr << " is not handled explicitly yet\n");
  return nullptr;
}

Constant* ConversionPass::convertGlobalVariable(GlobalVariable* globalVariable,
                                                std::shared_ptr<FixedPointType>& fixpt) {
  bool hasFloats = false;
  std::shared_ptr<TransparentType> type = taffoInfo.getOrCreateTransparentType(*globalVariable);
  Type* newLLVMType = getLLVMFixedPointTypeForFloatType(type, fixpt, &hasFloats);
  if (!newLLVMType)
    return nullptr;
  if (!hasFloats)
    return globalVariable;

  Constant* oldInit = globalVariable->getInitializer();
  Constant* newInit = nullptr;
  if (oldInit && !oldInit->isNullValue()) {
    /* global variables can be written to, so we always convert them to the type allocated by the DTA */
    newInit = convertConstant(oldInit, fixpt, TypeMatchPolicy::ForceHint);
  }
  else
    newInit = Constant::getNullValue(newLLVMType);

  auto* newGlobalVariable = new GlobalVariable(
    *(globalVariable->getParent()), newLLVMType, globalVariable->isConstant(), globalVariable->getLinkage(), newInit);
  newGlobalVariable->setAlignment(MaybeAlign(globalVariable->getAlignment()));
  newGlobalVariable->setName(globalVariable->getName() + ".fixp");
  return newGlobalVariable;
}

Constant* ConversionPass::convertConstantAggregate(ConstantAggregate* constantAggregate,
                                                   std::shared_ptr<FixedPointType>& fixpt,
                                                   TypeMatchPolicy policy) {
  std::vector<Constant*> constants;
  for (unsigned i = 0; i < constantAggregate->getNumOperands(); i++) {
    Constant* oldConst = constantAggregate->getOperand(i);
    Constant* newConst = nullptr;
    if (getFullyUnwrappedType(oldConst)->isFloatingPointTy()) {
      newConst = convertConstant(constantAggregate->getOperand(i), fixpt, TypeMatchPolicy::ForceHint);
      if (!newConst)
        return nullptr;
    }
    else
      newConst = oldConst;
    constants.push_back(newConst);
  }

  if (isa<ConstantArray>(constantAggregate)) {
    ArrayType* aty = ArrayType::get(constants[0]->getType(), constants.size());
    return ConstantArray::get(aty, constants);
  }
  if (isa<ConstantVector>(constantAggregate))
    return ConstantVector::get(constants);
  if (isa<ConstantStruct>(constantAggregate)) {
    std::vector<Type*> types;
    types.reserve(constants.size());
    for (Constant* constant : constants)
      types.push_back(constant->getType());
    StructType* structType = StructType::get(constantAggregate->getContext(), types);
    return ConstantStruct::get(structType, constants);
  }
  llvm_unreachable("a ConstantAggregate is not an array, vector or struct...");
}

template <class T>
Constant* ConversionPass::createConstantDataSequential(ConstantDataSequential* cds,
                                                       const std::shared_ptr<FixedPointType>& fixpt) {
  std::vector<T> newConsts;

  for (unsigned i = 0; i < cds->getNumElements(); i++) {
    APFloat thisElem = cds->getElementAsAPFloat(i);
    APSInt fixVal;
    if (!convertAPFloat(thisElem, fixVal, nullptr, std::static_ptr_cast<FixedPointScalarType>(fixpt))) {
      LLVM_DEBUG(log() << *cds << " conv failed because an apfloat cannot be converted to " << *fixpt << "\n");
      return nullptr;
    }
    newConsts.push_back(fixVal.getExtValue());
  }

  if (isa<ConstantDataArray>(cds))
    return ConstantDataArray::get(cds->getContext(), newConsts);
  return ConstantDataVector::get(cds->getContext(), newConsts);
}

template <class T>
Constant* ConversionPass::createConstantDataSequentialFP(ConstantDataSequential* cds,
                                                         const std::shared_ptr<FixedPointType>& fixpt) {
  std::vector<T> newConsts;

  for (unsigned i = 0; i < cds->getNumElements(); i++) {
    bool dontCare;

    APFloat thisElem = cds->getElementAsAPFloat(i);
    thisElem.convert(APFloat::IEEEdouble(), APFloatBase::rmTowardZero, &dontCare);
    newConsts.push_back(thisElem.convertToDouble());
  }

  if (isa<ConstantDataArray>(cds))
    return ConstantDataArray::get(cds->getContext(), newConsts);
  return ConstantDataVector::get(cds->getContext(), newConsts);
}

Constant* ConversionPass::convertConstantDataSequential(ConstantDataSequential* cds,
                                                        const std::shared_ptr<FixedPointScalarType>& fixpt) {
  if (!getFullyUnwrappedType(cds)->isFloatingPointTy())
    return cds;

  if (fixpt->isFixedPoint()) {
    if (fixpt->getBits() <= 8)
      return createConstantDataSequential<uint8_t>(cds, fixpt);
    if (fixpt->getBits() <= 16)
      return createConstantDataSequential<uint16_t>(cds, fixpt);
    if (fixpt->getBits() <= 32)
      return createConstantDataSequential<uint32_t>(cds, fixpt);
    if (fixpt->getBits() <= 64)
      return createConstantDataSequential<uint64_t>(cds, fixpt);
  }

  if (fixpt->isFloatingPoint()) {
    if (fixpt->getFloatStandard() == FixedPointScalarType::Float_float)
      return createConstantDataSequentialFP<float>(cds, fixpt);

    if (fixpt->getFloatStandard() == FixedPointScalarType::Float_double)
      return createConstantDataSequentialFP<double>(cds, fixpt);
    // As the sequential data does not accept anything different from float or double, we are doomed.
    // It's better to crash, so we see this kind of error. Maybe we can modify something at program source code level?
    llvm_unreachable("You cannot have anything different from float or double here, my friend!");
  }

  LLVM_DEBUG(log() << *fixpt << " too big for ConstantDataArray/Vector; 64 bit max\n");
  return nullptr;
}

Constant* ConversionPass::convertLiteral(ConstantFP* constantFloat,
                                         Instruction* context,
                                         std::shared_ptr<FixedPointType>& fixpt,
                                         TypeMatchPolicy policy) {
  APFloat val = constantFloat->getValueAPF();
  APSInt fixVal;

  // Old workflow, convert the value to a fixed point value
  if (fixpt->isFixedPoint()) {
    if (!policy.isHintPreferredPolicy()) {
      APFloat tmp(val);
      bool precise = false;
      tmp.convert(APFloatBase::IEEEdouble(), APFloat::rmTowardNegative, &precise);
      double dblval = tmp.convertToDouble();
      int nbits = std::static_ptr_cast<FixedPointScalarType>(fixpt)->getBits();
      Range range(dblval, dblval);
      int minflt = policy.isMaxIntPolicy() ? -1 : 0;
      FixedPointInfo t = fixedPointTypeFromRange(range, nullptr, nbits, minflt);
      fixpt = std::make_shared<FixedPointScalarType>(&t);
    }

    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);
    if (convertAPFloat(val, fixVal, context, scalarFixpt)) {
      Type* intty = scalarFixpt->scalarToLLVMType(constantFloat->getContext());
      return ConstantInt::get(intty, fixVal);
    }
    else {
      return nullptr;
    }
  }

  // Just "convert", actually recast, the value to the correct data type if using floating point data
  if (fixpt->isFloatingPoint()) {
    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_pointer_cast<FixedPointScalarType>(fixpt);
    Type* intty = scalarFixpt->scalarToLLVMType(constantFloat->getContext());
    bool loosesInfo;

    val.convert(intty->getFltSemantics(), APFloatBase::rmTowardPositive, &loosesInfo);

    return ConstantFP::get(intty, val);
  }

  llvm_unreachable("We should have already covered all values, are you introducing a new data type?");
}

bool ConversionPass::convertAPFloat(APFloat val,
                                    APSInt& fixval,
                                    Instruction* context,
                                    const std::shared_ptr<FixedPointScalarType>& fixpt) {
  bool precise = false;

  APFloat exp(pow(2.0, fixpt->getFractionalBits()));
  exp.convert(val.getSemantics(), APFloat::rmTowardNegative, &precise);
  val.multiply(exp, APFloat::rmTowardNegative);

  fixval = APSInt(fixpt->getBits(), !fixpt->isSigned());
  APFloat::opStatus res = val.convertToInteger(fixval, APFloat::rmTowardNegative, &precise);

  if (res != APFloat::opStatus::opOK && context) {
    OptimizationRemarkEmitter ORE(context->getFunction());
    if (res == APFloat::opStatus::opInexact) {
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "ImpreciseConstConversion", context)
               << "fixed point conversion of constant " << toString(val) << " is not precise\n");
    }
    else {
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "ConstConversionFailed", context)
               << "impossible to convert constant " << toString(val) << " to fixed point\n");
      return false;
    }
  }

  return true;
}
