#include "ConversionPass.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <cmath>

using namespace llvm;
using namespace taffo;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

Constant*
FloatToFixed::convertConstant(Constant* flt, std::shared_ptr<FixedPointType>& fixpt, TypeMatchPolicy typepol) {
  if (dyn_cast<UndefValue>(flt)) {
    return UndefValue::get(
      getLLVMFixedPointTypeForFloatType(TaffoInfo::getInstance().getOrCreateTransparentType(*flt), fixpt));
  }
  if (GlobalVariable* gvar = dyn_cast<GlobalVariable>(flt)) {
    return convertGlobalVariable(gvar, fixpt);
  }
  else if (ConstantFP* fpc = dyn_cast<ConstantFP>(flt)) {
    return convertLiteral(fpc, nullptr, fixpt, typepol);
  }
  else if (ConstantAggregate* cag = dyn_cast<ConstantAggregate>(flt)) {
    return convertConstantAggregate(cag, fixpt, typepol);
  }
  else if (ConstantDataSequential* cds = dyn_cast<ConstantDataSequential>(flt)) {
    return convertConstantDataSequential(cds, std::static_ptr_cast<FixedPointScalarType>(fixpt));
  }
  else if (dyn_cast<ConstantAggregateZero>(flt)) {
    Type* newt = getLLVMFixedPointTypeForFloatType(TaffoInfo::getInstance().getOrCreateTransparentType(*flt), fixpt);
    return ConstantAggregateZero::get(newt);
  }
  else if (ConstantExpr* cexp = dyn_cast<ConstantExpr>(flt)) {
    return convertConstantExpr(cexp, fixpt, typepol);
  }
  return nullptr;
}

Constant*
FloatToFixed::convertConstantExpr(ConstantExpr* cexp, std::shared_ptr<FixedPointType>& fixpt, TypeMatchPolicy typepol) {
  if (isa<GEPOperator>(cexp)) {
    Value* newval = convertedValues[cexp->getOperand(0)];
    if (!newval) {
      LLVM_DEBUG(dbgs() << "[Warning] Operand of constant GEP not found in operandPool!\n");
      return nullptr;
    }
    Constant* newconst = dyn_cast<Constant>(newval);
    if (!newconst)
      return nullptr;

    if (typepol == TypeMatchPolicy::ForceHint)
      assert(fixpt == getFixpType(newval) && "type adjustment forbidden...");
    else
      fixpt = getFixpType(newval);

    std::vector<Constant*> vals;
    for (unsigned int i = 1; i < cexp->getNumOperands(); i++)
      vals.push_back(cexp->getOperand(i));

    ArrayRef<Constant*> idxlist(vals);
    return ConstantExpr::getInBoundsGetElementPtr(nullptr, newconst, idxlist);
  }
  else {
    LLVM_DEBUG(dbgs() << "constant expression " << *cexp << " is not handled explicitly yet\n");
  }
  return nullptr;
}

Constant* FloatToFixed::convertGlobalVariable(GlobalVariable* glob, std::shared_ptr<FixedPointType>& fixpt) {
  bool hasfloats = false;
  Type* prevt = getUnwrappedType(glob);
  Type* newt =
    getLLVMFixedPointTypeForFloatType(TaffoInfo::getInstance().getOrCreateTransparentType(*glob), fixpt, &hasfloats);
  if (!newt)
    return nullptr;
  if (!hasfloats)
    return glob;

  Constant* oldinit = glob->getInitializer();
  Constant* newinit = nullptr;
  if (oldinit && !oldinit->isNullValue()) {
    /* global variables can be written to, so we always convert them to the type allocated by the DTA */
    newinit = convertConstant(oldinit, fixpt, TypeMatchPolicy::ForceHint);
  }
  else
    newinit = Constant::getNullValue(newt);

  GlobalVariable* newglob =
    new GlobalVariable(*(glob->getParent()), newt, glob->isConstant(), glob->getLinkage(), newinit);
  newglob->setAlignment(MaybeAlign(glob->getAlignment()));
  newglob->setName(glob->getName() + ".fixp");
  return newglob;
}

Constant* FloatToFixed::convertConstantAggregate(ConstantAggregate* cag,
                                                 std::shared_ptr<FixedPointType>& fixpt,
                                                 TypeMatchPolicy typepol) {
  std::vector<Constant*> consts;
  for (unsigned int i = 0; i < cag->getNumOperands(); i++) {
    Constant* oldconst = cag->getOperand(i);
    Constant* newconst = nullptr;
    if (getUnwrappedType(oldconst)->isFloatTy()) {
      newconst = convertConstant(cag->getOperand(i), fixpt, TypeMatchPolicy::ForceHint);
      if (!newconst)
        return nullptr;
    }
    else {
      newconst = oldconst;
    }
    consts.push_back(newconst);
  }

  if (isa<ConstantArray>(cag)) {
    ArrayType* aty = ArrayType::get(consts[0]->getType(), consts.size());
    return ConstantArray::get(aty, consts);
  }
  else if (isa<ConstantVector>(cag)) {
    return ConstantVector::get(consts);
  }
  else if (isa<ConstantStruct>(cag)) {
    std::vector<Type*> types;
    types.reserve(consts.size());
    for (Constant* c : consts)
      types.push_back(c->getType());
    StructType* strtype = StructType::get(cag->getContext(), types);
    return ConstantStruct::get(strtype, consts);
  }
  llvm_unreachable("a ConstantAggregate is not an array, vector or struct...");
}

template <class T>
Constant* FloatToFixed::createConstantDataSequential(ConstantDataSequential* cds,
                                                     const std::shared_ptr<FixedPointType>& fixpt) {
  std::vector<T> newConsts;

  for (unsigned int i = 0; i < cds->getNumElements(); i++) {
    APFloat thiselem = cds->getElementAsAPFloat(i);
    APSInt fixval;
    if (!convertAPFloat(thiselem, fixval, nullptr, std::static_ptr_cast<FixedPointScalarType>(fixpt))) {
      LLVM_DEBUG(dbgs() << *cds << " conv failed because an apfloat cannot be converted to " << *fixpt << "\n");
      return nullptr;
    }
    newConsts.push_back(fixval.getExtValue());
  }

  if (isa<ConstantDataArray>(cds))
    return ConstantDataArray::get(cds->getContext(), newConsts);
  return ConstantDataVector::get(cds->getContext(), newConsts);
}

template <class T>
Constant* FloatToFixed::createConstantDataSequentialFP(ConstantDataSequential* cds,
                                                       const std::shared_ptr<FixedPointType>& fixpt) {
  std::vector<T> newConsts;

  for (unsigned int i = 0; i < cds->getNumElements(); i++) {
    bool dontCare;

    APFloat thiselem = cds->getElementAsAPFloat(i);
    thiselem.convert(APFloat::IEEEdouble(), APFloatBase::rmTowardZero, &dontCare);
    newConsts.push_back(thiselem.convertToDouble());
  }

  if (isa<ConstantDataArray>(cds))
    return ConstantDataArray::get(cds->getContext(), newConsts);
  return ConstantDataVector::get(cds->getContext(), newConsts);
}

Constant* FloatToFixed::convertConstantDataSequential(ConstantDataSequential* cds,
                                                      const std::shared_ptr<FixedPointScalarType>& fixpt) {
  if (!getUnwrappedType(cds)->isFloatTy())
    return cds;

  if (fixpt->isFixedPoint()) {
    if (fixpt->getBits() <= 8)
      return createConstantDataSequential<uint8_t>(cds, fixpt);
    else if (fixpt->getBits() <= 16)
      return createConstantDataSequential<uint16_t>(cds, fixpt);
    else if (fixpt->getBits() <= 32)
      return createConstantDataSequential<uint32_t>(cds, fixpt);
    else if (fixpt->getBits() <= 64)
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

  LLVM_DEBUG(dbgs() << *fixpt << " too big for ConstantDataArray/Vector; 64 bit max\n");
  return nullptr;
}

Constant* FloatToFixed::convertLiteral(ConstantFP* fpc,
                                       Instruction* context,
                                       std::shared_ptr<FixedPointType>& fixpt,
                                       TypeMatchPolicy typepol) {
  APFloat val = fpc->getValueAPF();
  APSInt fixval;

  // Old workflow, convert the value to a fixed point value
  if (fixpt->isFixedPoint()) {
    if (!isHintPreferredPolicy(typepol)) {
      APFloat tmp(val);
      bool precise = false;
      tmp.convert(APFloatBase::IEEEdouble(), APFloat::rmTowardNegative, &precise);
      double dblval = tmp.convertToDouble();
      int nbits = std::static_ptr_cast<FixedPointScalarType>(fixpt)->getBits();
      Range range(dblval, dblval);
      int minflt = isMaxIntPolicy(typepol) ? -1 : 0;
      FixedPointInfo t = fixedPointTypeFromRange(range, nullptr, nbits, minflt);
      fixpt = std::make_shared<FixedPointScalarType>(&t);
    }

    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_ptr_cast<FixedPointScalarType>(fixpt);
    if (convertAPFloat(val, fixval, context, scalarFixpt)) {
      Type* intty = scalarFixpt->scalarToLLVMType(fpc->getContext());
      return ConstantInt::get(intty, fixval);
    }
    else {
      return nullptr;
    }
  }

  // Just "convert", actually recast, the value to the correct data type if using floating point data
  if (fixpt->isFloatingPoint()) {
    std::shared_ptr<FixedPointScalarType> scalarFixpt = std::static_pointer_cast<FixedPointScalarType>(fixpt);
    Type* intty = scalarFixpt->scalarToLLVMType(fpc->getContext());
    bool loosesInfo;

    val.convert(intty->getFltSemantics(), APFloatBase::rmTowardPositive, &loosesInfo);

    return ConstantFP::get(intty, val);
  }

  llvm_unreachable("We should have already covered all values, are you introducing a new data type?");
}

bool FloatToFixed::convertAPFloat(APFloat val,
                                  APSInt& fixval,
                                  Instruction* context,
                                  const std::shared_ptr<FixedPointScalarType>& fixpt) {
  bool precise = false;

  APFloat exp(pow(2.0, fixpt->getFractionalBits()));
  exp.convert(val.getSemantics(), APFloat::rmTowardNegative, &precise);
  val.multiply(exp, APFloat::rmTowardNegative);

  fixval = APSInt(fixpt->getBits(), !fixpt->isSigned());
  APFloat::opStatus cvtres = val.convertToInteger(fixval, APFloat::rmTowardNegative, &precise);

  if (cvtres != APFloat::opStatus::opOK && context) {
    SmallVector<char, 64> valstr;
    val.toString(valstr);
    std::string valstr2(valstr.begin(), valstr.end());
    OptimizationRemarkEmitter ORE(context->getFunction());
    if (cvtres == APFloat::opStatus::opInexact) {
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "ImpreciseConstConversion", context)
               << "fixed point conversion of constant " << valstr2 << " is not precise\n");
    }
    else {
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "ConstConversionFailed", context)
               << "impossible to convert constant " << valstr2 << " to fixed point\n");
      return false;
    }
  }

  return true;
}
