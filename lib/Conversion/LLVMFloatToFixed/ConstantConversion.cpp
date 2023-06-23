#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"
#include "PositConstant.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
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

#define DEBUG_TYPE "taffo-conversion"

Constant *FloatToFixed::convertConstant(Constant *flt, FixedPointType &fixpt,
                                        TypeMatchPolicy typepol)
{
  if (dyn_cast<UndefValue>(flt)) {
    return UndefValue::get(
        getLLVMFixedPointTypeForFloatType(flt->getType(), fixpt));
  }
  if (GlobalVariable *gvar = dyn_cast<GlobalVariable>(flt)) {
    return convertGlobalVariable(gvar, fixpt);
  } else if (ConstantFP *fpc = dyn_cast<ConstantFP>(flt)) {
    return convertLiteral(fpc, nullptr, fixpt, typepol);
  } else if (ConstantAggregate *cag = dyn_cast<ConstantAggregate>(flt)) {
    return convertConstantAggregate(cag, fixpt, typepol);
  } else if (ConstantDataSequential *cds =
                 dyn_cast<ConstantDataSequential>(flt)) {
    return convertConstantDataSequential(cds, fixpt);
  } else if (dyn_cast<ConstantAggregateZero>(flt)) {
    Type *newt = getLLVMFixedPointTypeForFloatType(flt->getType(), fixpt);
    return ConstantAggregateZero::get(newt);
  } else if (ConstantExpr *cexp = dyn_cast<ConstantExpr>(flt)) {
    return convertConstantExpr(cexp, fixpt, typepol);
  }
  return nullptr;
}

Constant *FloatToFixed::convertConstantExpr(ConstantExpr *cexp,
                                            FixedPointType &fixpt,
                                            TypeMatchPolicy typepol)
{
  if (isa<GEPOperator>(cexp)) {
    Value *newval = operandPool[cexp->getOperand(0)];
    if (!newval) {
      LLVM_DEBUG(dbgs() << "[Warning] Operand of constant GEP not found in operandPool!\n");
      return nullptr;
    }
    Constant *newconst = dyn_cast<Constant>(newval);
    if (!newconst)
      return nullptr;

    if (typepol == TypeMatchPolicy::ForceHint)
      assert(fixpt == fixPType(newval) && "type adjustment forbidden...");
    else
      fixpt = fixPType(newval);

    std::vector<Constant *> vals;
    for (unsigned int i = 1; i < cexp->getNumOperands(); i++) {
      vals.push_back(cexp->getOperand(i));
    }

    ArrayRef<Constant *> idxlist(vals);
    return ConstantExpr::getInBoundsGetElementPtr(newconst->getType()->getPointerElementType(), newconst, idxlist);

  } else {
    LLVM_DEBUG(dbgs() << "constant expression " << *cexp << " is not handled explicitly yet\n");
  }
  return nullptr;
}

Constant *FloatToFixed::convertGlobalVariable(GlobalVariable *glob,
                                              FixedPointType &fixpt)
{
  bool hasfloats = false;
  Type *prevt = glob->getType()->getPointerElementType();
  Type *newt = getLLVMFixedPointTypeForFloatType(prevt, fixpt, &hasfloats);
  if (!newt)
    return nullptr;
  if (!hasfloats)
    return glob;

  Constant *oldinit = glob->getInitializer();
  Constant *newinit = nullptr;
  if (oldinit && !oldinit->isNullValue()) {
    /* global variables can be written to, so we always convert them to the type allocated by the DTA */
    newinit = convertConstant(oldinit, fixpt, TypeMatchPolicy::ForceHint);
  } else
    newinit = Constant::getNullValue(newt);

  GlobalVariable *newglob = new GlobalVariable(*(glob->getParent()), newt, glob->isConstant(), glob->getLinkage(),
                                               newinit);
  newglob->setAlignment(llvm::MaybeAlign(glob->getAlignment()));
  newglob->setName(glob->getName() + ".fixp");
  return newglob;
}


Constant *
FloatToFixed::convertConstantAggregate(ConstantAggregate *cag, FixedPointType &fixpt, TypeMatchPolicy typepol)
{
  std::vector<Constant *> consts;
  for (unsigned int i = 0; i < cag->getNumOperands(); i++) {
    Constant *oldconst = cag->getOperand(i);
    Constant *newconst = nullptr;
    if (isFloatType(oldconst->getType())) {
      newconst = convertConstant(cag->getOperand(i), fixpt, TypeMatchPolicy::ForceHint);
      if (!newconst)
        return nullptr;
    } else {
      newconst = oldconst;
    }
    consts.push_back(newconst);
  }

  if (isa<ConstantArray>(cag)) {
    ArrayType *aty = ArrayType::get(consts[0]->getType(), consts.size());
    return ConstantArray::get(aty, consts);

  } else if (isa<ConstantVector>(cag)) {
    return ConstantVector::get(consts);

  } else if (isa<ConstantStruct>(cag)) {
    std::vector<Type *> types;
    types.reserve(consts.size());
    for (Constant *c : consts) {
      types.push_back(c->getType());
    }
    StructType *strtype = StructType::get(cag->getContext(), types);
    return ConstantStruct::get(strtype, consts);
  }
  llvm_unreachable("a ConstantAggregate is not an array, vector or struct...");
}

template <class T>
Constant *
FloatToFixed::createConstantDataSequential(ConstantDataSequential *cds,
                                           const FixedPointType &fixpt)
{
  std::vector<T> newConsts;

  for (unsigned int i = 0; i < cds->getNumElements(); i++) {
    APFloat thiselem = cds->getElementAsAPFloat(i);
    APSInt fixval;
    if (!convertAPFloat(thiselem, fixval, nullptr, fixpt)) {
      LLVM_DEBUG(dbgs() << *cds << " conv failed because an apfloat cannot be converted to " << fixpt << "\n");
      return nullptr;
    }
    newConsts.push_back(fixval.getExtValue());
  }

  if (isa<ConstantDataArray>(cds)) {
    return ConstantDataArray::get(cds->getContext(), newConsts);
  }
  return ConstantDataVector::get(cds->getContext(), newConsts);
}

template <class T>
Constant *FloatToFixed::createConstantDataSequentialFP(ConstantDataSequential *cds, const FixedPointType &fixpt)
{
  std::vector<T> newConsts;
  Type *dstType = fixpt.scalarToLLVMType(cds->getContext());

  for (unsigned int i = 0; i < cds->getNumElements(); i++) {
    bool dontCare;

    APFloat thiselem = cds->getElementAsAPFloat(i);
    thiselem.convert(dstType->getFltSemantics(), APFloatBase::rmTowardPositive, &dontCare);
    T byteRepr = thiselem.bitcastToAPInt().getLimitedValue();
    newConsts.push_back(byteRepr);
  }

  if (isa<ConstantDataArray>(cds)) {
    return ConstantDataArray::getFP(dstType, newConsts);
  }
  return ConstantDataVector::getFP(dstType, newConsts);
}


Constant *FloatToFixed::convertConstantDataSequential(ConstantDataSequential *cds, const FixedPointType &fixpt)
{
  if (!isFloatType(cds->getElementType()))
    return cds;

  if (fixpt.isFixedPoint()) {
    if (fixpt.scalarBitsAmt() <= 8)
      return createConstantDataSequential<uint8_t>(cds, fixpt);
    else if (fixpt.scalarBitsAmt() <= 16)
      return createConstantDataSequential<uint16_t>(cds, fixpt);
    else if (fixpt.scalarBitsAmt() <= 32)
      return createConstantDataSequential<uint32_t>(cds, fixpt);
    else if (fixpt.scalarBitsAmt() <= 64)
      return createConstantDataSequential<uint64_t>(cds, fixpt);
  }

  if (fixpt.isFloatingPoint()) {
    if (fixpt.getFloatingPointStandard() == FixedPointType::Float_float) {
      return createConstantDataSequentialFP<uint32_t>(cds, fixpt);
    }

    if (fixpt.getFloatingPointStandard() == FixedPointType::Float_double) {
      return createConstantDataSequentialFP<uint64_t>(cds, fixpt);
    }

    if (fixpt.getFloatingPointStandard() == FixedPointType::Float_half) {
      return createConstantDataSequentialFP<uint16_t>(cds, fixpt);
    }

    if (fixpt.getFloatingPointStandard() == FixedPointType::Float_bfloat) {
      return createConstantDataSequentialFP<uint16_t>(cds, fixpt);
    }

    llvm_unreachable("Unsupported float constant conversion");
  }

  if (fixpt.isPosit()) {
    std::vector<Constant *> posits;
    for (unsigned int i = 0; i < cds->getNumElements(); i++) {
      APFloat thiselem = cds->getElementAsAPFloat(i);
      posits.push_back(PositConstant::get(cds->getContext(), fixpt, thiselem.convertToDouble()));
    }
    if (isa<ConstantDataArray>(cds)) {
      ArrayType *t = ArrayType::get(fixpt.scalarToLLVMType(cds->getContext()), cds->getNumElements());
      return ConstantArray::get(t, posits);
    }
    return ConstantVector::get(posits);
  }

  LLVM_DEBUG(dbgs() << fixpt << " too big for ConstantDataArray/Vector; 64 bit max\n");
  return nullptr;
}


Constant *
FloatToFixed::convertLiteral(ConstantFP *fpc, Instruction *context, FixedPointType &fixpt, TypeMatchPolicy typepol)
{
  APFloat val = fpc->getValueAPF();
  APSInt fixval;

  if (fixpt.isFixedPoint() || fixpt.isPosit()) {
    if (!isHintPreferredPolicy(typepol)) {
      APFloat tmp(val);
      bool precise = false;
      tmp.convert(APFloatBase::IEEEdouble(), APFloat::rmTowardNegative, &precise);
      double dblval = tmp.convertToDouble();
      int nbits = fixpt.scalarBitsAmt();
      mdutils::Range range(dblval, dblval);
      int minflt = isMaxIntPolicy(typepol) ? -1 : 0;
      if (fixpt.isFixedPoint()) {
        mdutils::FPType t = taffo::fixedPointTypeFromRange(range, nullptr, nbits, minflt);
        fixpt = FixedPointType(&t);
      } else { // fixpt.isPosit()
        mdutils::PositType t = taffo::positTypeFromRange(range);
        fixpt = FixedPointType(&t);
      }
    }
  }

  if (fixpt.isFixedPoint()) {
    if (convertAPFloat(val, fixval, context, fixpt)) {
      Type *intty = fixpt.scalarToLLVMType(fpc->getContext());
      return ConstantInt::get(intty, fixval);
    } else {
      return nullptr;
    }
  }

  if (fixpt.isPosit()) {
    return PositConstant::get(fpc->getContext(), fixpt, val.convertToDouble());
  }

  // Just "convert", actually recast, the value to the correct data type if using floating point data
  if (fixpt.isFloatingPoint()) {
    Type *intty = fixpt.scalarToLLVMType(fpc->getContext());
    bool loosesInfo;

    val.convert(intty->getFltSemantics(), llvm::APFloatBase::rmTowardPositive, &loosesInfo);

    return ConstantFP::get(intty, val);
  }

  llvm_unreachable("We should have already covered all values, are you introducing a new data type?");
}


bool FloatToFixed::convertAPFloat(APFloat val, APSInt &fixval, Instruction *context, const FixedPointType &fixpt)
{
  bool precise = false;

  APFloat exp(pow(2.0, fixpt.scalarFracBitsAmt()));
  exp.convert(val.getSemantics(), APFloat::rmTowardNegative, &precise);
  val.multiply(exp, APFloat::rmTowardNegative);

  fixval = APSInt(fixpt.scalarBitsAmt(), !fixpt.scalarIsSigned());
  APFloat::opStatus cvtres = val.convertToInteger(fixval, APFloat::rmTowardNegative, &precise);

  if (cvtres != APFloat::opStatus::opOK && context) {
    SmallVector<char, 64> valstr;
    val.toString(valstr);
    std::string valstr2(valstr.begin(), valstr.end());
    OptimizationRemarkEmitter ORE(context->getFunction());
    if (cvtres == APFloat::opStatus::opInexact) {
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "ImpreciseConstConversion", context) << "fixed point conversion of constant "
                                                                                   << valstr2
                                                                                   << " is not precise\n");
    } else {
      ORE.emit(OptimizationRemark(DEBUG_TYPE, "ConstConversionFailed", context) << "impossible to convert constant "
                                                                                << valstr2
                                                                                << " to fixed point\n");
      return false;
    }
  }

  return true;
}
