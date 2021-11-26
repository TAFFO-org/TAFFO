#include <sstream>
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include "TypeUtils.h"
#include "FixedPointType.h"
#include "LLVMFloatToFixedPass.h"


using namespace llvm;
using namespace flttofix;
using namespace mdutils;
using namespace taffo;


FixedPointType::FixedPointType()
{
  structData = nullptr;
  scalarData = {false, 0, 0};
}


FixedPointType::FixedPointType(bool s, int f, int b)
{
  structData = nullptr;
  scalarData = {s, f, b};
}


FixedPointType::FixedPointType(Type *llvmtype, bool signd)
{
  structData = nullptr;
  scalarData.isSigned = signd;
  if (isFloatType(llvmtype)) {
    scalarData.fracBitsAmt = 0;
    scalarData.bitsAmt = 0;
  } else if (llvmtype->isIntegerTy()) {
    scalarData.fracBitsAmt = 0;
    scalarData.bitsAmt = llvmtype->getIntegerBitWidth();
  } else {
    scalarData.isSigned = false;
    scalarData.fracBitsAmt = 0;
    scalarData.bitsAmt = 0;
  }
}


FixedPointType::FixedPointType(const ArrayRef<FixedPointType>& elems)
{
  structData.reset(new SmallVector<FixedPointType, 2>(elems.begin(), elems.end()));
}


FixedPointType::FixedPointType(TType *mdtype)
{
  structData = nullptr;
  FPType *fpt;
  if (mdtype && (fpt = dyn_cast<FPType>(mdtype))) {
    scalarData.bitsAmt = fpt->getWidth();
    scalarData.fracBitsAmt = fpt->getPointPos();
    scalarData.isSigned = fpt->isSigned();
  } else {
    scalarData = {false, 0, 0};
  }
}


FixedPointType FixedPointType::get(MDInfo *mdnfo, int *enableConversion)
{
  if (mdnfo == nullptr) {
    return FixedPointType();
    
  } else if (InputInfo *ii = dyn_cast<InputInfo>(mdnfo)) {
    if (ii->IEnableConversion) {
      if (enableConversion) (*enableConversion)++;
      return FixedPointType(ii->IType.get());
    } else {
      return FixedPointType();
    }
    
  } else if (StructInfo *si = dyn_cast<StructInfo>(mdnfo)) {
    SmallVector<FixedPointType, 2> elems;
    for (auto i = si->begin(); i != si->end(); i++) {
      elems.push_back(FixedPointType::get(i->get(), enableConversion));
    }
    return FixedPointType(elems);
    
  }
  assert("unknown type of MDInfo");
  return FixedPointType();
}


Type *FixedPointType::scalarToLLVMType(LLVMContext& ctxt) const
{
  assert(!structData && "fixed point type not a scalar");
  return Type::getIntNTy(ctxt, scalarData.bitsAmt);
}


std::string FixedPointType::Primitive::toString() const
{
  std::stringstream stm;
  if (isSigned)
    stm << "s";
  else
    stm << "u";
  stm << bitsAmt - fracBitsAmt << "_" << fracBitsAmt << "fixp";
  return stm.str();
}


std::string FixedPointType::toString() const
{
  std::stringstream stm;
  
  if (!structData) {
    stm << scalarData.toString();
  } else {
    stm << '<';
    for (int i=0; i<structData->size(); i++) {
      stm << (*structData)[i].toString();
      if (i != structData->size()-1)
        stm << ',';
    }
    stm << '>';
  }
  
  return stm.str();
}


FixedPointType FixedPointType::unwrapIndexList(Type *valType, const iterator_range<const Use *> indices)
{
  Type *resolvedType = valType;
  FixedPointType tempFixpt = *this;
  for (Value *a : indices) {
    if (resolvedType->isPointerTy()) {
      resolvedType = resolvedType->getPointerElementType();
    } else if (resolvedType->isArrayTy()) {
      resolvedType = resolvedType->getArrayElementType();
    } else if (resolvedType->isVectorTy()) {
      resolvedType = resolvedType->getVectorElementType();
    } else if (resolvedType->isStructTy()) {
      ConstantInt *val = dyn_cast<ConstantInt>(a);
      assert(val && "non-constant index for struct in GEP");
      int n = val->getZExtValue();
      resolvedType = resolvedType->getStructElementType(n);
      tempFixpt = tempFixpt.structItem(n);
    } else {
      assert("unsupported type in GEP");
    }
  }
  return tempFixpt;
}


FixedPointType FixedPointType::unwrapIndexList(Type *valType, ArrayRef<unsigned> indices)
{
  Type *resolvedType = valType;
  FixedPointType tempFixpt = *this;
  for (unsigned n : indices) {
    if (resolvedType->isPointerTy()) {
      resolvedType = resolvedType->getPointerElementType();
    } else if (resolvedType->isArrayTy()) {
      resolvedType = resolvedType->getArrayElementType();
    } else if (resolvedType->isVectorTy()) {
      resolvedType = resolvedType->getVectorElementType();
    } else if (resolvedType->isStructTy()) {
      resolvedType = resolvedType->getStructElementType(n);
      tempFixpt = tempFixpt.structItem(n);
    } else {
      assert("unsupported type in GEP");
    }
  }
  return tempFixpt;
}


raw_ostream& operator<<(raw_ostream& stm, const FixedPointType& f)
{
  stm << f.toString();
  return stm;
}


bool FixedPointType::operator==(const FixedPointType& rhs) const
{
  if (!structData) {
    return scalarData == rhs.scalarData;
  } else {
    if (structData->size() != rhs.structData->size())
      return false;
    for (int i=0; i<structData->size(); i++) {
      if (!((*structData)[i] == (*rhs.structData)[i]))
        return false;
    }
  }
  return true;
}

