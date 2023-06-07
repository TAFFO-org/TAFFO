#include "FixedPointType.h"
#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>


using namespace llvm;
using namespace flttofix;
using namespace mdutils;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"


FixedPointType::FixedPointType()
{
  structData = nullptr;
  scalarData = {false, 0, 0, FloatStandard::Float_NotFloat};
}


FixedPointType::FixedPointType(bool s, int f, int b)
{
  structData = nullptr;
  scalarData = {s, f, b, FloatStandard::Float_NotFloat};
}


FixedPointType::FixedPointType(Type *llvmtype, bool signd)
{
  structData = nullptr;
  scalarData.isSigned = signd;
  if (isFloatType(llvmtype)) {
    scalarData.fracBitsAmt = 0;
    scalarData.bitsAmt = 0;

    if (llvmtype->getTypeID() == Type::TypeID::HalfTyID) {
      scalarData.floatStandard = FloatStandard::Float_half;
    } else if (llvmtype->getTypeID() == Type::TypeID::DoubleTyID) {
      scalarData.floatStandard = FloatStandard::Float_double;
    } else if (llvmtype->getTypeID() == Type::TypeID::FloatTyID) {
      scalarData.floatStandard = FloatStandard::Float_float;
    } else if (llvmtype->getTypeID() == Type::TypeID::FP128TyID) {
      scalarData.floatStandard = FloatStandard::Float_fp128;
    } else if (llvmtype->getTypeID() == Type::TypeID::PPC_FP128TyID) {
      scalarData.floatStandard = FloatStandard::Float_ppc_fp128;
    } else if (llvmtype->getTypeID() == Type::TypeID::X86_FP80TyID) {
      scalarData.floatStandard = FloatStandard::Float_x86_fp80;
    } else if (llvmtype->getTypeID() == Type::TypeID::BFloatTyID) {
      scalarData.floatStandard = FloatStandard::Float_bfloat;
    } else {
      // Invalid...
      scalarData.floatStandard = FloatStandard::Float_NotFloat;
    }

  } else if (llvmtype->isIntegerTy()) {
    scalarData.fracBitsAmt = 0;
    scalarData.bitsAmt = llvmtype->getIntegerBitWidth();
    scalarData.floatStandard = FloatStandard::Float_NotFloat;
  } else {
    scalarData.isSigned = false;
    scalarData.fracBitsAmt = 0;
    scalarData.bitsAmt = 0;
    scalarData.floatStandard = FloatStandard::Float_NotFloat;
  }
}


FixedPointType::FixedPointType(const ArrayRef<FixedPointType> &elems)
{
  structData.reset(new SmallVector<FixedPointType, 2>(elems.begin(), elems.end()));
}


FixedPointType::FixedPointType(TType *mdtype)
{
  structData = nullptr;
  FPType *fpt;
  FloatType *flt;
  PositType *pst;
  if (mdtype && (fpt = dyn_cast<FPType>(mdtype))) {
    scalarData.bitsAmt = fpt->getWidth();
    scalarData.fracBitsAmt = fpt->getPointPos();
    scalarData.isSigned = fpt->isSigned();
    scalarData.floatStandard = FloatStandard::Float_NotFloat;
  } else if (mdtype && (flt = dyn_cast<FloatType>(mdtype))) {
    scalarData.bitsAmt = 0;
    scalarData.fracBitsAmt = 0;
    scalarData.isSigned = true;
    scalarData.floatStandard = static_cast<FloatStandard>(flt->getStandard());
  } else if (mdtype && (pst = dyn_cast<PositType>(mdtype))) {
    scalarData.bitsAmt = pst->getWidth();
    scalarData.fracBitsAmt = 0;
    scalarData.isSigned = true;
    scalarData.floatStandard = FloatStandard::Float_Posit;
  } else {
    scalarData = {false, 0, 0, FloatStandard::Float_NotFloat};
  }
}


FixedPointType FixedPointType::get(MDInfo *mdnfo, int *enableConversion)
{
  if (mdnfo == nullptr) {
    return FixedPointType();

  } else if (InputInfo *ii = dyn_cast<InputInfo>(mdnfo)) {
    if (ii->IEnableConversion) {
      if (enableConversion)
        (*enableConversion)++;
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
  assert(0 && "unknown type of MDInfo");
  return FixedPointType();
}


Type *FixedPointType::scalarToLLVMType(LLVMContext &ctxt) const
{
  assert(!structData && "fixed point type not a scalar");
  if (isFixedPoint()) {
    return Type::getIntNTy(ctxt, scalarData.bitsAmt);
  } else if (isPosit()) {
    StringRef className = "Posit" + std::to_string(scalarData.bitsAmt);
    StructType *classType = StructType::getTypeByName(ctxt, className);
    if (!classType) {
      classType = StructType::create(ctxt, className);
      classType->setBody({IntegerType::get(ctxt, scalarData.bitsAmt)});
    }
    return classType;
  } else {
    switch (scalarData.floatStandard) {
    case Float_half: /*16-bit floating-point value*/
      return Type::getHalfTy(ctxt);
    case Float_float: /*32-bit floating-point value*/
      return Type::getFloatTy(ctxt);
    case Float_double: /*64-bit floating-point value*/
      return Type::getDoubleTy(ctxt);
    case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/
      return Type::getFP128Ty(ctxt);
    case Float_x86_fp80: /*80-bit floating-point value (X87)*/
      return Type::getX86_FP80Ty(ctxt);
    case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
      return Type::getPPC_FP128Ty(ctxt);
    case Float_bfloat: /*128-bit floating-point value (two 64-bits)*/
      return Type::getBFloatTy(ctxt);
    case Float_NotFloat:
    default:
      dbgs() << "getting LLVMType of " << scalarData.floatStandard << "\n";
      llvm_unreachable("This should've been handled before");
    }
  }
}


std::string FixedPointType::Primitive::toString() const
{
  std::stringstream stm;
  if (floatStandard == Float_NotFloat) {
    if (isSigned)
      stm << "s";
    else
      stm << "u";

    stm << bitsAmt - fracBitsAmt << "_" << fracBitsAmt << "fixp";
  } else if (floatStandard == Float_Posit) {
    stm << "posit" << bitsAmt;
  } else {
    stm << floatStandard << "flp";
  }
  return stm.str();
}


std::string FixedPointType::toString() const
{
  std::stringstream stm;

  if (!structData) {
    stm << scalarData.toString();
  } else {
    stm << '<';
    for (size_t i = 0; i < structData->size(); i++) {
      stm << (*structData)[i].toString();
      if (i != structData->size() - 1)
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
      resolvedType = resolvedType->getContainedType(0);
    } else if (resolvedType->isStructTy()) {
      ConstantInt *val = dyn_cast<ConstantInt>(a);
      assert(val && "non-constant index for struct in GEP");
      int n = val->getZExtValue();
      resolvedType = resolvedType->getStructElementType(n);
      tempFixpt = tempFixpt.structItem(n);
    } else {
      assert(0 && "unsupported type in GEP");
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
      resolvedType = resolvedType->getContainedType(0);
    } else if (resolvedType->isStructTy()) {
      resolvedType = resolvedType->getStructElementType(n);
      tempFixpt = tempFixpt.structItem(n);
    } else {
      assert(0 && "unsupported type in GEP");
    }
  }
  return tempFixpt;
}


Type *FixedPointType::toLLVMType(Type *srct, bool *hasfloats) const
{
  // this == baset
  if (srct->isPointerTy()) {
    Type *enc = toLLVMType(srct->getPointerElementType(), hasfloats);
    if (enc)
      return enc->getPointerTo(srct->getPointerAddressSpace());
    return nullptr;

  } else if (srct->isArrayTy()) {
    int nel = srct->getArrayNumElements();
    Type *enc = toLLVMType(srct->getArrayElementType(), hasfloats);
    if (enc)
      return ArrayType::get(enc, nel);
    return nullptr;

  } else if (srct->isStructTy()) {
    SmallVector<Type *, 2> elems;
    bool allinvalid = true;
    for (unsigned i = 0; i < srct->getStructNumElements(); i++) {
      const FixedPointType &fpelemt = structItem(i);
      Type *baseelemt = srct->getStructElementType(i);
      Type *newelemt;
      if (!fpelemt.isInvalid()) {
        allinvalid = false;
        newelemt = fpelemt.toLLVMType(baseelemt, hasfloats);
      } else {
        newelemt = baseelemt;
      }
      elems.push_back(newelemt);
    }
    if (!allinvalid)
      return StructType::get(srct->getContext(), elems, dyn_cast<StructType>(srct)->isPacked());
    return srct;

  } else if (srct->isFloatingPointTy()) {
    if (hasfloats)
      *hasfloats = true;
    return scalarToLLVMType(srct->getContext());
  }
  
  LLVM_DEBUG(dbgs() << "FixedPointType::toLLVMType given unexpected non-float type " << *srct << "\n");
  if (hasfloats)
    *hasfloats = false;
  return srct;
}


raw_ostream &operator<<(raw_ostream &stm, const FixedPointType &f)
{
  stm << f.toString();
  return stm;
}


bool FixedPointType::operator==(const FixedPointType &rhs) const
{
  if (!structData) {
    return scalarData == rhs.scalarData;
  } else {
    if (structData->size() != rhs.structData->size())
      return false;
    for (size_t i = 0; i < structData->size(); i++) {
      if (!((*structData)[i] == (*rhs.structData)[i]))
        return false;
    }
  }
  return true;
}
