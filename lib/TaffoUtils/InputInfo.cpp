//===-- InputInfo.cpp - Data Structures for Input Info Metadata -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Various data structures that support in-memory representation of
/// input info metadata.
///
//===----------------------------------------------------------------------===//

#include "InputInfo.h"

#include "llvm/IR/Constants.h"
#include <cmath>

namespace mdutils
{

using namespace llvm;

std::unique_ptr<TType> TType::createFromMetadata(MDNode *MDN)
{
  if (FPType::isFPTypeMetadata(MDN))
    return FPType::createFromMetadata(MDN);

  if (FloatType::isFloatTypeMetadata(MDN))
    return FloatType::createFromMetadata(MDN);

  llvm_unreachable("Unsupported data type.");
}

bool TType::isTTypeMetadata(Metadata *MD)
{
  if (MDNode *MDN = dyn_cast_or_null<MDNode>(MD))
    // Keep in mind that here you should concatenate all the type
    return FPType::isFPTypeMetadata(MDN) || FloatType::isFloatTypeMetadata(MDN);
  else
    return false;
}

bool FPType::isFPTypeMetadata(MDNode *MDN)
{
  if (MDN->getNumOperands() < 1)
    return false;

  MDString *Flag = dyn_cast<MDString>(MDN->getOperand(0U).get());
  return Flag && Flag->getString().equals(FIXP_TYPE_FLAG);
}

bool FloatType::isFloatTypeMetadata(MDNode *MDN)
{
  if (MDN->getNumOperands() < 1)
    return false;

  MDString *Flag = dyn_cast<MDString>(MDN->getOperand(0U).get());
  return Flag && Flag->getString().equals(FLOAT_TYPE_FLAG);
}

std::unique_ptr<FPType> FPType::createFromMetadata(MDNode *MDN)
{
  assert(isFPTypeMetadata(MDN) && "Must be of fixp type.");
  assert(MDN->getNumOperands() >= 3U && "Must have flag, width, PointPos.");

  int Width;
  Metadata *WMD = MDN->getOperand(1U).get();
  ConstantAsMetadata *WCMD = cast<ConstantAsMetadata>(WMD);
  ConstantInt *WCI = cast<ConstantInt>(WCMD->getValue());
  Width = WCI->getSExtValue();

  unsigned PointPos;
  Metadata *PMD = MDN->getOperand(2U).get();
  ConstantAsMetadata *PCMD = cast<ConstantAsMetadata>(PMD);
  ConstantInt *PCI = cast<ConstantInt>(PCMD->getValue());
  PointPos = PCI->getZExtValue();

  return std::unique_ptr<FPType>(new FPType(Width, PointPos));
}


MDNode *FPType::toMetadata(LLVMContext &C) const
{
  Metadata *TypeFlag = MDString::get(C, FIXP_TYPE_FLAG);

  IntegerType *Int32Ty = Type::getInt32Ty(C);
  ConstantInt *WCI = ConstantInt::getSigned(Int32Ty, this->getSWidth());
  Metadata *WidthMD = ConstantAsMetadata::get(WCI);

  ConstantInt *PCI = ConstantInt::get(Int32Ty, this->getPointPos());
  ConstantAsMetadata *PointPosMD = ConstantAsMetadata::get(PCI);

  Metadata *MDs[] = {TypeFlag, WidthMD, PointPosMD};
  return MDNode::get(C, MDs);
}


double FPType::getRoundingError() const
{
  return std::ldexp(1.0, -this->getPointPos());
}

llvm::APFloat FPType::getMinValueBound() const
{
  if (isSigned()) {
    return llvm::APFloat(std::ldexp(-1.0, getWidth() - getPointPos() - 1));
  } else {
    return llvm::APFloat((double)0.0);
  }
}

llvm::APFloat FPType::getMaxValueBound() const
{
  int MaxIntExp = (isSigned()) ? getWidth() - 1 : getWidth();
  double MaxIntPlus1 = std::ldexp(1.0, MaxIntExp);
  double MaxInt = MaxIntPlus1 - 1.0;
  if (MaxInt == MaxIntPlus1)
    MaxInt = std::nextafter(MaxInt, 0.0);
  return llvm::APFloat(std::ldexp(MaxInt, -getPointPos()));
}

Metadata *createDoubleMetadata(LLVMContext &C, double Value)
{
  Type *DoubleTy = Type::getDoubleTy(C);
  Constant *ValC = ConstantFP::get(DoubleTy, Value);
  return ConstantAsMetadata::get(ValC);
}

MDNode *createDoubleMDNode(LLVMContext &C, double Value)
{
  return MDNode::get(C, createDoubleMetadata(C, Value));
}

double retrieveDoubleMetadata(Metadata *DMD)
{
  ConstantAsMetadata *DCMD = cast<ConstantAsMetadata>(DMD);
  ConstantFP *DCFP = cast<ConstantFP>(DCMD->getValue());
  return DCFP->getValueAPF().convertToDouble();
}

double retrieveDoubleMDNode(MDNode *MDN)
{
  assert(MDN != nullptr);
  assert(MDN->getNumOperands() > 0 && "Must have at least one operand.");

  return retrieveDoubleMetadata(MDN->getOperand(0U).get());
}

std::unique_ptr<Range> Range::createFromMetadata(MDNode *MDN)
{
  assert(MDN != nullptr);
  assert(MDN->getNumOperands() == 2U && "Must contain Min and Max.");

  double Min = retrieveDoubleMetadata(MDN->getOperand(0U).get());
  double Max = retrieveDoubleMetadata(MDN->getOperand(1U).get());
  return std::unique_ptr<Range>(new Range(Min, Max));
}

bool Range::isRangeMetadata(Metadata *MD)
{
  MDNode *MDN = dyn_cast_or_null<MDNode>(MD);
  return MDN != nullptr && MDN->getNumOperands() == 2U && isa<ConstantAsMetadata>(MDN->getOperand(0U).get()) && isa<ConstantAsMetadata>(MDN->getOperand(1U).get());
}

MDNode *Range::toMetadata(LLVMContext &C) const
{
  Metadata *RangeMD[] = {createDoubleMetadata(C, this->Min),
                         createDoubleMetadata(C, this->Max)};
  return MDNode::get(C, RangeMD);
}

std::unique_ptr<double> CreateInitialErrorFromMetadata(MDNode *MDN)
{
  return std::unique_ptr<double>(new double(retrieveDoubleMDNode(MDN)));
}

MDNode *InitialErrorToMetadata(LLVMContext &C, double Error)
{
  return createDoubleMDNode(C, Error);
}

bool IsInitialErrorMetadata(Metadata *MD)
{
  MDNode *MDN = dyn_cast_or_null<MDNode>(MD);
  if (MDN == nullptr || MDN->getNumOperands() != 1U)
    return false;

  return isa<ConstantAsMetadata>(MDN->getOperand(0U).get());
}

MDNode *InputInfo::toMetadata(LLVMContext &C) const
{
  Metadata *Null = ConstantAsMetadata::get(ConstantInt::getFalse(C));
  Metadata *TypeMD = (IType) ? IType->toMetadata(C) : Null;
  Metadata *RangeMD = (IRange) ? IRange->toMetadata(C) : Null;
  Metadata *ErrorMD = (IError) ? InitialErrorToMetadata(C, *IError) : Null;

  uint64_t Flags = IEnableConversion | (IFinal << 1);
  Metadata *FlagsMD = ConstantAsMetadata::get(ConstantInt::get(Type::getIntNTy(C, 2U), Flags));

  Metadata *InputMDs[] = {TypeMD, RangeMD, ErrorMD, FlagsMD};
  return MDNode::get(C, InputMDs);
}

bool InputInfo::isInputInfoMetadata(Metadata *MD)
{
  MDNode *MDN = dyn_cast<MDNode>(MD);
  if (MDN == nullptr || (MDN->getNumOperands() != 4U && MDN->getNumOperands() != 3U))
    return false;

  Metadata *Op0 = MDN->getOperand(0U).get();
  if (!(IsNullInputInfoField(Op0) || TType::isTTypeMetadata(Op0)))
    return false;

  Metadata *Op1 = MDN->getOperand(1U).get();
  if (!(IsNullInputInfoField(Op1) || Range::isRangeMetadata(Op1)))
    return false;

  Metadata *Op2 = MDN->getOperand(2U).get();
  if (!(IsNullInputInfoField(Op2) || IsInitialErrorMetadata(Op2)))
    return false;

  if (MDN->getNumOperands() == 4U) {
    Metadata *Op3 = MDN->getOperand(3U).get();
    if (!(IsNullInputInfoField(Op3) || isa<ConstantAsMetadata>(Op3)))
      return false;
  }

  return true;
}

MDNode *StructInfo::toMetadata(LLVMContext &C) const
{
  Metadata *Null = ConstantAsMetadata::get(ConstantInt::getFalse(C));
  SmallVector<Metadata *, 4U> FieldMDs;
  FieldMDs.reserve(Fields.size());
  for (std::shared_ptr<MDInfo> MDI : Fields) {
    FieldMDs.push_back((MDI) ? MDI->toMetadata(C) : Null);
  }
  return MDNode::get(C, FieldMDs);
}

std::unique_ptr<CmpErrorInfo> CmpErrorInfo::createFromMetadata(MDNode *MDN)
{
  if (MDN == nullptr)
    return std::unique_ptr<CmpErrorInfo>(new CmpErrorInfo(0.0, false));

  double MaxTol = retrieveDoubleMDNode(MDN);
  return std::unique_ptr<CmpErrorInfo>(new CmpErrorInfo(MaxTol));
}

MDNode *CmpErrorInfo::toMetadata(LLVMContext &C) const
{
  return createDoubleMDNode(C, MaxTolerance);
}

bool IsNullInputInfoField(Metadata *MD)
{
  ConstantAsMetadata *CMD = dyn_cast<ConstantAsMetadata>(MD);
  if (CMD == nullptr)
    return false;

  ConstantInt *CI = dyn_cast<ConstantInt>(CMD->getValue());
  if (CI == nullptr)
    return false;

  return CI->isZero() && CI->getBitWidth() == 1U;
}

std::string FloatType::getFloatStandardName(FloatType::FloatStandard standard)
{
  switch (standard) {
  case Float_half: /*16-bit floating-point value*/
    return "Float_half";
  case Float_float: /*32-bit floating-point value*/
    return "Float_float";
  case Float_double: /*64-bit floating-point value*/
    return "Float_double";
  case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/
    return "Float_fp128";
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/
    return "Float_x86_fp80";
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    return "Float_ppc_fp128";
  case Float_bfloat: /*bfloat floating point value)*/
    return "Float_bfloat";
  }
  llvm_unreachable("[TAFFO] Unknown FloatType standard!");
  return "[UNKNOWN TYPE]";
}

llvm::Type::TypeID FloatType::getLLVMTypeID() const
{
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/
    return llvm::Type::TypeID::HalfTyID;
  case Float_float: /*32-bit floating-point value*/
    return llvm::Type::TypeID::FloatTyID;
  case Float_double: /*64-bit floating-point value*/
    return llvm::Type::TypeID::DoubleTyID;
  case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/
    return llvm::Type::TypeID::FP128TyID;
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/
    return llvm::Type::TypeID::X86_FP80TyID;
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    return llvm::Type::TypeID::PPC_FP128TyID;
  case Float_bfloat: /*bfloat floating point value)*/
    return llvm::Type::TypeID::BFloatTyID;
  }
  llvm_unreachable("[TAFFO] Unknown FloatType standard!");
}

// FIXME: some values are not computed correctly because we can not!
llvm::APFloat FloatType::getMinValueBound() const
{
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEhalf(), true);
  case Float_float: /*32-bit floating-point value*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEsingle(), true);
  case Float_double: /*64-bit floating-point value*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEdouble(), true);
  case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEquad(), true);
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/
    return llvm::APFloat::getLargest(llvm::APFloat::x87DoubleExtended(), true);
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    return llvm::APFloat::getLargest(llvm::APFloat::PPCDoubleDouble(), true);
  case Float_bfloat: /*bfloat floating point value)*/
    return llvm::APFloat::getLargest(llvm::APFloat::BFloat(), true);
  }
  llvm_unreachable("[TAFFO] Unknown FloatType standard!");
}

// FIXME: some values are not computed correctly because we can not!
llvm::APFloat FloatType::getMaxValueBound() const
{
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEhalf(), false);
  case Float_float: /*32-bit floating-point value*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEsingle(), false);
  case Float_double: /*64-bit floating-point value*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEdouble(), false);
  case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/
    return llvm::APFloat::getLargest(llvm::APFloat::IEEEquad(), false);
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/
    return llvm::APFloat::getLargest(llvm::APFloat::x87DoubleExtended(), false);
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    return llvm::APFloat::getLargest(llvm::APFloat::PPCDoubleDouble(), false);
  case Float_bfloat: /*bfloat floating point value)*/
    return llvm::APFloat::getLargest(llvm::APFloat::BFloat(), false);
  }

  llvm_unreachable("Unknown limit for this float type");
}

// FIXME: this can give incorrect results if used in corner cases
double FloatType::getRoundingError() const
{
  int p = getP();

  // Computing the exponent value
  double k = floor(log2(this->greatestNunber));

  // given that epsilon is the maximum error achievable given a certain amount of bit in mantissa (p) on the mantissa itself
  // it will be multiplied by the exponent, that will be at most 2^k
  // BTW we are probably carrying some type of error here Hehehe
  // Complete formula -> epsilon * exponent_value
  // that is (beta/2)*(b^-p)     *     b^k
  // thus (beta/2) b*(k-p)
  // given beta = 2 on binary machines (so I hope the target one is binary too...)
  return exp2(k - p);
}

// This function will return the number of bits in the mantissa
int FloatType::getP() const
{
  // The plus 1 is due to the fact that there is always an implicit 1 stored (the d_0 value)
  // Therefore, we have actually 1 bit more wrt the ones stored
  int p;
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/
    p = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEhalf());
    break;
  case Float_float: /*32-bit floating-point value*/
    p = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEsingle());
    break;
  case Float_double: /*64-bit floating-point value*/
    p = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEdouble());
    break;
  case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/
    p = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEquad());
    break;
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/
    // But in this case, it has a fractionary part of 63 and an "integer" part of 1, total 64 for the significand
    p = llvm::APFloat::semanticsPrecision(llvm::APFloat::x87DoubleExtended());
    break;
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    p = llvm::APFloat::semanticsPrecision(llvm::APFloat::PPCDoubleDouble());
    break;
  case Float_bfloat: /*128-bit floating-point value (two 64-bits)*/
    p = llvm::APFloat::semanticsPrecision(llvm::APFloat::BFloat());
    break;
  }

  return p;
}

std::unique_ptr<FloatType> FloatType::createFromMetadata(MDNode *MDN)
{
  assert(isFloatTypeMetadata(MDN) && "Must be of float type.");
  assert(MDN->getNumOperands() >= 3U && "Must have flag, FloatType.");

  int Width;
  Metadata *WMD = MDN->getOperand(1U).get();
  ConstantAsMetadata *WCMD = cast<ConstantAsMetadata>(WMD);
  ConstantInt *WCI = cast<ConstantInt>(WCMD->getValue());
  Width = WCI->getSExtValue();

  FloatStandard type = static_cast<FloatStandard>(Width);

  double GreatestNumber;
  Metadata *WMD2 = MDN->getOperand(2U).get();
  GreatestNumber = retrieveDoubleMetadata(WMD2);

  return std::unique_ptr<FloatType>(new FloatType(type, GreatestNumber));
}

MDNode *FloatType::toMetadata(LLVMContext &C) const
{
  Metadata *TypeFlag = MDString::get(C, FLOAT_TYPE_FLAG);

  IntegerType *Int32Ty = Type::getInt32Ty(C);
  ConstantInt *WCI = ConstantInt::getSigned(Int32Ty, this->standard);
  Metadata *floatType = ConstantAsMetadata::get(WCI);

  Metadata *greatestValue = createDoubleMetadata(C, this->greatestNunber);

  Metadata *MDs[] = {TypeFlag, floatType, greatestValue};
  return MDNode::get(C, MDs);
}


} // end namespace mdutils
