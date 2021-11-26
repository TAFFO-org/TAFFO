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

#include <cmath>
#include "llvm/IR/Constants.h"

namespace mdutils {

using namespace llvm;

std::unique_ptr<TType> TType::createFromMetadata(MDNode *MDN) {
  if (FPType::isFPTypeMetadata(MDN))
    return FPType::createFromMetadata(MDN);

  llvm_unreachable("Unsupported data type.");
}

bool TType::isTTypeMetadata(Metadata *MD) {
  if (MDNode *MDN = dyn_cast_or_null<MDNode>(MD))
    return FPType::isFPTypeMetadata(MDN);
  else
    return false;
}

bool FPType::isFPTypeMetadata(MDNode *MDN) {
  if (MDN->getNumOperands() < 1)
    return false;

  MDString *Flag = dyn_cast<MDString>(MDN->getOperand(0U).get());
  return Flag && Flag->getString().equals(FIXP_TYPE_FLAG);
}

std::unique_ptr<FPType> FPType::createFromMetadata(MDNode *MDN) {
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

MDNode *FPType::toMetadata(LLVMContext &C) const {
  Metadata *TypeFlag = MDString::get(C, FIXP_TYPE_FLAG);

  IntegerType *Int32Ty = Type::getInt32Ty(C);
  ConstantInt *WCI = ConstantInt::getSigned(Int32Ty, this->getSWidth());
  Metadata *WidthMD = ConstantAsMetadata::get(WCI);

  ConstantInt *PCI = ConstantInt::get(Int32Ty, this->getPointPos());
  ConstantAsMetadata *PointPosMD = ConstantAsMetadata::get(PCI);

  Metadata *MDs[] = {TypeFlag, WidthMD, PointPosMD};
  return MDNode::get(C, MDs);
}

double FPType::getRoundingError() const {
  return std::ldexp(1.0, -this->getPointPos());
}

double FPType::getMinValueBound() const {
  if (isSigned()) {
    return std::ldexp(-1.0, getWidth() - getPointPos() - 1);
  }
  else {
    return 0.0;
  }
}

double FPType::getMaxValueBound() const  {
  int MaxIntExp = (isSigned()) ? getWidth() - 1 : getWidth();
  double MaxIntPlus1 = std::ldexp(1.0, MaxIntExp);
  double MaxInt = MaxIntPlus1 - 1.0;
  if (MaxInt == MaxIntPlus1)
    MaxInt = std::nextafter(MaxInt, 0.0);
  return std::ldexp(MaxInt, -getPointPos());
}

Metadata *createDoubleMetadata(LLVMContext &C, double Value) {
  Type *DoubleTy = Type::getDoubleTy(C);
  Constant *ValC = ConstantFP::get(DoubleTy, Value);
  return ConstantAsMetadata::get(ValC);
}

MDNode *createDoubleMDNode(LLVMContext &C, double Value) {
  return MDNode::get(C, createDoubleMetadata(C, Value));
}

double retrieveDoubleMetadata(Metadata *DMD) {
  ConstantAsMetadata *DCMD = cast<ConstantAsMetadata>(DMD);
  ConstantFP *DCFP = cast<ConstantFP>(DCMD->getValue());
  return DCFP->getValueAPF().convertToDouble();
}

double retrieveDoubleMDNode(MDNode *MDN) {
  assert(MDN != nullptr);
  assert(MDN->getNumOperands() > 0 && "Must have at least one operand.");

  return retrieveDoubleMetadata(MDN->getOperand(0U).get());
}

std::unique_ptr<Range> Range::createFromMetadata(MDNode *MDN) {
  assert(MDN != nullptr);
  assert(MDN->getNumOperands() == 2U && "Must contain Min and Max.");

  double Min = retrieveDoubleMetadata(MDN->getOperand(0U).get());
  double Max = retrieveDoubleMetadata(MDN->getOperand(1U).get());
  return std::unique_ptr<Range>(new Range(Min, Max));
}

bool Range::isRangeMetadata(Metadata *MD) {
  MDNode *MDN = dyn_cast_or_null<MDNode>(MD);
  return MDN != nullptr
    && MDN->getNumOperands() == 2U
    && isa<ConstantAsMetadata>(MDN->getOperand(0U).get())
    && isa<ConstantAsMetadata>(MDN->getOperand(1U).get());
}

MDNode *Range::toMetadata(LLVMContext &C) const {
  Metadata *RangeMD[] = {createDoubleMetadata(C, this->Min),
			 createDoubleMetadata(C, this->Max)};
  return MDNode::get(C, RangeMD);
}

std::unique_ptr<double> CreateInitialErrorFromMetadata(MDNode *MDN) {
  return std::unique_ptr<double>(new double(retrieveDoubleMDNode(MDN)));
}

MDNode *InitialErrorToMetadata(LLVMContext &C, double Error) {
    return createDoubleMDNode(C, Error);
}

bool IsInitialErrorMetadata(Metadata *MD) {
  MDNode *MDN = dyn_cast_or_null<MDNode>(MD);
  if (MDN == nullptr || MDN->getNumOperands() != 1U)
    return false;

  return isa<ConstantAsMetadata>(MDN->getOperand(0U).get());
}

MDNode *InputInfo::toMetadata(LLVMContext &C) const {
  Metadata *Null = ConstantAsMetadata::get(ConstantInt::getFalse(C));
  Metadata *TypeMD = (IType) ? IType->toMetadata(C) : Null;
  Metadata *RangeMD = (IRange) ? IRange->toMetadata(C) : Null;
  Metadata *ErrorMD = (IError) ? InitialErrorToMetadata(C, *IError) : Null;

  uint64_t Flags = IEnableConversion | (IFinal << 1);
  Metadata *FlagsMD = ConstantAsMetadata::get(ConstantInt::get(Type::getIntNTy(C, 2U), Flags));

  Metadata *InputMDs[] = {TypeMD, RangeMD, ErrorMD, FlagsMD};
  return MDNode::get(C, InputMDs);
}

bool InputInfo::isInputInfoMetadata(Metadata *MD) {
  MDNode *MDN = dyn_cast<MDNode>(MD);
  if (MDN == nullptr || MDN->getNumOperands() != 4U)
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

  Metadata *Op3 = MDN->getOperand(3U).get();
  if (!(IsNullInputInfoField(Op3) || isa<ConstantAsMetadata>(Op3)))
    return false;

  return true;
}

MDNode *StructInfo::toMetadata(LLVMContext &C) const {
  Metadata *Null = ConstantAsMetadata::get(ConstantInt::getFalse(C));
  SmallVector<Metadata *, 4U> FieldMDs;
  FieldMDs.reserve(Fields.size());
  for (std::shared_ptr<MDInfo> MDI : Fields) {
    FieldMDs.push_back((MDI) ? MDI->toMetadata(C) : Null);
  }
  return MDNode::get(C, FieldMDs);
}

std::unique_ptr<CmpErrorInfo> CmpErrorInfo::createFromMetadata(MDNode *MDN) {
  if (MDN == nullptr)
    return std::unique_ptr<CmpErrorInfo>(new CmpErrorInfo(0.0, false));

  double MaxTol = retrieveDoubleMDNode(MDN);
  return std::unique_ptr<CmpErrorInfo>(new CmpErrorInfo(MaxTol));
}

MDNode *CmpErrorInfo::toMetadata(LLVMContext &C) const {
  return createDoubleMDNode(C, MaxTolerance);
}

bool IsNullInputInfoField(Metadata *MD) {
  ConstantAsMetadata *CMD = dyn_cast<ConstantAsMetadata>(MD);
  if (CMD == nullptr)
    return false;

  ConstantInt *CI = dyn_cast<ConstantInt>(CMD->getValue());
  if (CI == nullptr)
    return false;

  return CI->isZero() && CI->getBitWidth() == 1U;
}

} // end namespace mdutils
