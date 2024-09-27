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
#include "TypeUtils.h"
#include "llvm/IR/Constants.h"
#include <cmath>
#include <iostream>

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

  Metadata *TypeMD;
  if (Type == Type::getVoidTy(C))
    TypeMD = MDString::get(C, "void");
  else
    TypeMD = ConstantAsMetadata::get(Constant::getNullValue(Type));

  Metadata *SizeInBitsMD = ConstantAsMetadata::get(ConstantInt::get(C, APInt(32, SizeInBits)));
  Metadata *ITypeMD = (IType) ? IType->toMetadata(C) : Null;
  Metadata *IRangeMD = (IRange) ? IRange->toMetadata(C) : Null;
  Metadata *IErrorMD = (IError) ? InitialErrorToMetadata(C, *IError) : Null;
  Metadata *BufferIDMD = MDString::get(C, BufferID);

  uint64_t Flags = IEnableConversion | (IFinal << 1);
  Metadata *FlagsMD = ConstantAsMetadata::get(ConstantInt::get(Type::getIntNTy(C, 2U), Flags));

  Metadata *InputMDs[] = {TypeMD, SizeInBitsMD, ITypeMD, IRangeMD, IErrorMD, FlagsMD, BufferIDMD};
  return MDNode::get(C, InputMDs);
}

bool InputInfo::isInputInfoMetadata(Metadata *MD)
{
  MDNode *MDN = dyn_cast<MDNode>(MD);
  if (MDN == nullptr || MDN->getNumOperands() < 5U || MDN->getNumOperands() > 7U)
    return false;

  Metadata *Op0 = MDN->getOperand(0U).get();
  if (!(isa<ConstantAsMetadata>(Op0) || isa<MDString>(Op0)))
    return false;

  Metadata *Op1 = MDN->getOperand(1U).get();
  if (!isa<ConstantAsMetadata>(Op1))
    return false;

  Metadata *Op2 = MDN->getOperand(2U).get();
  if (!(IsNullInputInfoField(Op2) || TType::isTTypeMetadata(Op2)))
    return false;

  Metadata *Op3 = MDN->getOperand(3U).get();
  if (!(IsNullInputInfoField(Op3) || Range::isRangeMetadata(Op3)))
    return false;

  Metadata *Op4 = MDN->getOperand(4U).get();
  if (!(IsNullInputInfoField(Op4) || IsInitialErrorMetadata(Op4)))
    return false;

  if (MDN->getNumOperands() == 6U) {
    auto Op5 = MDN->getOperand(5U).get();
    if (!(IsNullInputInfoField(Op5) || isa<ConstantAsMetadata>(Op5) || isa<MDString>(Op5)))
      return false;
  }
  else if (MDN->getNumOperands() == 7U) {
    auto Op5 = MDN->getOperand(5U).get();
    if (!(IsNullInputInfoField(Op5) || isa<ConstantAsMetadata>(Op5)))
      return false;
    auto Op6 = MDN->getOperand(6U).get();
    if (!(IsNullInputInfoField(Op6) || isa<MDString>(Op6)))
      return false;
  }

  return true;
}

MDNode *StructInfo::toMetadata(LLVMContext &C) const
{
  Metadata *Null = ConstantAsMetadata::get(ConstantInt::getFalse(C));
  SmallVector<Metadata*, 4U> StructMDs;
  StructMDs.reserve(4 + Fields.size() + FieldsLayout.size());

  StructMDs.push_back(ConstantAsMetadata::get(Constant::getNullValue(Type)));
  StructMDs.push_back(ConstantAsMetadata::get(ConstantInt::get(C, APInt(32, SizeInBits))));
  StructMDs.push_back(ConstantAsMetadata::get(ConstantInt::get(C, APInt(32, StructSizeInBits))));
  StructMDs.push_back(MDString::get(C, getBufferIDsString()));

  for (unsigned int i = 0; i < Fields.size(); i++) {
    auto MDI = Fields[i];
    StructMDs.push_back(MDI ? MDI->toMetadata(C) : Null);
    StructMDs.push_back(MDString::get(C, FieldsLayout[i].toString()));
  }
  return MDNode::get(C, StructMDs);
}

void StructInfo::setType(llvm::Type *T, const DataLayout &DL) {
  MDInfo::setType(T, DL);
  auto ST = getStructType();
  assert(size() == ST->getNumElements());
  if (ST->isOpaque()) {
    LLVM_DEBUG(dbgs() << "Cannot infer struct layout as the struct type is opaque (" << *ST << ")\n");
    return;
  }
  StructSizeInBits = DL.getStructLayout(ST)->getSizeInBits();
  setFieldsLayout(ST, DL);
}

void StructInfo::setFieldsLayout(StructType *ST, const DataLayout &DL) {
  FieldsLayout.clear();
  auto StructLayout = DL.getStructLayout(ST);
  for (unsigned int i = 0; i < Fields.size(); i++) {

    llvm::Type *FieldType = ST->getElementType(i);
    unsigned int FieldSize;
    if (auto FieldST = dyn_cast<llvm::StructType>(FieldType))
      FieldSize = DL.getStructLayout(FieldST)->getSizeInBits();
    else
      FieldSize = DL.getTypeSizeInBits(FieldType);

    unsigned int start = StructLayout->getElementOffsetInBits(i);
    FieldBits Bits(start, start + FieldSize);
    FieldsLayout.push_back(Bits);
  }
}

void StructInfo::flatten(StructInfo::FieldsType &flatFields, StructInfo::FieldsLayoutType &flatFieldsLayout, unsigned int firstBit) const {
  for (unsigned int i = 0; i < Fields.size(); i++) {
    auto Field = Fields[i];
    auto FieldBits = FieldsLayout[i];
    if (!FieldBits.isUndefined())
      FieldBits += firstBit;

    if (Field.get() && getStructType()->getElementType(i)->isStructTy()) {
      auto StructField = dyn_cast<StructInfo>(Field.get());
      StructField->flatten(flatFields, flatFieldsLayout, FieldBits.start);
    }
    else {
      flatFields.push_back(Field);
      flatFieldsLayout.push_back(FieldBits);
    }
  }
}

void StructInfo::setFlatField(unsigned int flatIndex, const std::shared_ptr<MDInfo> &field, unsigned int *flatCurr, bool *found) {
  unsigned int initFlatCurr = 0;
  bool initFound = false;
  if (flatCurr == nullptr)
    flatCurr = &initFlatCurr;
  if (found == nullptr)
    found = &initFound;

  for (unsigned int i = 0; i < Fields.size() && !(*found); i++) {
    auto Field = Fields[i];
    if (Field.get() && getStructType()->getElementType(i)->isStructTy()) {
      auto StructField = dyn_cast<StructInfo>(Field.get());
      StructField->setFlatField(flatIndex, field, flatCurr, found);
    }
    else if (*flatCurr == flatIndex) {
      setField(i, field);
      *found = true;
      return;
    }
    else (*flatCurr)++;
  }
}

std::shared_ptr<StructInfo> StructInfo::constructFromLLVMType(llvm::Type *t, const DataLayout &DL,
                                                              SmallDenseMap<llvm::Type *, std::shared_ptr<StructInfo>> *recursionMap)
{
  t = taffo::fullyUnwrapPointerOrArrayType(t);

  std::unique_ptr<SmallDenseMap<llvm::Type*, std::shared_ptr<StructInfo>>> _recursionMap;
  if (!recursionMap) {
    _recursionMap.reset(new SmallDenseMap<llvm::Type*, std::shared_ptr<StructInfo>>());
    recursionMap = _recursionMap.get();
  }

  auto rec = recursionMap->find(t);
  if (rec != recursionMap->end()) {
    //return rec->getSecond();    recursion mess things up
    return nullptr;
  }

  if (t->isStructTy()) {
    auto st = cast<StructType>(t);
    std::shared_ptr<StructInfo> res = std::make_shared<StructInfo>(StructInfo(st, DL));
    recursionMap->insert({t, res});
    for (unsigned int i = 0; i < st->getNumElements(); i++)
      res->setField(i, StructInfo::constructFromLLVMType(st->getElementType(i), DL, recursionMap));
    return res;
  }

  recursionMap->insert({t, nullptr});
  return nullptr;
}

void StructInfo::copyCommon(const std::shared_ptr<MDInfo> &src, std::shared_ptr<MDInfo> &dst, bool copyBufferID, bool clone) {
  FieldsType flatSrcFields, flatDstFields;
  FieldsLayoutType flatSrcFieldsLayout, flatDstFieldsLayout;

  auto structSrc = dyn_cast<StructInfo>(src.get());
  if (structSrc && !structSrc->getStructType()->isOpaque())
    structSrc->flatten(flatSrcFields, flatSrcFieldsLayout);
  else {
    flatSrcFields.push_back(src);
    flatSrcFieldsLayout.push_back(FieldBits(0, src->getSizeInBits()));
  }

  auto structDst = dyn_cast<StructInfo>(dst.get());
  if (structDst && !structDst->getStructType()->isOpaque())
    structDst->flatten(flatDstFields, flatDstFieldsLayout);
  else {
    flatDstFields.push_back(dst);
    flatDstFieldsLayout.push_back(FieldBits(0, dst->getSizeInBits()));
  }

  LLVM_DEBUG(
  dbgs() << "copycommon:\n\n";

  dbgs() << *src->getType() << "\n";
  if (structSrc) {
    for (auto fieldType : structSrc->getStructType()->elements()) dbgs() << *fieldType << ", ";
    dbgs() << "\n";
    for (auto fieldBits : flatSrcFieldsLayout) dbgs() << fieldBits << ", ";
    dbgs() << "\n";
  }
  dbgs() << "\n";

  dbgs() << *dst->getType() << "\n";
  if (structDst) {
    for (auto fieldType : structDst->getStructType()->elements()) dbgs() << *fieldType << ", ";
    dbgs() << "\n";
    for (auto fieldBits : flatDstFieldsLayout) dbgs() << fieldBits << ", ";
    dbgs() << "\n";
  }
  dbgs() << "\n";
  );

  unsigned int currDst = 0;
  std::shared_ptr<MDInfo> dstField = flatDstFields[currDst];
  FieldBits dstBits = flatDstFieldsLayout[currDst];
  for (unsigned int currSrc = 0; currSrc < flatSrcFields.size(); currSrc++) {
    std::shared_ptr<MDInfo> srcField = flatSrcFields[currSrc];
    FieldBits srcBits = flatSrcFieldsLayout[currSrc];
    while (dstBits.end < srcBits.end && currDst < flatDstFields.size()) {
      dstField = flatDstFields[currDst];
      dstBits = flatDstFieldsLayout[currDst];
      currDst++;
    }
    LLVM_DEBUG(dbgs() << srcBits << " - " << dstBits);

    if (!srcField.get() || (dstBits != srcBits && (!dstField.get() || dstField->getType() != Type::getFloatTy(dstField->getType()->getContext())))) {
      LLVM_DEBUG(dbgs() << "\n");
      continue;
    }
    else
      LLVM_DEBUG(dbgs() << " -> copy" << "\n");

    auto structSrcField = dyn_cast<StructInfo>(srcField.get());
    if (structSrcField && !structSrcField->getStructType()->isOpaque())
      copyCommon(srcField, dstField, copyBufferID, clone);
    else {
      auto copiedField = clone ? std::shared_ptr<MDInfo>(srcField->clone(copyBufferID)) : srcField;
      if (structDst)
        structDst->setFlatField(currDst-1, copiedField);
      else
        dst = copiedField;
    }

    LLVM_DEBUG(dbgs() << "Result: " << dst->toString() << "\n");
  }
}

void StructInfo::cloneCommon(const std::shared_ptr<MDInfo> &src, std::shared_ptr<MDInfo> &dst, bool copyBufferID) {
  copyCommon(src, dst, copyBufferID, true);
}

void StructInfo::copyCommon(const std::shared_ptr<MDInfo> &src, std::shared_ptr<MDInfo> &dst) {
    copyCommon(src, dst, true, false);
}

bool StructInfo::reduce() {
  bool keep = false;
  for (auto &Field : Fields) {
    if (Field.get()) {
      bool keepField;
      if (auto StructField = dyn_cast<StructInfo>(Field.get()))
        keepField = StructField->reduce();
      else
        keepField = true;

      if (!keepField)
        Field.reset();
      keep |= keepField;
    }
  }
  return keep;
}

llvm::StructType *StructInfo::getStructType() const {
  auto UT = taffo::fullyUnwrapPointerOrArrayType(Type);
  assert(UT->isStructTy());
  return llvm::cast<llvm::StructType>(UT);
}

std::string StructInfo::getBufferIDsString() const {
  std::stringstream ss;
  unsigned int numIDs = BufferIDs.size();
  unsigned int i = 0;
  for (const auto &id : BufferIDs) {
    ss << id;
    if (i < numIDs - 1)
      ss << ", ";
    i++;
  }
  return ss.str();
}

void StructInfo::setBufferIDs(const std::string &IDs) {
  std::string id;
  std::stringstream ss(IDs);
  while (std::getline(ss, id, ',')) {
    size_t start = id.find_first_not_of(' ');
    size_t end = id.find_last_not_of(' ');
    if (start != std::string::npos && end != std::string::npos)
      BufferIDs.insert(id.substr(start, end - start + 1));
  }
}

std::set<TType*> StructInfo::getBufferIDTypes(const std::string &BufferID) const {
  std::set<TType*> Types;
  if (BufferIDs.find(BufferID) == BufferIDs.end())
    return Types;
  for (const auto &Field : Fields) {
    if (auto SIField = dyn_cast_or_null<StructInfo>(Field.get())) {
      auto FieldTypes = SIField->getBufferIDTypes(BufferID);
      Types.insert(FieldTypes.begin(), FieldTypes.end());
    }
    else if (auto IIField = dyn_cast_or_null<InputInfo>(Field.get()))
      Types.insert(IIField->IType.get());
  }
  return Types;
}

void StructInfo::setBufferIDType(const std::string &BufferID, const std::shared_ptr<TType> &Type) {
  if (BufferIDs.find(BufferID) == BufferIDs.end())
    return;
  for (const auto &Field : Fields) {
    if (auto SIField = dyn_cast_or_null<StructInfo>(Field.get()))
      SIField->setBufferIDType(BufferID, Type);
    else if (auto IIField = dyn_cast_or_null<InputInfo>(Field.get()))
      if (IIField->getBufferID() == BufferID)
        IIField->IType.reset(Type->clone());
  }
}

std::unique_ptr<CmpErrorInfo> CmpErrorInfo::createFromMetadata(MDNode *MDN) {
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
