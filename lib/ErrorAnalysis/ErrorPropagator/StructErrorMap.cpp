//===-- StructErrorMap.cpp - Struct Range and Error Map ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A class that keeps track of Struct filed errors and ranges.
///
//===----------------------------------------------------------------------===//

#include "StructErrorMap.h"

#include "TypeUtils.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"
#include <memory>

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

using namespace llvm;
using namespace mdutils;

StructNode::StructNode(const StructInfo *SI, StructType *ST, StructTree *Parent)
    : StructTree(STK_Node, Parent), Fields(), SType(ST)
{
  assert(ST != nullptr);
  Fields.resize(ST->getNumElements());

  LLVM_DEBUG(dbgs() << "{ ");
  for (std::size_t Idx = 0; Idx < SI->size(); ++Idx) {
    const MDInfo *FieldMDI = SI->getField(Idx);
    if (FieldMDI == nullptr) {
      LLVM_DEBUG(dbgs() << "null ,");
      continue;
    }

    if (const StructInfo *FieldSI = dyn_cast<StructInfo>(FieldMDI)) {
      Fields[Idx].reset(new StructNode(FieldSI,
                                       getElementStructType(ST->getElementType(Idx)),
                                       this));
    } else if (const InputInfo *FieldII = dyn_cast<InputInfo>(FieldMDI)) {
      Fields[Idx].reset(new StructError(FieldII, this));
    } else {
      llvm_unreachable("Unhandled MDInfo kind.");
    }
    LLVM_DEBUG(dbgs() << ", ");
  }
  LLVM_DEBUG(dbgs() << "}");
}

StructNode::StructNode(const StructNode &SN)
    : StructTree(SN), Fields(), SType(SN.SType)
{
  this->Fields.reserve(SN.Fields.size());
  for (const std::unique_ptr<StructTree> &STPtr : SN.Fields) {
    std::unique_ptr<StructTree> New;
    if (STPtr != nullptr)
      New.reset(STPtr->clone());
    this->Fields.push_back(std::move(New));
  }
}

StructNode &StructNode::operator=(const StructNode &O)
{
  this->SType = O.SType;
  this->Fields.clear();
  this->Fields.reserve(O.Fields.size());
  for (const std::unique_ptr<StructTree> &STPtr : O.Fields) {
    std::unique_ptr<StructTree> New(STPtr->clone());
    this->Fields.push_back(std::move(New));
  }

  return *this;
}

StructType *StructNode::getElementStructType(Type *T)
{
  while (!T->isStructTy()) {
    if (PointerType *PT = dyn_cast<PointerType>(T))
      T = PT->getPointerElementType();
    else if (isa<ArrayType>(T) || isa<VectorType>(T)) {
      T = T->getContainedType(0);
    } else
      return nullptr;
  }
  return cast<StructType>(T);
}

StructError::StructError(const InputInfo *II, StructTree *Parent)
    : StructTree(STK_Error, Parent), Error()
{
  FPInterval FPI(II);

  LLVM_DEBUG(dbgs() << "{Range: [" << static_cast<double>(FPI.Min) << ", "
                    << static_cast<double>(FPI.Max) << "], Error: ");

  if (FPI.hasInitialError()) {
    AffineForm<inter_t> Err(0.0, FPI.getInitialError());
    Error = std::make_pair(FPI, Err);

    LLVM_DEBUG(dbgs() << FPI.getInitialError() << "} ");
  } else {
    Error = std::make_pair(FPI, std::nullopt);

    LLVM_DEBUG(dbgs() << "none}");
  }
}

Value *StructTreeWalker::retrieveRootPointer(Value *P)
{
  IndexStack.clear();
  return navigatePointerTreeToRoot(P);
}

StructError *StructTreeWalker::getOrCreateFieldNode(StructTree *Root)
{
  return navigateStructTree(Root, true);
}

StructError *StructTreeWalker::getFieldNode(StructTree *Root)
{
  return navigateStructTree(Root, false);
}

StructTree *StructTreeWalker::makeRoot(Value *P)
{
  StructType *ST = cast<StructType>(cast<PointerType>(P->getType())->getPointerElementType());
  return new StructNode(ST);
}

Value *StructTreeWalker::navigatePointerTreeToRoot(Value *P)
{
  assert(P != nullptr);
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(P)) {
    auto IdxIt = GEPI->idx_begin();
    ++IdxIt;
    for (; IdxIt != GEPI->idx_end(); ++IdxIt)
      IndexStack.push_back(parseIndex(*IdxIt));

    return navigatePointerTreeToRoot(GEPI->getPointerOperand());
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(P)) {
    if (isa<GEPOperator>(CE)) {
      auto OpIt = CE->op_begin();

      // Get the Pointer Operand.
      assert(OpIt != CE->op_end());
      Value *PointerOp = *OpIt;
      assert(PointerOp->getType()->isPointerTy());
      ++OpIt;

      // Discard first index.
      assert(OpIt != CE->op_end());
      ++OpIt;

      // Push other indices.
      for (; OpIt != CE->op_end(); ++OpIt)
        IndexStack.push_back(parseIndex(*OpIt));

      return navigatePointerTreeToRoot(PointerOp);
    } else
      return nullptr;
  } else if (LoadInst *LI = dyn_cast<LoadInst>(P)) {
    return navigatePointerTreeToRoot(LI->getPointerOperand());
  } else if (Argument *A = dyn_cast<Argument>(P)) {
    auto AArg = ArgBindings.find(A);
    if (AArg != ArgBindings.end() && AArg->second != nullptr)
      return navigatePointerTreeToRoot(AArg->second);
    else
      return (isa<StructType>(cast<PointerType>(A->getType())->getPointerElementType())) ? P : nullptr;
  } else if (AllocaInst *AI = dyn_cast<AllocaInst>(P)) {
    return (isa<StructType>(AI->getAllocatedType())) ? P : nullptr;
  } else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(P)) {
    return (GV->getValueType()->isStructTy()) ? P : nullptr;
  } else if (ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(P)) {
    for (unsigned Idx : EVI->indices())
      IndexStack.push_back(Idx);

    return EVI->getAggregateOperand();
  } else if (InsertValueInst *IVI = dyn_cast<InsertValueInst>(P)) {
    for (unsigned Idx : IVI->indices())
      IndexStack.push_back(Idx);

    return IVI->getAggregateOperand();
  }
  return nullptr;
}

StructError *StructTreeWalker::navigateStructTree(StructTree *Root, bool Create)
{
  if (isa<StructError>(Root))
    return cast<StructError>(Root);

  StructNode *RN = cast<StructNode>(Root);
  unsigned ChildIdx = IndexStack.back();
  IndexStack.pop_back();

  StructTree *Child = RN->getStructElement(ChildIdx);
  if (Child != nullptr) {
    return navigateStructTree(Child, Create);
  } else if (Create) {
    Type *ChildType = RN->getStructType()->getElementType(ChildIdx);
    while (isa<ArrayType>(ChildType) || isa<VectorType>(ChildType)) {
      // Just discard array indices.
      ChildType = ChildType->getContainedType(0);
      if (IndexStack.size() == 0) {
        LLVM_DEBUG(dbgs() << "WARNING: struct tree shape mismatch.\n");
        return nullptr;
      }
      IndexStack.pop_back();
    }

    if (StructType *ChildST = dyn_cast<StructType>(ChildType)) {
      RN->setStructElement(ChildIdx, new StructNode(ChildST, Root));
      return navigateStructTree(RN->getStructElement(ChildIdx), Create);
    } else {
      RN->setStructElement(ChildIdx, new StructError(Root));
      return cast<StructError>(RN->getStructElement(ChildIdx));
    }
  } else
    return nullptr;
}

unsigned StructTreeWalker::parseIndex(const Use &U) const
{
  if (ConstantInt *CIdx = dyn_cast<ConstantInt>(U.get()))
    return CIdx->getZExtValue();
  else
    return 0U;
}

StructErrorMap::StructErrorMap(const StructErrorMap &M)
    : StructMap(), ArgBindings(M.ArgBindings)
{
  for (auto &KV : M.StructMap) {
    std::unique_ptr<StructTree> New(KV.second->clone());
    this->StructMap.insert(std::make_pair(KV.first, std::move(New)));
  }
}

StructErrorMap &StructErrorMap::operator=(const StructErrorMap &O)
{
  StructErrorMap Tmp(O);
  std::swap(this->StructMap, Tmp.StructMap);
  std::swap(this->ArgBindings, Tmp.ArgBindings);

  return *this;
}

void StructErrorMap::initArgumentBindings(Function &F,
                                          const ArrayRef<Value *> AArgs)
{
  auto AArgIt = AArgs.begin();
  for (Argument &FArg : F.args()) {
    if (AArgIt == AArgs.end())
      break;
    if (FArg.getType()->isPointerTy() && cast<PointerType>(FArg.getType())->getPointerElementType()->isStructTy())
      ArgBindings.insert(std::make_pair(&FArg, *AArgIt));

    ++AArgIt;
  }
}

void StructErrorMap::setFieldError(Value *P, const StructTree::RangeError &Err)
{
  StructTreeWalker STW(ArgBindings);
  Value *RootP = STW.retrieveRootPointer(P);
  if (RootP == nullptr)
    return;

  auto RootIt = StructMap.find(RootP);
  if (RootIt == StructMap.end()) {
    RootIt = StructMap.insert(std::make_pair(RootP, std::unique_ptr<StructTree>(STW.makeRoot(RootP)))).first;
  }

  StructError *FE = STW.getOrCreateFieldNode(RootIt->second.get());
  if (FE) {
    FE->setError(Err);
  } else {
    LLVM_DEBUG(dbgs() << "WARNING: could not retrieve struct field error.\n");
  }
}

const StructTree::RangeError *StructErrorMap::getFieldError(Value *P) const
{
  StructTreeWalker STW(ArgBindings);
  Value *RootP = STW.retrieveRootPointer(P);
  if (RootP == nullptr)
    return nullptr;

  auto RootIt = StructMap.find(RootP);
  if (RootIt == StructMap.end())
    return nullptr;

  StructError *FE = STW.getFieldNode(RootIt->second.get());
  if (FE)
    return &FE->getError();
  else
    return nullptr;
}

void StructErrorMap::updateStructTree(const StructErrorMap &O, const ArrayRef<Value *> Pointers)
{
  StructTreeWalker STW(this->ArgBindings);
  for (Value *P : Pointers) {
    if (P == nullptr || !P->getType()->isPointerTy())
      continue;

    if (Value *Root = STW.retrieveRootPointer(P)) {
      auto OTreeIt = O.StructMap.find(Root);
      if (OTreeIt != O.StructMap.end() && OTreeIt->second != nullptr)
        this->StructMap[Root].reset(OTreeIt->second->clone());
    }
  }
}

void StructErrorMap::createStructTreeFromMetadata(Value *V,
                                                  const mdutils::MDInfo *MDI)
{
  LLVM_DEBUG(dbgs() << "[taffo-err] Retrieving data for struct [" << *V << "]: ");

  StructType *ST = nullptr;
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V))
    ST = cast<StructType>(taffo::fullyUnwrapPointerOrArrayType(GV->getValueType()));
  else
    ST = cast<StructType>(taffo::fullyUnwrapPointerOrArrayType(V->getType()));

  StructNode *RootSN = new StructNode(cast<StructInfo>(MDI), ST);
  // Erase previous data
  StructMap[V].reset(RootSN);

  LLVM_DEBUG(dbgs() << ".\n");
}

} // end namespace ErrorProp
