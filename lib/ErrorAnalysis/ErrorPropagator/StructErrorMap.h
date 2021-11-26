//===-- StructErrorMap.h - Struct Range and Error Map -----------*- C++ -*-===//
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

#ifndef ERRORPROPAGATOR_STRUCTERRORMAP_H
#define ERRORPROPAGATOR_STRUCTERRORMAP_H

#include <map>
#include "llvm/Support/Casting.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Argument.h"
#include "FixedPoint.h"
#include "Metadata.h"

namespace ErrorProp {

class StructTree {
public:
  typedef std::pair<FPInterval, llvm::Optional<AffineForm<inter_t> > > RangeError;
  enum StructTreeKind {
    STK_Node,
    STK_Error
  };

  StructTree(StructTreeKind K, StructTree *P = nullptr)
    : Parent(P), Kind(K) {}

  virtual StructTree *clone() const = 0;
  StructTree *getParent() const { return Parent; }
  void setParent(StructTree *P) { Parent = P; }
  virtual ~StructTree() = default;

  StructTreeKind getKind() const { return Kind; }
protected:
  StructTree *Parent;

private:
  StructTreeKind Kind;
};

class StructNode : public StructTree {
public:
  StructNode(llvm::StructType *ST, StructTree *Parent = nullptr)
    : StructTree(STK_Node, Parent), Fields(), SType(ST) {
    assert(ST != nullptr);
    Fields.resize(ST->getNumElements());
  }

  StructNode(const mdutils::StructInfo *MDI, llvm::StructType *ST, StructTree *Parent = nullptr);
  StructNode(const StructNode &SN);
  StructNode &operator=(const StructNode &O);

  StructTree *clone() const override { return new StructNode(*this); }
  llvm::StructType *getStructType() const { return SType; }
  StructTree *getStructElement(unsigned I) {
    return (I < Fields.size()) ? Fields[I].get() : nullptr;
  }

  void setStructElement(unsigned I, StructTree *NewEl) { Fields[I].reset(NewEl); }

  static bool classof(const StructTree *ST) { return ST->getKind() == STK_Node; }
private:
  llvm::SmallVector<std::unique_ptr<StructTree>, 2U> Fields;
  llvm::StructType *SType;

  static llvm::StructType *getElementStructType(llvm::Type *T);
};

class StructError : public StructTree {
public:
  StructError(StructTree *Parent = nullptr)
    : StructTree(STK_Error, Parent), Error() {}

  StructError(const RangeError &Err, StructTree *Parent = nullptr)
    : StructTree(STK_Error, Parent), Error(Err) {}

  StructError(const mdutils::InputInfo *MDI,  StructTree *Parent = nullptr);

  StructTree *clone() const override { return new StructError(*this); }
  const RangeError& getError() const { return Error; }
  void setError (const RangeError &Err) { Error = Err; }

  static bool classof(const StructTree *ST) { return ST->getKind() == STK_Error; }
private:
  RangeError Error;
};

class StructTreeWalker {
public:
  StructTreeWalker(const llvm::DenseMap<llvm::Argument *, llvm::Value *> &ArgBindings)
    : IndexStack(), ArgBindings(ArgBindings) {}

  llvm::Value *retrieveRootPointer(llvm::Value *P);
  StructError *getOrCreateFieldNode(StructTree *Root);
  StructError *getFieldNode(StructTree *Root);
  StructTree *makeRoot(llvm::Value *P);

protected:
  llvm::SmallVector<unsigned, 4U> IndexStack;
  const llvm::DenseMap<llvm::Argument *, llvm::Value *> &ArgBindings;

  llvm::Value *navigatePointerTreeToRoot(llvm::Value *P);
  StructError *navigateStructTree(StructTree *Root, bool Create = false);
  unsigned parseIndex(const llvm::Use &U) const;
};

class StructErrorMap {
public:
  StructErrorMap() = default;
  StructErrorMap(const StructErrorMap &M);
  StructErrorMap &operator=(const StructErrorMap &O);

  void initArgumentBindings(llvm::Function &F, const llvm::ArrayRef<llvm::Value *> AArgs);
  void setFieldError(llvm::Value *P, const StructTree::RangeError &Err);
  const StructTree::RangeError *getFieldError(llvm::Value *P) const;
  void updateStructTree(const StructErrorMap &O, const llvm::ArrayRef<llvm::Value *> Pointers);
  void createStructTreeFromMetadata(llvm::Value *V,
				    const mdutils::MDInfo *MDI);

protected:
  std::map<llvm::Value *, std::unique_ptr<StructTree> > StructMap;
  llvm::DenseMap<llvm::Argument *, llvm::Value *> ArgBindings;
};

} // end namespace ErrorProp

#endif
