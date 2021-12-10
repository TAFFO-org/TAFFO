//===-- RangeErrorMap.h - Range and Error Maps ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains classes that map Instructions and other Values
/// to the corresponding computed ranges and errors.
///
//===----------------------------------------------------------------------===//

#ifndef ERRORPROPAGATOR_RANGEERRORMAP_H
#define ERRORPROPAGATOR_RANGEERRORMAP_H

#include "AffineForms.h"
#include "FixedPoint.h"
#include "Metadata.h"
#include "StructErrorMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include <map>

namespace ErrorProp
{

class TargetErrors
{
public:
  void updateTarget(const llvm::Value *V, const inter_t &Error);
  void updateTarget(const llvm::Instruction *I, const inter_t &Error);
  void updateTarget(const llvm::GlobalVariable *V, const inter_t &Error);
  void updateTarget(llvm::StringRef T, const inter_t &Error);
  void updateAllTargets(const TargetErrors &Other);

  inter_t getErrorForTarget(llvm::StringRef T) const;

  void printTargetErrors(llvm::raw_ostream &OS) const;

protected:
  llvm::DenseMap<llvm::StringRef, inter_t> Targets;
};

class RangeErrorMap
{
public:
  typedef std::pair<FPInterval, llvm::Optional<AffineForm<inter_t>>> RangeError;

  RangeErrorMap(mdutils::MetadataManager &MDManager, bool Absolute = true, bool ExactConst = false)
      : REMap(), MDMgr(&MDManager), SEMap(), TErrs(),
        OutputAbsolute(Absolute), ExactConst(ExactConst) {}

  const FPInterval *getRange(const llvm::Value *) const;

  const AffineForm<inter_t> *getError(const llvm::Value *) const;

  const RangeError *
  getRangeError(const llvm::Value *) const;

  /// Set error for Value V.
  /// Err cannot be a reference to an error contained in this map.
  void setError(const llvm::Value *V, const AffineForm<inter_t> &Err);

  /// Set range and error for Value V.
  /// RE cannot be a reference to a RangeError contained in this map.
  void setRangeError(const llvm::Value *V, const RangeError &RE);

  void erase(const llvm::Value *V)
  {
    REMap.erase(V);
  }

  /// Retrieve range for instruction I from metadata.
  /// Return true if initial error metadata was found attached to I.
  bool retrieveRangeError(llvm::Instruction &I);

  /// Retrieve ranges and errors for arguments of function F from metadata.
  void retrieveRangeErrors(llvm::Function &F);

  /// Associate the errors of the actual parameters of F contained in Args
  /// to the corresponding formal parameters.
  void applyArgumentErrors(llvm::Function &F,
                           llvm::SmallVectorImpl<llvm::Value *> *Args);

  /// Retrieve range and error for global variable V, and add it to the map.
  void retrieveRangeError(llvm::GlobalObject &V);

  mdutils::MetadataManager &getMetadataManager() { return *MDMgr; }

  const RangeError *getStructRangeError(llvm::Value *V) const;
  void setStructRangeError(llvm::Value *V, const RangeError &RE);

  void initArgumentBindings(llvm::Function &F,
                            const llvm::ArrayRef<llvm::Value *> AArgs)
  {
    SEMap.initArgumentBindings(F, AArgs);
  }
  void updateStructErrors(const RangeErrorMap &O,
                          const llvm::ArrayRef<llvm::Value *> Pointers)
  {
    SEMap.updateStructTree(O.SEMap, Pointers);
  }

  void updateTargets(const RangeErrorMap &Other);
  void printTargetErrors(llvm::raw_ostream &OS) const { TErrs.printTargetErrors(OS); }

  double getOutputError(const llvm::Value *V) const;
  double getOutputError(const RangeError &RE) const;

  bool isExactConst() const { return ExactConst; }

protected:
  std::map<const llvm::Value *, RangeError> REMap;
  mdutils::MetadataManager *MDMgr;
  StructErrorMap SEMap;
  TargetErrors TErrs;
  bool OutputAbsolute;
  bool ExactConst;

  void retrieveConstRanges(const llvm::Instruction &I);
  static double computeRelativeError(const RangeError &RE);
}; // end class RangeErrorMap

typedef llvm::DenseMap<llvm::Value *, mdutils::CmpErrorInfo> CmpErrorMap;
#define CMPERRORMAP_NUMINITBUCKETS 4U

} // end namespace ErrorProp

#endif
