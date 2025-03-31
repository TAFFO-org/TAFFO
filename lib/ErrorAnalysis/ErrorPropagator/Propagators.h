//===-- Propagators.h - Error Propagators for LLVM Instructions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declarations of functions that propagate fixed point computation errors
/// for each LLVM Instruction.
///
//===----------------------------------------------------------------------===//

#ifndef ERRORPROPAGATOR_PROPAGATORS_H
#define ERRORPROPAGATOR_PROPAGATORS_H

#include "RangeErrorMap.h"
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/IR/Instruction.h>

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

class InstructionPropagator
{
public:
  InstructionPropagator(RangeErrorMap &RMap, llvm::MemorySSA &MemSSA, bool SloppyAA)
      : RMap(RMap), MemSSA(MemSSA), SloppyAA(SloppyAA) {}

  /// Propagate errors for a Binary Operator instruction.
  bool propagateBinaryOp(llvm::Instruction &);

  /// Propagate errors for a store instruction
  /// by associating the errors of the source to the destination.
  bool propagateStore(llvm::Instruction &);

  /// Propagate the errors for a Load instruction
  /// by associating the errors of the source to it.
  bool propagateLoad(llvm::Instruction &);

  /// Propagate the errors for an Extend instruction
  /// by associating the errors of the source to it.
  bool propagateExt(llvm::Instruction &);

  /// Propagate the errors for a Trunc instruction
  /// by associating the errors of the source to it.
  bool propagateTrunc(llvm::Instruction &);

  /// Propagate the errors for a FNeg instruction
  /// by associating the errors of the source to it.
  bool propagateFNeg(llvm::Instruction &I);

  /// Propagate errors for a SIToFP or UIToFP instruction
  /// by associating the errors of the source to it.
  bool propagateIToFP(llvm::Instruction &);

  /// Propagate errors for a FPToSI or FPToUI instruction
  /// by associating to it the error of the source plus the rounding error.
  bool propagateFPToI(llvm::Instruction &);

  /// Propagate errors for a Select instruction
  /// by associating the maximum error from the source values to it.
  bool propagateSelect(llvm::Instruction &);

  /// Propagate errors for a PHI Node
  /// by associating the maximum error from the source values to it.
  bool propagatePhi(llvm::Instruction &);

  /// Check whether the error on the operands could make this comparison wrong.
  bool checkCmp(CmpErrorMap &, llvm::Instruction &);

  /// Associate the error previously computed for the returned value
  /// to the containing function, only if larger
  /// than the one already associated (if any).
  bool propagateRet(llvm::Instruction &I);

  static bool isSpecialFunction(llvm::Function &F);

  /// Associate the error of the called function to I.
  /// Works woth both CallInst and InvokeInst.
  bool propagateCall(llvm::Instruction &I);

  /// Associate the error of the source pointer to I.
  bool propagateGetElementPtr(llvm::Instruction &I);

  bool propagateExtractValue(llvm::Instruction &I);
  bool propagateInsertValue(llvm::Instruction &I);

private:
  RangeErrorMap &RMap;
  llvm::MemorySSA &MemSSA;
  bool SloppyAA;

  const RangeErrorMap::RangeError *getConstantFPRangeError(llvm::ConstantFP *VFP);

  const RangeErrorMap::RangeError *
  getConstantRangeError(llvm::Instruction &I, llvm::ConstantInt *VInt,
                        bool DoublePP = false,
                        const mdutils::FPType *FallbackTy = nullptr);

  const RangeErrorMap::RangeError *
  getOperandRangeError(llvm::Instruction &I, llvm::Value *V,
                       bool DoublePP = false,
                       const mdutils::FPType *FallbackTy = nullptr);

  const RangeErrorMap::RangeError *
  getOperandRangeError(llvm::Instruction &I, unsigned Op,
                       bool DoublePP = false,
                       const mdutils::FPType *FallbackTy = nullptr);

  void updateArgumentRE(llvm::Value *Pointer, const RangeErrorMap::RangeError *NewRE);

  bool unOpErrorPassThrough(llvm::Instruction &I);

  static bool isSqrt(llvm::Function &F);
  static bool isLog(llvm::Function &F);
  static bool isExp(llvm::Function &F);
  static bool isAcos(llvm::Function &F);
  static bool isAsin(llvm::Function &F);
  bool propagateSqrt(llvm::Instruction &I);
  bool propagateLog(llvm::Instruction &I);
  bool propagateExp(llvm::Instruction &I);
  bool propagateAcos(llvm::Instruction &I);
  bool propagateAsin(llvm::Instruction &I);
  bool propagateSpecialCall(llvm::Instruction &I, llvm::Function &Called);

  inter_t computeMinRangeDiff(const FPInterval &R1, const FPInterval &R2);

public:
  static void logInstruction(const llvm::Value &I);
  static void logInfo(const llvm::StringRef Msg);
  static void logInfoln(const llvm::StringRef Msg);
  static void logError(const AffineForm<inter_t> &Err);
  static void logError(const RangeErrorMap::RangeError &RE);
  static void logErrorln(const AffineForm<inter_t> &Err);
  static void logErrorln(const RangeErrorMap::RangeError &RE);
};

} // end of namespace ErrorProp

#undef DEBUG_TYPE // "errorprop"

#endif
