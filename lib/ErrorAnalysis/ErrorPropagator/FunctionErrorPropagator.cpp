//===-- FunctionErrorPropagator.cpp - Error Propagator ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Error propagator for fixed point computations in a single function.
///
//===----------------------------------------------------------------------===//

#include "FunctionErrorPropagator.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"

#include "Propagators.h"
#include "MemSSAUtils.h"
#include "Metadata.h"
#include "TypeUtils.h"

namespace ErrorProp {

using namespace llvm;
using namespace mdutils;

#define DEBUG_TYPE "errorprop"

void
FunctionErrorPropagator::computeErrorsWithCopy(RangeErrorMap &GlobRMap,
					       SmallVectorImpl<Value *> *Args,
					       bool GenMetadata) {
  if (F.empty() || FCopy == nullptr) {
    LLVM_DEBUG(dbgs() << "[taffo-err] Function " << F.getName() << " could not be processed.\n");
    return;
  }

  // Increase count of consecutive recursive calls.
  unsigned OldRecCount = FCMap.incRecursionCount(&F);

  Function &CF = *FCopy;

  LLVM_DEBUG(dbgs() << "\n[taffo-err] *** Processing function " << CF.getName()
	<< " (iteration " << OldRecCount + 1 << ")... ***\n");

  CmpMap.clear();
  RMap = GlobRMap;
  // Reset the error associated to this function.
  RMap.erase(FCopy);

  // CFLSteensAAWrapperPass *CFLSAA =
  //   EPPass.getAnalysisIfAvailable<CFLSteensAAWrapperPass>();
  // if (CFLSAA != nullptr)
  //   CFLSAA->getResult().scan(FCopy);

  MemSSA = &(EPPass.getAnalysis<MemorySSAWrapperPass>(CF).getMSSA());

  computeFunctionErrors(Args);

  if (GenMetadata) {
    // Put error metadata in original function.
    attachErrorMetadata();
  }

  applyActualParametersErrors(GlobRMap, Args);

  // Associate computed errors to global variables.
  for (const GlobalVariable &GV : F.getParent()->globals()) {
    const AffineForm<inter_t> *GVErr = RMap.getError(&GV);
    if (GVErr == nullptr)
      continue;
    GlobRMap.setError(&GV, *GVErr);
  }

  // Update target errors
  GlobRMap.updateTargets(RMap);

  // Associate computed error to the original function.
  auto FErr = RMap.getError(FCopy);
  if (FErr != nullptr)
    GlobRMap.setError(&F, AffineForm<inter_t>(*FErr));

  // Restore original recursion count.
  FCMap.setRecursionCount(&F, OldRecCount);

  LLVM_DEBUG(dbgs() << "[taffo-err] Finished processing function " << CF.getName() << ".\n\n");
}

void
FunctionErrorPropagator::computeFunctionErrors(SmallVectorImpl<Value *> *ArgErrs) {
  assert(FCopy != nullptr);

  if (ArgErrs)
    RMap.initArgumentBindings(*FCopy, *ArgErrs);

  RMap.retrieveRangeErrors(*FCopy);
  RMap.applyArgumentErrors(*FCopy, ArgErrs);

  LoopInfo &LInfo =
    EPPass.getAnalysis<LoopInfoWrapperPass>(*FCopy).getLoopInfo();

  // Compute errors for all instructions in the function
  BBScheduler BBSched(*FCopy, LInfo);

  // Restore MemSSA
  assert(FCopy != nullptr);
  MemSSA = &(EPPass.getAnalysis<MemorySSAWrapperPass>(*FCopy).getMSSA());

  for (BasicBlock *BB : BBSched)
    for (Instruction &I : *BB)
      computeInstructionErrors(I);
}

void
FunctionErrorPropagator::computeInstructionErrors(Instruction &I) {
  bool HasInitialError = RMap.retrieveRangeError(I);

  double InitialError;
  if (HasInitialError) {
    auto *IEP = RMap.getError(&I);
    assert(IEP != nullptr);
    InitialError = IEP->noiseTermsAbsSum();
  }

  bool ComputedError = dispatchInstruction(I);

  // if (HasInitialError) {
  //   if (ComputedError) {
  //     LLVM_DEBUG(dbgs() << "WARNING: computed error for instruction "
  // 	    << I.getName() << " ignored because of metadata error "
  // 	    << InitialError << ".\n");
  //     RMap.setError(&I, AffineForm<inter_t>(0.0, InitialError));
  //   }
  //   else {
  //     LLVM_DEBUG(dbgs() << "Initial error for instruction "
  // 	    << I.getName() << ": " << InitialError << ".\n");
  //   }
  // }

  if (!ComputedError && HasInitialError) {
    LLVM_DEBUG(dbgs() << "[taffo-err] WARNING: metadata error "
	  << InitialError << " attached to instruction ("
	  << I << ").\n");
    RMap.setError(&I, AffineForm<inter_t>(0.0, InitialError));
  }

  LLVM_DEBUG(
	if(checkOverflow(I))
	  dbgs() << "[taffo-err] Possible overflow detected for instruction ("
		 << I << ").\n";
	);
}

bool
FunctionErrorPropagator::dispatchInstruction(Instruction &I) {
  assert(MemSSA != nullptr);

  InstructionPropagator IP(RMap, *MemSSA, SloppyAA);

  if (I.isBinaryOp())
    return IP.propagateBinaryOp(I);

  switch (I.getOpcode()) {
    case Instruction::Store:
      return IP.propagateStore(I);
    case Instruction::Load:
      return IP.propagateLoad(I);
    case Instruction::FPExt:
      // Fall-through.
    case Instruction::SExt:
      // Fall-through.
    case Instruction::ZExt:
      return IP.propagateExt(I);
    case Instruction::FPTrunc:
      // Fall-through.
    case Instruction::Trunc:
      return IP.propagateTrunc(I);
    case Instruction::FNeg:
      return IP.propagateFNeg(I);
    case Instruction::Select:
      return IP.propagateSelect(I);
    case Instruction::PHI:
      return IP.propagatePhi(I);
    case Instruction::FCmp:
      // Fall-through.
    case Instruction::ICmp:
      return IP.checkCmp(CmpMap, I);
    case Instruction::Ret:
      return IP.propagateRet(I);
    case Instruction::Call:
     // Fall-through.
    case Instruction::Invoke:
      prepareErrorsForCall(I);
      return IP.propagateCall(I);
    case Instruction::UIToFP:
      // Fall-through.
    case Instruction::SIToFP:
      return IP.propagateIToFP(I);
    case Instruction::FPToUI:
      // Fall-through.
    case Instruction::FPToSI:
      return IP.propagateFPToI(I);
    default:
      LLVM_DEBUG(InstructionPropagator::logInstruction(I);
		 InstructionPropagator::logInfoln("unhandled."));
      return false;
  }
  llvm_unreachable("No return statement.");
}

void
FunctionErrorPropagator::prepareErrorsForCall(Instruction &I) {
  CallSite CS(&I);
  Function *CalledF = CS.getCalledFunction();
  SmallVector<Value *, 0U> Args;
  for (Use &U : CS.args()) {
    Value *Arg = U.get();
    if (Arg->getType()->isPointerTy()
	&& !taffo::fullyUnwrapPointerOrArrayType(Arg->getType())->isStructTy()) {
      auto RE = RMap.getRangeError(Arg);
      if (RE != nullptr && RE->second.hasValue())
	Args.push_back(Arg);
      else {
	Value *OrigPointer = MemSSAUtils::getOriginPointer(*MemSSA, Arg);
	Args.push_back(OrigPointer);
      }
    }
    else {
      Args.push_back(Arg);
    }
  }

  if (CalledF == nullptr
      || InstructionPropagator::isSpecialFunction(*CalledF))
    return;

  LLVM_DEBUG(dbgs() << "[taffo-err] Preparing errors for function call/invoke "
	<< I.getName() << "...\n");

  // Stop if we have reached the maximum recursion count.
  if (FCMap.maxRecursionCountReached(CalledF))
    return;

  // Now propagate the errors for this call.
  FunctionErrorPropagator CFEP(EPPass, *CalledF,
			       FCMap, RMap.getMetadataManager(), SloppyAA);
  CFEP.computeErrorsWithCopy(RMap, &Args, false);

  // Restore MemorySSA
  assert(FCopy != nullptr);
  MemSSA = &(EPPass.getAnalysis<MemorySSAWrapperPass>(*FCopy).getMSSA());
}

void
FunctionErrorPropagator::applyActualParametersErrors(RangeErrorMap &GlobRMap,
						     SmallVectorImpl<Value *> *Args) {
  assert(FCopy != nullptr);
  if (Args == nullptr)
    return;

  auto FArg = FCopy->arg_begin();
  auto FArgEnd = FCopy->arg_end();
  for (auto AArg = Args->begin(), AArgEnd = Args->end();
       AArg != AArgEnd && FArg != FArgEnd;
       ++AArg, ++FArg) {
    if (*AArg == nullptr)
      continue;
    if (!FArg->getType()->isPointerTy())
      continue;

    const AffineForm<inter_t> *Err = RMap.getError(&(*FArg));
    if (Err == nullptr) {
      Value *OrigPointer = MemSSAUtils::getOriginPointer(*MemSSA, &*FArg);
      Err = RMap.getError(OrigPointer);
      if (Err == nullptr)
	continue;
    }

    LLVM_DEBUG(dbgs() << "[taffo-err] Setting actual parameter (" << **AArg
	  << ") error " << static_cast<double>(Err->noiseTermsAbsSum()) << "\n");
    GlobRMap.setError(*AArg, *Err);
  }

  // Now update structs:
  GlobRMap.updateStructErrors(RMap, *Args);
}

void
FunctionErrorPropagator::attachErrorMetadata() {
  ValueToValueMapTy *VMap = FCMap.getValueToValueMap(&F);
  assert(VMap != nullptr);

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    Value *InstCopy = (Cloned) ? (*VMap)[cast<Value>(&*I)] : &*I;
    if (InstCopy == nullptr)
      continue;

    double Error = RMap.getOutputError(InstCopy);
    if (!std::isnan(Error)) {
      MetadataManager::setErrorMetadata(*I, Error);
    }

    CmpErrorMap::const_iterator CmpErr = CmpMap.find(InstCopy);
    if (CmpErr != CmpMap.end())
      MetadataManager::setCmpErrorMetadata(*I, CmpErr->second);
  }
}

bool FunctionErrorPropagator::checkOverflow(Instruction &I) {
  const FPInterval *Range = RMap.getRange(&I);
  const TType *Type;
  if (Range == nullptr
      || (Type = Range->getTType()) == nullptr)
    return false;

  return Range->Min < Type->getMinValueBound()
    || Range->Max > Type->getMaxValueBound();
}

void BBScheduler::enqueueChildren(BasicBlock *BB) {
  assert(BB != nullptr && "Null basic block.");

  // Do nothing if already visited.
  if (Set.count(BB))
    return;

  LLVM_DEBUG(dbgs() << "[taffo-err] Scheduling " << BB->getName() << ".\n");

  Set.insert(BB);

  Instruction *TI = BB->getTerminator();
  if (TI != nullptr) {
    Loop *L = LInfo.getLoopFor(BB);
    if (L == nullptr) {
      // Not part of a loop, just visit all unvisited successors.
      int c = TI->getNumSuccessors();
      for (int i=0; i<c; i++)
        enqueueChildren(TI->getSuccessor(i));
    }
    else {
      // Part of a loop:
      // visit exiting blocks first, so they are scheduled at the end.
      SmallVector<BasicBlock *, 2U> BodyQueue;
      int c = TI->getNumSuccessors();
      for (int i=0; i<c; i++) {
        BasicBlock *DestBB = TI->getSuccessor(i);
	if (isExiting(DestBB, L))
	  enqueueChildren(DestBB);
	else
	  BodyQueue.push_back(DestBB);
      }

      // If the header is also the exit, but not a latch,
      // it is visited also after the loop body
      if (L->isLoopExiting(BB) && !L->isLoopLatch(BB))
	Queue.push_back(BB);

      for (BasicBlock *BodyBB : BodyQueue)
	enqueueChildren(BodyBB);
    }
  }
  Queue.push_back(BB);
}

bool BBScheduler::isExiting(BasicBlock *Dst, Loop *L) const {
  assert(L != nullptr);

  if (!L->contains(Dst))
    return true;

  return L->isLoopExiting(Dst);
}


} // end namespace ErrorProp
