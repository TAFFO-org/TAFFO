//===-- RangeErrorMap.cpp - Range and Error Maps ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains definitons of members of classes
/// that map Instructions/Arguments to the corresponding
/// computed ranges and errors.
///
//===----------------------------------------------------------------------===//

#include "RangeErrorMap.h"

#include <utility>
#include "llvm/Support/Debug.h"
#include "TypeUtils.h"

namespace ErrorProp {

using namespace llvm;
using namespace mdutils;

#define DEBUG_TYPE "errorprop"

const FPInterval *RangeErrorMap::getRange(const Value *I) const {
  auto RError = REMap.find(I);
  if (RError == REMap.end()) {
    return nullptr;
  }
  return &((RError->second).first);
}

const AffineForm<inter_t> *RangeErrorMap::getError(const Value *I) const {
  auto RError = REMap.find(I);
  if (RError == REMap.end()) {
    return nullptr;
  }
  const Optional<AffineForm<inter_t> > &Error = RError->second.second;
  if (Error.hasValue())
    return Error.getPointer();
  else
    return nullptr;
}

const RangeErrorMap::RangeError*
RangeErrorMap::getRangeError(const Value *I) const {
  auto RE = REMap.find(I);
  if (RE == REMap.end()) {
    return nullptr;
  }
  return &(RE->second);
}

void RangeErrorMap::setError(const Value *I, const AffineForm<inter_t> &E) {
  // If Range does not exist, the default is created.
  auto RE = REMap.find(I);
  if (RE == REMap.end()) {
    REMap[I] = std::make_pair(Interval<inter_t>(std::numeric_limits<double>::quiet_NaN(),
						std::numeric_limits<double>::quiet_NaN()),
			      E);
  }
  else
    RE->second.second = E;

  double OutError = getOutputError(I);
  if (!std::isnan(OutError))
    TErrs.updateTarget(I, OutError);
}

void RangeErrorMap::setRangeError(const Value *I,
				  const RangeError &RE) {
  REMap[I] = RE;

  if (RE.second.hasValue()) {
    double OutError = getOutputError(RE);
    if (!std::isnan(OutError))
      TErrs.updateTarget(I, OutError);
  }
}

bool RangeErrorMap::retrieveRangeError(Instruction &I) {
  retrieveConstRanges(I);

  if (const StructInfo *SI = MDMgr->retrieveStructInfo(I)) {
    SEMap.createStructTreeFromMetadata(&I, SI);
    return false;
  }

  const InputInfo *II = MDMgr->retrieveInputInfo(I);
  if (II == nullptr || II->IRange == nullptr)
    return false;

  if (II->IError == nullptr) {
    REMap[&I] = std::make_pair(FPInterval(II), NoneType());
    return false;
  }
  else {
    REMap[&I] = std::make_pair(FPInterval(II), AffineForm<inter_t>(0.0, *II->IError));
    return true;
  }
}

void RangeErrorMap::retrieveRangeErrors(Function &F) {
  SmallVector<MDInfo *, 1U> REs;
  MDMgr->retrieveArgumentInputInfo(F, REs);

  auto REIt = REs.begin(), REEnd = REs.end();
  for (Function::arg_iterator Arg = F.arg_begin(), ArgE = F.arg_end();
       Arg != ArgE && REIt != REEnd; ++Arg, ++REIt) {
    if (*REIt == nullptr)
      continue;

    if (const InputInfo *II = dyn_cast<InputInfo>(*REIt)) {
      if (II->IRange == nullptr)
	continue;

      FPInterval FPI(II);

      LLVM_DEBUG(dbgs() << "Retrieving data for Argument " << Arg->getName() << "... "
	    << "Range: [" << static_cast<double>(FPI.Min) << ", "
	    << static_cast<double>(FPI.Max) << "], Error: ");

      if (FPI.hasInitialError()) {
	AffineForm<inter_t> Err(0.0, FPI.getInitialError());
	this->setRangeError(Arg, std::make_pair(FPI, Err));

	LLVM_DEBUG(dbgs() << FPI.getInitialError() << ".\n");
      }
      else {
	this->setRangeError(Arg, std::make_pair(FPI, NoneType()));

	LLVM_DEBUG(dbgs() << "none.\n");
      }
    }
    else {
      assert(taffo::fullyUnwrapPointerOrArrayType(Arg->getType())->isStructTy()
	     && "Must be a Struct Argument.");
      const StructInfo *SI = cast<StructInfo>(*REIt);
      SEMap.createStructTreeFromMetadata(Arg, SI);
    }
  }
}

void RangeErrorMap::applyArgumentErrors(Function &F,
					SmallVectorImpl<Value *> *Args) {
  if (Args == nullptr)
    return;

  auto FArg = F.arg_begin();
  auto FArgEnd = F.arg_end();
  for (auto AArg = Args->begin(), AArgEnd = Args->end();
       AArg != AArgEnd && FArg != FArgEnd;
       ++AArg, ++FArg) {
    Value *AArgV = *AArg;
    const AffineForm<inter_t> *Err = this->getError(AArgV);
    if (Err == nullptr) {
      LLVM_DEBUG(
	    dbgs() << "[taffo-err] No pre-computed error available for formal parameter (" << *FArg << ")";
	    if (AArgV != nullptr)
	      dbgs() << "from actual parameter (" << *AArgV << ").\n";
	    else
	      dbgs() << ".\n";
	    );
      continue;
    }

    this->setError(&*FArg, *Err);

    LLVM_DEBUG(dbgs() << "[taffo-err] Pre-computed error applied to formal parameter (" << *FArg
	  << ") from actual parameter (" << *AArgV
	  << "): " << static_cast<double>(Err->noiseTermsAbsSum()) << ".\n");
  }
}

void RangeErrorMap::retrieveRangeError(GlobalObject &V) {
  if (V.getValueType()->isStructTy()) {
    const StructInfo *SI = MDMgr->retrieveStructInfo(V);
    if (SI == nullptr) {
      LLVM_DEBUG(dbgs() << "[taffo-err] No struct data for Global Variable " << V << ".\n");
      return;
    }
    SEMap.createStructTreeFromMetadata(&V, SI);
    return;
  }

  LLVM_DEBUG(dbgs() << "[taffo-err] Retrieving data for Global Variable " << V << "... ");

  const InputInfo *II = MDMgr->retrieveInputInfo(V);
  if (II == nullptr) {
    LLVM_DEBUG(dbgs() << "ignored (no data).\n");
    return;
  }

  FPInterval FPI(II);

  LLVM_DEBUG(dbgs() << "Range: [" << static_cast<double>(FPI.Min) << ", "
	<< static_cast<double>(FPI.Max) << "], Error: ");

  if (FPI.hasInitialError()) {
    REMap[&V] = std::make_pair(FPI, AffineForm<inter_t>(0.0, FPI.getInitialError()));

    LLVM_DEBUG(dbgs() << FPI.getInitialError() << ".\n");
  }
  else {
    REMap[&V] = std::make_pair(FPI, NoneType());

    LLVM_DEBUG(dbgs() << "none.\n");
  }
}

const RangeErrorMap::RangeError *
RangeErrorMap::getStructRangeError(Value *V) const {
  return SEMap.getFieldError(V);
}

void RangeErrorMap::setStructRangeError(Value *V, const RangeError &RE) {
  SEMap.setFieldError(V, RE);
}

void RangeErrorMap::updateTargets(const RangeErrorMap &Other) {
  this->TErrs.updateAllTargets(Other.TErrs);
}

double RangeErrorMap::computeRelativeError(const RangeError &RE) {
  double divisor = std::max(std::abs(RE.first.Min), std::abs(RE.first.Max));
  if (divisor != 0)
    return RE.second->noiseTermsAbsSum() / divisor;
  else
    return std::numeric_limits<double>::quiet_NaN();
}

double RangeErrorMap::getOutputError(const llvm::Value *V) const {
  const RangeError *RE = getRangeError(V);
  if (RE && RE->second.hasValue()) {
    return getOutputError(*RE);
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

double RangeErrorMap::getOutputError(const RangeError &RE) const {
  return (OutputAbsolute) ? RE.second->noiseTermsAbsSum() : computeRelativeError(RE);
}

void TargetErrors::updateTarget(const Value *V, const inter_t &Error) {
  if (isa<Instruction>(V))
    updateTarget(cast<Instruction>(V), Error);
  else if (isa<GlobalVariable>(V))
    updateTarget(cast<GlobalVariable>(V), Error);
}

void TargetErrors::updateTarget(const Instruction *I, const inter_t &Error) {
  assert(I != nullptr);
  Optional<StringRef> Target = MetadataManager::retrieveTargetMetadata(*I);
  if (Target.hasValue())
    updateTarget(Target.getValue(), Error);
}

void TargetErrors::updateTarget(const GlobalVariable *V, const inter_t &Error) {
  assert(V != nullptr);
  Optional<StringRef> Target = MetadataManager::retrieveTargetMetadata(*V);
  if (Target.hasValue())
    updateTarget(Target.getValue(), Error);
}

void TargetErrors::updateTarget(StringRef T, const inter_t &Error) {
  Targets[T] = std::max(Targets[T], Error);
  LLVM_DEBUG(dbgs() << "(Target " << T << " updated with "
	     << static_cast<double>(Error) << ") ");
}

void TargetErrors::updateAllTargets(const TargetErrors &Other) {
  for (auto &T : Other.Targets)
    this->updateTarget(T.first, T.second);
}

inter_t TargetErrors::getErrorForTarget(StringRef T) const {
  auto Error = Targets.find(T);
  if (Error == Targets.end())
    return 0;

  return Error->second;
}

void TargetErrors::printTargetErrors(raw_ostream &OS) const {
  for (auto &T : Targets) {
    OS << "Computed error for target " << T.first << ": "
       << static_cast<double>(T.second) << "\n";
  }
}

void RangeErrorMap::retrieveConstRanges(const Instruction &I) {
  SmallVector<InputInfo *, 2U> CII;
  MDMgr->retrieveConstInfo(I, CII);
  if (CII.empty())
    return;

  assert(CII.size() == I.getNumOperands() && "Malformed ConstInfo metadata.");
  for (unsigned Idx = 0; Idx < I.getNumOperands(); ++Idx) {
    InputInfo *II = CII[Idx];
    if (II != nullptr && isa<Constant>(I.getOperand(Idx))) {
      AffineForm<inter_t> Error = (!ExactConst && II->IType && cast<FPType>(II->IType.get())->getPointPos() != 0)
	? AffineForm<inter_t>(0.0, II->IType->getRoundingError())
	: AffineForm<inter_t>();
      REMap[I.getOperand(Idx)] = std::make_pair(FPInterval(II), Error);
    }
  }
}

} // end namespace ErrorProp
