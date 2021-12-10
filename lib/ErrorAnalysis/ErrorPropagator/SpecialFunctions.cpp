#include "Propagators.h"

namespace ErrorProp
{

using namespace llvm;

bool InstructionPropagator::isSqrt(Function &F)
{
  StringRef FName = F.getName();
  return FName == "sqrtf" || FName == "sqrt" || FName == "_ZSt4sqrtf" || (FName.find("sqrt") != StringRef::npos && FName.find("fixp") != StringRef::npos) || FName == "_ZSt4sqrtf_fixp";
}

bool InstructionPropagator::isLog(Function &F)
{
  StringRef FName = F.getName();
  return FName == "log" || FName == "logf" || FName == "_ZSt3logf" || (FName.find("log") != StringRef::npos && FName.find("fixp") != StringRef::npos);
}

bool InstructionPropagator::isExp(Function &F)
{
  StringRef FName = F.getName();
  return FName == "expf" || FName == "exp" || FName == "_ZSt3expf" || (FName.find("exp") != StringRef::npos && FName.find("fixp") != StringRef::npos);
}

bool InstructionPropagator::isAcos(Function &F)
{
  StringRef FName = F.getName();
  return F.getName() == "acos" || F.getName() == "acosf" || (FName.find("acos") != StringRef::npos && FName.find("fixp") != StringRef::npos);
}

bool InstructionPropagator::isAsin(Function &F)
{
  StringRef FName = F.getName();
  return F.getName() == "asin" || F.getName() == "asinf" || (FName.find("asin") != StringRef::npos && FName.find("fixp") != StringRef::npos);
  ;
}

bool InstructionPropagator::isSpecialFunction(Function &F)
{
  return F.arg_size() == 1U && (F.empty() || !F.hasName() || isSqrt(F) || isLog(F) || isExp(F) || isAcos(F) || isAsin(F));
}

bool InstructionPropagator::propagateSqrt(Instruction &I)
{
  LLVM_DEBUG(dbgs() << "(special: sqrt) ");
  auto *OpRE = getOperandRangeError(I, 0U);
  if (OpRE == nullptr || !OpRE->second.hasValue()) {
    LLVM_DEBUG(dbgs() << "no data.\n");
    return false;
  }

  const FPInterval *IRange = RMap.getRange(&I);
  AffineForm<inter_t> NewErr =
      LinearErrorApproximationDecr([](inter_t x) { return static_cast<inter_t>(0.5) / std::sqrt(x); },
                                   OpRE->first, OpRE->second.getValue()) +
      ((IRange) ? AffineForm<inter_t>(0.0, IRange->getRoundingError()) : AffineForm<inter_t>(0.0, OpRE->first.getRoundingError()));

  RMap.setError(&I, NewErr);

  LLVM_DEBUG(dbgs() << static_cast<double>(NewErr.noiseTermsAbsSum()) << ".\n");
  return true;
}

bool InstructionPropagator::propagateLog(Instruction &I)
{
  LLVM_DEBUG(dbgs() << "(special: log) ");
  auto *OpRE = getOperandRangeError(I, 0U);
  if (OpRE == nullptr || !OpRE->second.hasValue()) {
    LLVM_DEBUG(dbgs() << "no data.\n");
    return false;
  }

  const FPInterval *IRange = RMap.getRange(&I);
  AffineForm<inter_t> NewErr =
      LinearErrorApproximationDecr([](inter_t x) { return static_cast<inter_t>(1) / x; },
                                   OpRE->first, OpRE->second.getValue()) +
      ((IRange) ? AffineForm<inter_t>(0.0, IRange->getRoundingError()) : AffineForm<inter_t>(0.0, OpRE->first.getRoundingError()));

  RMap.setError(&I, NewErr);

  LLVM_DEBUG(dbgs() << static_cast<double>(NewErr.noiseTermsAbsSum()) << ".\n");
  return true;
}

bool InstructionPropagator::propagateExp(Instruction &I)
{
  LLVM_DEBUG(dbgs() << "(special: exp) ");
  auto *OpRE = getOperandRangeError(I, 0U);
  if (OpRE == nullptr || !OpRE->second.hasValue()) {
    LLVM_DEBUG(dbgs() << "no data.\n");
    return false;
  }

  const FPInterval *IRange = RMap.getRange(&I);
  AffineForm<inter_t> NewErr =
      LinearErrorApproximationIncr([](inter_t x) { return std::exp(x); },
                                   OpRE->first, OpRE->second.getValue()) +
      ((IRange) ? AffineForm<inter_t>(0.0, IRange->getRoundingError()) : AffineForm<inter_t>(0.0, OpRE->first.getRoundingError()));

  RMap.setError(&I, NewErr);

  LLVM_DEBUG(dbgs() << static_cast<double>(NewErr.noiseTermsAbsSum()) << ".\n");
  return true;
}

bool InstructionPropagator::propagateAcos(Instruction &I)
{
  LLVM_DEBUG(dbgs() << "(special: acos) ");
  auto *OpRE = getOperandRangeError(I, 0U);
  if (OpRE == nullptr || !OpRE->second.hasValue()) {
    LLVM_DEBUG(dbgs() << "no data.\n");
    return false;
  }
  Interval<inter_t> R(std::max(static_cast<inter_t>(-0.99), OpRE->first.Min),
                      std::min(static_cast<inter_t>(0.99), OpRE->first.Max));

  const FPInterval *IRange = RMap.getRange(&I);
  AffineForm<inter_t> NewErr =
      LinearErrorApproximationIncr([](inter_t x) { return static_cast<inter_t>(-1) / std::sqrt(1 - x * x); },
                                   R, OpRE->second.getValue()) +
      ((IRange) ? AffineForm<inter_t>(0.0, IRange->getRoundingError()) : AffineForm<inter_t>(0.0, OpRE->first.getRoundingError()));

  RMap.setError(&I, NewErr);

  LLVM_DEBUG(dbgs() << static_cast<double>(NewErr.noiseTermsAbsSum()) << ".\n");
  return true;
}

bool InstructionPropagator::propagateAsin(Instruction &I)
{
  LLVM_DEBUG(dbgs() << "(special: asin) ");
  auto *OpRE = getOperandRangeError(I, 0U);
  if (OpRE == nullptr || !OpRE->second.hasValue()) {
    LLVM_DEBUG(dbgs() << "no data.\n");
    return false;
  }
  Interval<inter_t> R(std::max(static_cast<inter_t>(-0.99), OpRE->first.Min),
                      std::min(static_cast<inter_t>(0.99), OpRE->first.Max));

  const FPInterval *IRange = RMap.getRange(&I);
  AffineForm<inter_t> NewErr =
      LinearErrorApproximationIncr([](inter_t x) { return static_cast<inter_t>(1) / std::sqrt(1 - x * x); },
                                   R, OpRE->second.getValue()) +
      ((IRange) ? AffineForm<inter_t>(0.0, IRange->getRoundingError()) : AffineForm<inter_t>(0.0, OpRE->first.getRoundingError()));

  RMap.setError(&I, NewErr);

  LLVM_DEBUG(dbgs() << static_cast<double>(NewErr.noiseTermsAbsSum()) << ".\n");
  return true;
}

bool InstructionPropagator::propagateSpecialCall(Instruction &I, Function &Called)
{
  assert(InstructionPropagator::isSpecialFunction(Called));
  if (isSqrt(Called)) {
    return propagateSqrt(I);
  } else if (isLog(Called)) {
    return propagateLog(I);
  } else if (isExp(Called)) {
    return propagateExp(I);
  } else if (isAcos(Called)) {
    return propagateAcos(I);
  } else if (isAsin(Called)) {
    return propagateAsin(I);
  } else {
    LLVM_DEBUG(dbgs() << "(special pass-through) ");
    return unOpErrorPassThrough(I);
  }
}

} // end of namespace ErrorProp
