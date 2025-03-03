#include "Propagators.h"

#include "MemSSAUtils.hpp"

#define DEBUG_TYPE "errorprop"

namespace ErrorProp
{

using namespace llvm;
using namespace mdutils;

const RangeErrorMap::RangeError *
InstructionPropagator::getConstantFPRangeError(ConstantFP *VFP)
{
  double CVal;
  if (VFP->getType()->isDoubleTy())
    CVal = VFP->getValueAPF().convertToDouble();
  else if (VFP->getType()->isFloatTy())
    CVal = VFP->getValueAPF().convertToFloat();
  else
    return nullptr;

  // Kludge! ;-)
  CVal = 1.0;

  FPInterval FPI(Interval<inter_t>(CVal, CVal));
  RMap.setRangeError(VFP, std::make_pair(FPI, AffineForm<inter_t>(0.0)));
  return RMap.getRangeError(VFP);
}

const RangeErrorMap::RangeError *
InstructionPropagator::getConstantRangeError(Instruction &I, ConstantInt *VInt,
                                             bool DoublePP,
                                             const FPType *FallbackTy)
{
  const RangeErrorMap::RangeError *RE = RMap.getRangeError(VInt);
  if (RE != nullptr)
    return RE;

  // We interpret the value of VInt with the same
  // fractional bits and sign of the result.
  LLVM_DEBUG(logInfo("(WARNING: constant with no range metadata, trying to guess type)"));
  const FPInterval *RInfo = RMap.getRange(&I);
  const FPType *Ty = nullptr;
  if (RInfo != nullptr)
    Ty = dyn_cast_or_null<FPType>(RInfo->getTType());
  if (Ty == nullptr && FallbackTy != nullptr)
    Ty = FallbackTy;

  FPInterval VRange;
  AffineForm<inter_t> Error;
  if (Ty != nullptr) {
    unsigned PointPos = Ty->getPointPos();
    if (DoublePP)
      PointPos *= 2U;
    int SPointPos = (Ty->isSigned()) ? -PointPos : PointPos;
    std::unique_ptr<FixedPointValue> VFPRange =
        FixedPointValue::createFromConstantInt(SPointPos, nullptr, VInt, VInt);
    VRange = VFPRange->getInterval();
    // We use the rounding error of this format as the only error.
    Error = (RMap.isExactConst())
                ? AffineForm<inter_t>(0)
                : AffineForm<inter_t>(0, Ty->getRoundingError());
  } else {
    VRange.Min = VRange.Max = VInt->getSExtValue();
    LLVM_DEBUG(dbgs() << "(WARNING: interpreting ConstantInt " << *VInt
                      << " as integer.)");
  }

  RMap.setRangeError(VInt, std::make_pair(VRange, Error));
  return RMap.getRangeError(VInt);
}

const RangeErrorMap::RangeError *
InstructionPropagator::getOperandRangeError(Instruction &I, Value *V,
                                            bool DoublePP, const FPType *FallbackTy)
{
  assert(V != nullptr);

  // If V is a Constant Int extract its value.
  ConstantInt *VInt = dyn_cast<ConstantInt>(V);
  if (VInt != nullptr)
    return getConstantRangeError(I, VInt, DoublePP, FallbackTy);

  ConstantFP *VFP = dyn_cast<ConstantFP>(V);
  if (VFP != nullptr)
    return getConstantFPRangeError(VFP);

  // Otherwise, check if Range and Error have already been computed.
  return RMap.getRangeError(V);
}

const RangeErrorMap::RangeError *
InstructionPropagator::getOperandRangeError(Instruction &I, unsigned Op,
                                            bool DoublePP,
                                            const FPType *FallbackTy)
{
  Value *V = I.getOperand(Op);
  if (V == nullptr)
    return nullptr;

  return getOperandRangeError(I, V, DoublePP, FallbackTy);
}

void InstructionPropagator::
    updateArgumentRE(Value *Pointer,
                     const RangeErrorMap::RangeError *NewRE)
{
  assert(Pointer != nullptr);
  assert(NewRE != nullptr);

  Pointer = taffo::MemSSAUtils::getOriginPointer(MemSSA, Pointer);
  if (Pointer != nullptr) {
    auto *PointerRE = RMap.getRangeError(Pointer);
    if (PointerRE == nullptr || !PointerRE->second.has_value() || PointerRE->second->noiseTermsAbsSum() < NewRE->second->noiseTermsAbsSum()) {
      RMap.setRangeError(Pointer, *NewRE);
      LLVM_DEBUG(dbgs() << "(Error of pointer (" << *Pointer << ") updated.) ");
    }
  }
}

bool InstructionPropagator::unOpErrorPassThrough(Instruction &I)
{
  // assert(isa<UnaryInstruction>(I) && "Must be Unary.");

  auto *OpRE = getOperandRangeError(I, 0U);
  if (OpRE == nullptr || !OpRE->second.has_value()) {
    LLVM_DEBUG(logInfoln("no data."));
    return false;
  }

  auto *DestRE = RMap.getRangeError(&I);
  if (DestRE == nullptr || DestRE->first.isUninitialized()) {
    // Add operand range and error to RMap.
    RMap.setRangeError(&I, *OpRE);
  } else {
    // Add only error to RMap.
    RMap.setError(&I, *OpRE->second);
  }

  LLVM_DEBUG(logErrorln(*OpRE));

  return true;
}

inter_t InstructionPropagator::computeMinRangeDiff(const FPInterval &R1,
                                                   const FPInterval &R2)
{
  // Check if the two ranges overlap.
  if (R1.Min <= R2.Max && R2.Min <= R1.Max) {
    return 0.0;
  }

  // Otherwise either R1 < R2 or R2 < R1.
  if (R1.Max < R2.Min) {
    // R1 < R2
    return R2.Min - R1.Max;
  }

  // Else R2 < R1
  assert(R2.Max < R1.Min);
  return R1.Min - R2.Max;
}

void InstructionPropagator::logInstruction(const llvm::Value &I)
{
  dbgs() << "[taffo-err] " << I << ": ";
}

void InstructionPropagator::logInfo(const llvm::StringRef Msg)
{
  dbgs() << Msg << " ";
}

void InstructionPropagator::logInfoln(const llvm::StringRef Msg)
{
  dbgs() << Msg << "\n";
}

void InstructionPropagator::logError(const AffineForm<inter_t> &Err)
{
  dbgs() << static_cast<double>(Err.noiseTermsAbsSum());
}

void InstructionPropagator::logError(const RangeErrorMap::RangeError &RE)
{
  if (RE.second.has_value())
    logError(RE.second.value());
  else
    dbgs() << "null";
}

void InstructionPropagator::logErrorln(const AffineForm<inter_t> &Err)
{
  logError(Err);
  dbgs() << "\n";
}

void InstructionPropagator::logErrorln(const RangeErrorMap::RangeError &RE)
{
  logError(RE);
  dbgs() << "\n";
}

} // end of namespace ErrorProp
