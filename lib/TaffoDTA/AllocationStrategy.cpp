#include "AllocationStrategy.hpp"
#include "DTAConfig.hpp"
#include "DataTypeAllocationPass.hpp"
#include "Debug/Logger.hpp"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "taffo-dta"

using namespace llvm;
using namespace tda;
using namespace taffo;

bool FixedPointOnlyStrategy::apply(std::shared_ptr<ScalarInfo>& scalarInfo, Value* value) {
  if (!scalarInfo->isConversionEnabled()) {
    LLVM_DEBUG(log() << "conversion disabled: skipping\n");
    return false;
  }

  if (scalarInfo->numericType) {
    LLVM_DEBUG(log() << "numeric type already assigned\n");
    return true;
  }

  Range* range = scalarInfo->range.get();
  if (!range) {
    LLVM_DEBUG(log() << "no range: skipping\n");
    return false;
  }

  double greatest = DataTypeAllocationPass::getGreatest(scalarInfo, value, range);
  LLVM_DEBUG(log() << "maximum value involved: " << greatest << "\n");

  FixedPointTypeGenError fpgerr;

  /* Testing maximum type for operands, not deciding type yet */
  fixedPointInfoFromRange(Range(0, greatest), &fpgerr, totalBits, fracThreshold, maxTotalBits, totalBits);
  if (fpgerr == FixedPointTypeGenError::NoError) {
    FixedPointInfo res = fixedPointInfoFromRange(*range, &fpgerr, totalBits, fracThreshold, maxTotalBits, totalBits);
    if (fpgerr == FixedPointTypeGenError::NoError) {
      LLVM_DEBUG(log().log("converting to ").logln(res, Logger::Green));
      scalarInfo->numericType = res.clone();
      return true;
    }

    LLVM_DEBUG(
      Logger& logger = log();
      logger << Logger::Red << "Error generating fixed point type: ";
      switch (fpgerr) {
      case FixedPointTypeGenError::InvalidRange:            logger << "invalid range\n"; break;
      case FixedPointTypeGenError::UnboundedRange:          logger << "unbounded range\n"; break;
      case FixedPointTypeGenError::NotEnoughIntAndFracBits:
      case FixedPointTypeGenError::NotEnoughFracBits:       logger << "result not representable\n"; break;
      default:                                              logger << "error code unknown\n";
      }
      logger << Logger::Reset;);
  }
  else
    LLVM_DEBUG(log() << "operands not representable as fixed point with specified constraints\n");

  /* We failed, try to keep original type */
  Type* type = getFullyUnwrappedType(value);
  if (type->isFloatingPointTy()) {
    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(type->getTypeID(), greatest));
    scalarInfo->numericType = res;
    LLVM_DEBUG(log() << "keeping original type: " << *res << "\n");
    return true;
  }

  LLVM_DEBUG(log() << "original type was not floating point: skipping\n");
  return false;
}

std::shared_ptr<NumericTypeInfo> FixedPointOnlyStrategy::merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                                               const std::shared_ptr<NumericTypeInfo>& fpu) {
  std::shared_ptr<FixedPointInfo> fpv_fixed = dynamic_ptr_cast<FixedPointInfo>(fpv);
  std::shared_ptr<FixedPointInfo> fpu_fixed = dynamic_ptr_cast<FixedPointInfo>(fpu);

  int sign_v = fpv_fixed->isSigned() ? 1 : 0;
  int int_v = int(fpv_fixed->getBits()) - fpv_fixed->getFractionalBits() - sign_v;
  int sign_u = fpu_fixed->isSigned() ? 1 : 0;
  int int_u = int(fpu_fixed->getBits()) - fpu_fixed->getFractionalBits() - sign_u;

  int sign_res = std::max(sign_u, sign_v);
  int int_res = std::max(int_u, int_v);
  int size_res = std::max(fpv_fixed->getBits(), fpu_fixed->getBits());
  int frac_res = size_res - int_res - sign_res;
  if (sign_res + int_res + frac_res != size_res || frac_res < 0)
    return nullptr; // Invalid format.
  return std::make_shared<FixedPointInfo>(sign_res, size_res, frac_res);
}

bool FixedPointOnlyStrategy::isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                                         std::shared_ptr<NumericTypeInfo> userNumericType) {
  std::shared_ptr<FixedPointInfo> fpv = dynamic_ptr_cast<FixedPointInfo>(valueNumericType);
  std::shared_ptr<FixedPointInfo> fpu = dynamic_ptr_cast<FixedPointInfo>(userNumericType);
  if (!fpv || !fpu) {
    LLVM_DEBUG(log() << "not attempting merge of " << valueNumericType->toString() << ", "
                     << valueNumericType->toString() << " because one is not a FPType\n");
    return false;
  }

  return fpv->getBits() == fpu->getBits()
      && (std::abs(int(fpv->getFractionalBits()) - int(fpu->getFractionalBits()))
          + (fpv->isSigned() == fpu->isSigned() ? 0 : 1))
           <= similarBits;
}

bool FloatingPointOnlyStrategy::apply(std::shared_ptr<ScalarInfo>& scalarInfo, Value* value) {
  if (!scalarInfo->isConversionEnabled()) {
    LLVM_DEBUG(log() << "conversion disabled: skipping\n");
    return false;
  }

  if (scalarInfo->numericType) {
    LLVM_DEBUG(log() << "numeric type already assigned: skipping\n");
    return true;
  }

  Range* rng = scalarInfo->range.get();
  if (rng == nullptr) {
    LLVM_DEBUG(log() << "no range: skipping\n");
    return false;
  }

  double greatest = DataTypeAllocationPass::getGreatest(scalarInfo, value, rng);

  FloatingPointInfo::FloatStandard standard;
  if (UseFloat == "f16")
    standard = FloatingPointInfo::Float_half;
  else if (UseFloat == "f32")
    standard = FloatingPointInfo::Float_float;
  else if (UseFloat == "f64")
    standard = FloatingPointInfo::Float_double;
  else if (UseFloat == "bf16")
    standard = FloatingPointInfo::Float_bfloat;
  else {
    errs() << "Invalid format " << UseFloat << " specified to the -usefloat argument\n";
    abort();
  }
  // // auto standard = static_cast<mdutils::FloatType::FloatStandard>(ForceFloat.getValue());

  // standard = FloatingPointInfo::Float_double;

  auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(standard, greatest));
  double maxRep =
    std::max(std::abs(res->getMaxValueBound().convertToDouble()), std::abs(res->getMinValueBound().convertToDouble()));
  LLVM_DEBUG(log() << "maximum value representable: " << maxRep << "\n");

  if (greatest >= maxRep) {
    LLVM_DEBUG(log() << "cannot force conversion to float " << res << " because max value is not representable\n");
  }
  else {
    LLVM_DEBUG(log() << "forcing conversion to float\n");
    scalarInfo->numericType = res;
    return true;
  }

  /* We failed, try to keep original type */
  Type* type = getFullyUnwrappedType(value);
  if (type->isFloatingPointTy()) {
    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(type->getTypeID(), greatest));
    scalarInfo->numericType = res;
    LLVM_DEBUG(log() << "keeping original type " << *res << "\n");
    return true;
  }

  LLVM_DEBUG(log() << "the original type was not floating point: skipping\n");
  return false;
}

std::shared_ptr<NumericTypeInfo> FloatingPointOnlyStrategy::merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                                                  const std::shared_ptr<NumericTypeInfo>& fpu) {
  if (isa<FloatingPointInfo>(fpu.get()) && isa<FloatingPointInfo>(fpv.get())) {
    std::shared_ptr<FloatingPointInfo> a = dynamic_ptr_cast<FloatingPointInfo>(fpu);
    std::shared_ptr<FloatingPointInfo> b = dynamic_ptr_cast<FloatingPointInfo>(fpv);
    FloatingPointInfo::FloatStandard maxStd = std::max(a->getStandard(), b->getStandard());
    double maxMax = std::max(a->getGreatestNumber(), b->getGreatestNumber());
    return std::make_shared<FloatingPointInfo>(maxStd, maxMax);
  }
  llvm_unreachable("unknown numericType subclass");
}

// dunmmy strategy, always return true
bool FloatingPointOnlyStrategy::isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                                            std::shared_ptr<NumericTypeInfo> userNumericType) {

  std::shared_ptr<FloatingPointInfo> fpv = dynamic_ptr_cast<FloatingPointInfo>(valueNumericType);
  std::shared_ptr<FloatingPointInfo> fpu = dynamic_ptr_cast<FloatingPointInfo>(userNumericType);
  if (!fpv || !fpu) {
    LLVM_DEBUG(log() << "not attempting merge of " << valueNumericType->toString() << ", "
                     << valueNumericType->toString() << " because one is not a FixedPointType\n");
    return false;
  }

  return true;
}

// dummy strategy, use fixed if integer part of rng-> max is even use floating otherwise
bool FixedFloatingPointStrategy::apply(std::shared_ptr<ScalarInfo>& scalarInfo, Value* value) {
  if (!scalarInfo->isConversionEnabled()) {
    LLVM_DEBUG(log() << "conversion disabled: skipping\n");
    return false;
  }

  if (scalarInfo->numericType) {
    LLVM_DEBUG(log() << "numeric type already assigned: skipping\n");
    return true;
  }

  Range* rng = scalarInfo->range.get();
  if (rng == nullptr) {
    LLVM_DEBUG(log() << "no range: skipping\n");
    return false;
  }

  double greatest = DataTypeAllocationPass::getGreatest(scalarInfo, value, rng);

  if ((int) rng->max % 2 == 0) {
    FixedPointTypeGenError fpgerr;

    /* Testing maximum type for operands, not deciding type yet */
    fixedPointInfoFromRange(Range(0, greatest), &fpgerr, totalBits, fracThreshold, maxTotalBits, totalBits);
    if (fpgerr == FixedPointTypeGenError::NoError) {
      FixedPointInfo res = fixedPointInfoFromRange(*rng, &fpgerr, totalBits, fracThreshold, maxTotalBits, totalBits);
      if (fpgerr == FixedPointTypeGenError::NoError) {
        LLVM_DEBUG(log().log("converting to ").logln(res, Logger::Green));
        scalarInfo->numericType = res.clone();
        return true;
      }

      LLVM_DEBUG(
        Logger& logger = log();
        logger << Logger::Red << "error generating fixed point type: \n";
        switch (fpgerr) {
        case FixedPointTypeGenError::InvalidRange:            logger << "invalid range\n"; break;
        case FixedPointTypeGenError::UnboundedRange:          logger << "unbounded range\n"; break;
        case FixedPointTypeGenError::NotEnoughIntAndFracBits:
        case FixedPointTypeGenError::NotEnoughFracBits:       logger << "result not representable\n"; break;
        default:                                              logger << "error code unknown\n";
        }
        logger << Logger::Reset;);
    }
    else
      LLVM_DEBUG(log() << "operands not representable as fixed point with specified constraints\n");
  }
  else {

    FloatingPointInfo::FloatStandard standard;
    if (UseFloat == "f16")
      standard = FloatingPointInfo::Float_half;
    else if (UseFloat == "f32")
      standard = FloatingPointInfo::Float_float;
    else if (UseFloat == "f64")
      standard = FloatingPointInfo::Float_double;
    else if (UseFloat == "bf16")
      standard = FloatingPointInfo::Float_bfloat;
    else {
      errs() << "Invalid format " << UseFloat << " specified to the -usefloat argument\n";
      abort();
    }

    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(standard, greatest));
    double maxRep = std::max(std::abs(res->getMaxValueBound().convertToDouble()),
                             std::abs(res->getMinValueBound().convertToDouble()));
    LLVM_DEBUG(log() << "maximum value representable: " << maxRep << "\n");

    if (greatest >= maxRep) {
      LLVM_DEBUG(log() << "cannot force conversion to float " << res << " because max value is not representable\n");
    }
    else {
      LLVM_DEBUG(log() << "forcing conversion to float " << res << "\n");
      scalarInfo->numericType = res;
      return true;
    }
  }

  /* We failed, try to keep original type */
  Type* Ty = getFullyUnwrappedType(value);
  if (Ty->isFloatingPointTy()) {
    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(Ty->getTypeID(), greatest));
    scalarInfo->numericType = res;
    LLVM_DEBUG(log() << "keeping original type " << res << "\n");
    return true;
  }

  LLVM_DEBUG(log() << "original type was not floating point: skipping\n");
  return false;
}

// Dummy strategy, always return true
bool FixedFloatingPointStrategy::isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                                             std::shared_ptr<NumericTypeInfo> userNumericType) {

  return true;
}

// Dummy strategy, always return the fpu type
std::shared_ptr<NumericTypeInfo> FixedFloatingPointStrategy::merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                                                   const std::shared_ptr<NumericTypeInfo>& fpu) {
  if (isa<FloatingPointInfo>(fpu.get()))
    return dynamic_ptr_cast<FloatingPointInfo>(fpu)->clone();
  else
    return dynamic_ptr_cast<FixedPointInfo>(fpu)->clone();
}
