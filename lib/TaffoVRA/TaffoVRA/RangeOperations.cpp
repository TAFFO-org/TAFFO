#include "PtrCasts.hpp"
#include "RangeOperations.hpp"
#include "RangeOperationsCallWhitelist.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/Support/Casting.h>

#include <assert.h>
#include <map>

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace taffo;

//-----------------------------------------------------------------------------
// Wrappers
//-----------------------------------------------------------------------------

/** Handle binary instructions */
std::shared_ptr<Range> taffo::handleBinaryInstruction(const std::shared_ptr<Range> op1,
                                                      const std::shared_ptr<Range> op2,
                                                      const unsigned OpCode) {
  switch (OpCode) {
  case Instruction::Add:
  case Instruction::FAdd:
    return handleAdd(op1, op2);
    break;
  case Instruction::Sub:
  case Instruction::FSub:
    return handleSub(op1, op2);
    break;
  case Instruction::Mul:
  case Instruction::FMul:
    return handleMul(op1, op2);
    break;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
    return handleDiv(op1, op2);
    break;
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
    return handleRem(op1, op2);
    break;
  case Instruction::Shl:
    return handleShl(op1, op2);
  case Instruction::LShr: // TODO implement
  case Instruction::AShr:
    return handleAShr(op1, op2);
  case Instruction::And:  // TODO implement
  case Instruction::Or:   // TODO implement
  case Instruction::Xor:  // TODO implement
    break;
  default:
    assert(false);        // unsupported operation
    break;
  }
  return nullptr;
}

std::shared_ptr<Range> taffo::handleUnaryInstruction(const std::shared_ptr<Range> op, const unsigned OpCode) {
  if (!op)
    return nullptr;

  switch (OpCode) {
  case Instruction::FNeg:
    return std::make_shared<Range>(-op->max, -op->min);
    break;
  default:
    assert(false); // unsupported operation
    break;
  }
  return nullptr;
}

/** Cast instructions */
std::shared_ptr<Range>
taffo::handleCastInstruction(const std::shared_ptr<Range> scalar, const unsigned OpCode, const Type* dest) {
  switch (OpCode) {
  case Instruction::Trunc:
    return handleTrunc(scalar, dest);
    break;
  case Instruction::ZExt:
  case Instruction::SExt:
    return copyRange(scalar);
    break;
  case Instruction::FPToUI:
    return handleCastToUI(scalar);
    break;
  case Instruction::FPToSI:
    return handleCastToSI(scalar);
    break;
  case Instruction::UIToFP:
  case Instruction::SIToFP:
    return copyRange(scalar);
    break;
  case Instruction::FPTrunc:
    return handleFPTrunc(scalar, dest);
  case Instruction::FPExt:
    return copyRange(scalar);
    break;
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
    return handleCastToSI(scalar);
    break;
  case Instruction::BitCast: // TODO check
    return copyRange(scalar);
    break;
  case Instruction::AddrSpaceCast:
    return copyRange(scalar);
    break;
  default:
    assert(false); // unsupported operation
    break;
  }
  return nullptr;
}

/** Return true if this function call can be handled by taffo::handleMathCallInstruction */
bool taffo::isMathCallInstruction(const std::string& function) { return functionWhiteList.count(function); }

/** Handle call to known math functions. Return nullptr if unknown */
std::shared_ptr<Range> taffo::handleMathCallInstruction(const std::list<std::shared_ptr<Range>>& ops,
                                                        const std::string& function) {
  const auto it = functionWhiteList.find(function);
  if (it != functionWhiteList.end())
    return it->second(ops);
  return nullptr;
}

/** Handle call to known math functions. Return nullptr if unknown */
std::shared_ptr<Range> taffo::handleCompare(const std::list<std::shared_ptr<Range>>& ops,
                                            const CmpInst::Predicate pred) {
  switch (pred) {
  case CmpInst::Predicate::FCMP_FALSE:
    return getAlwaysFalse();
  case CmpInst::Predicate::FCMP_TRUE:
    return getAlwaysTrue();
  default:
    break;
  }

  // from now on only 2 operators compare
  assert(ops.size() > 1 && "too few operators in compare instruction");
  assert(ops.size() <= 2 && "too many operators in compare instruction");

  // extract values for easy access
  std::shared_ptr<Range> lhs = ops.front();
  std::shared_ptr<Range> rhs = ops.back();
  // if unavailable data, nothing can be said
  if (!lhs || !rhs)
    return getGenericBoolRange();

  // NOTE: not dealing with Ordered / Unordered variants
  switch (pred) {
  case CmpInst::Predicate::FCMP_OEQ:
  case CmpInst::Predicate::FCMP_UEQ:
  case CmpInst::Predicate::ICMP_EQ:
    if (lhs->min == lhs->max && rhs->min == rhs->max && lhs->min == rhs->min)
      return getAlwaysTrue();
    else if (lhs->max < rhs->min || rhs->max < lhs->min)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OGT:
  case CmpInst::Predicate::FCMP_UGT:
  case CmpInst::Predicate::ICMP_UGT:
  case CmpInst::Predicate::ICMP_SGT:
    if (lhs->min > rhs->max)
      return getAlwaysTrue();
    else if (lhs->max <= rhs->min)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OGE:
  case CmpInst::Predicate::FCMP_UGE:
  case CmpInst::Predicate::ICMP_UGE:
  case CmpInst::Predicate::ICMP_SGE:
    if (lhs->min >= rhs->max)
      return getAlwaysTrue();
    else if (lhs->max < rhs->min)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OLT:
  case CmpInst::Predicate::FCMP_ULT:
  case CmpInst::Predicate::ICMP_ULT:
  case CmpInst::Predicate::ICMP_SLT:
    if (lhs->max < rhs->min)
      return getAlwaysTrue();
    else if (lhs->min >= rhs->max)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_OLE:
  case CmpInst::Predicate::FCMP_ULE:
  case CmpInst::Predicate::ICMP_ULE:
  case CmpInst::Predicate::ICMP_SLE:
    if (lhs->max <= rhs->min)
      return getAlwaysTrue();
    else if (lhs->min > rhs->max)
      return getAlwaysFalse();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_ONE:
  case CmpInst::Predicate::FCMP_UNE:
  case CmpInst::Predicate::ICMP_NE:
    if (lhs->min == lhs->max && rhs->min == rhs->max && lhs->min == rhs->min)
      return getAlwaysFalse();
    else if (lhs->max < rhs->min || rhs->max < lhs->min)
      return getAlwaysTrue();
    else
      return getGenericBoolRange();
    break;
  case CmpInst::Predicate::FCMP_ORD: // none of the operands is NaN
  case CmpInst::Predicate::FCMP_UNO: // one of the operand is NaN
    // TODO implement
    break;
  default:
    break;
  }
  return nullptr;
}

//-----------------------------------------------------------------------------
// Arithmetic
//-----------------------------------------------------------------------------

/** operator+ */
std::shared_ptr<Range> taffo::handleAdd(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  double a = op1->min + op2->min;
  double b = op1->max + op2->max;
  return std::make_shared<Range>(a, b);
}

/** operator- */
std::shared_ptr<Range> taffo::handleSub(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  double a = op1->min - op2->max;
  double b = op1->max - op2->min;
  return std::make_shared<Range>(a, b);
}

/** operator* */
std::shared_ptr<Range> taffo::handleMul(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  if (op1 == op2) {
    // This is a square.
    double a = op1->min * op1->min;
    double b = op1->max * op1->max;
    double r1 = (op1->min <= 0.0 && op1->max >= 0) ? 0.0 : std::min(a, b);
    double r2 = std::max(a, b);
    return std::make_shared<Range>(r1, r2);
  }
  double a = op1->min * op2->min;
  double b = op1->max * op2->max;
  double c = op1->min * op2->max;
  double d = op1->max * op2->min;
  const double r1 = std::min({a, b, c, d});
  const double r2 = std::max({a, b, c, d});
  return std::make_shared<Range>(r1, r2);
}

/** operator/ */
std::shared_ptr<Range> taffo::handleDiv(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  double op2_min, op2_max;
  // Avoid division by 0
#define DIV_EPS (static_cast<double>(1e-8))
  if (op2->max <= 0) {
    op2_min = std::min(op2->min, -DIV_EPS);
    op2_max = std::min(op2->max, -DIV_EPS);
  }
  else if (op2->min < 0) {
    op2_min = -DIV_EPS;
    op2_max = +DIV_EPS;
  }
  else {
    op2_min = std::max(op2->min, +DIV_EPS);
    op2_max = std::max(op2->max, +DIV_EPS);
  }
  double a = op1->min / op2_min;
  double b = op1->max / op2_max;
  double c = op1->min / op2_max;
  double d = op1->max / op2_min;
  const double r1 = std::min({a, b, c, d});
  const double r2 = std::max({a, b, c, d});
  return std::make_shared<Range>(r1, r2);
}

double getRemMin(double op1_min, double op1_max, double op2_min, double op2_max);
double getRemMax(double op1_min, double op1_max, double op2_min, double op2_max);

/** operator% */
std::shared_ptr<Range> taffo::handleRem(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  const double min_value = getRemMin(op1->min, op1->max, op2->min, op2->max);
  const double max_value = getRemMax(op1->min, op1->max, op2->min, op2->max);
  return std::make_shared<Range>(min_value, max_value);
}

double getRemMin(double op1_min, double op1_max, double op2_min, double op2_max) {
  // the sign of the second operand does not affect the result, we always mirror negative into positive
  if (op2_max < 0)
    return getRemMin(op1_min, op1_max, -op2_max, -op2_min);
  if (op2_min < 0) {
    // we have to split second operand range into negative and positive parts and calculate min separately
    double neg = getRemMin(op1_min, op1_max, 1, -op2_min);
    double pos = getRemMin(op1_min, op1_max, 0, op2_max);
    return std::min(neg, pos);
  }
  if (op1_min >= 0) {
    // this is the case when remainder will always return a non-negative result
    // if any of the limits are 0, the min will always be 0
    if (op1_min == 0.0 || op1_max == 0.0)
      return 0.0;
    // the intervals are intersecting, there is always going to be n % n = 0
    if (op1_max >= op2_min && op1_min <= op2_max)
      return 0.0;
    // the first argument range is strictly lower than the second,
    // the mod is always going to return values from the first interval, just take the lowest
    if (op1_max < op2_min)
      return op1_min;
    // the first range is strictly higher that the second
    // we cannot tell the exact min, so return 0 as this is the lowest it can be
    return 0.0;
  }
  else if (op1_max < 0) {
    // this is the case when % will always return negative result
    // mirror the interval into positives and calculate max with "-" sign as the minimum
    double neg = -getRemMax(-op1_max, -op1_min, op2_min, op2_max);
    return neg;
  }
  else {
    // we need to split the interval into the negative and positive parts
    // first, we take the negative part of the interval [op1_min, -1]
    // we mirror it to [1, -op1_min], which is going to be positive
    // we calculate the max and take it with the "-" sign as the minimum value
    double neg = -getRemMax(1.0, -op1_min, op2_min, op2_max);
    // for the positive part we calculate it the standard way
    double pos = getRemMin(0.0, op1_max, op2_min, op2_max);
    return std::min(neg, pos);
  }
}

double getRemMax(double op1_min, double op1_max, double op2_min, double op2_max) {
  if (op1_min >= 0) {
    // this is the case when % will always return non-negative result
    // the range might include n*op2_max+(op2_max-1) value that will be the max
    if (op1_max >= op2_max)
      return op2_max - 1;
    // op1_max < op2_max, so op1_max % op2_max = op1_max
    return op1_max;
  }
  else if (op1_max < 0) {
    // this is the case when remainder will always return a negative result, we need to choose the highest max
    // mirror the interval and calculate the min, take it with "-" sign as max
    return -getRemMin(-op1_max, -op1_min, op2_min, op2_max);
  }
  else {
    // we can ignore the negative part of the interval as it always will be lower than the positive
    return getRemMax(0.0, op1_max, op2_min, op2_max);
  }
}

std::shared_ptr<Range> taffo::handleShl(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  // FIXME: it only works if no overflow occurs.
  if (!op1 || !op2)
    return nullptr;
  const unsigned sh_min = static_cast<unsigned>(op2->min);
  const unsigned sh_max = static_cast<unsigned>(op2->max);
  const long op_min = static_cast<long>(op1->min);
  const long op_max = static_cast<long>(op1->max);
  return std::make_shared<Range>(static_cast<double>(op_min << ((op_min < 0) ? sh_max : sh_min)),
                                 static_cast<double>(op_max << ((op_max < 0) ? sh_min : sh_max)));
}

std::shared_ptr<Range> taffo::handleAShr(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return nullptr;
  const unsigned sh_min = static_cast<unsigned>(op2->min);
  const unsigned sh_max = static_cast<unsigned>(op2->max);
  const long op_min = static_cast<long>(op1->min);
  const long op_max = static_cast<long>(op1->max);
  return std::make_shared<Range>(static_cast<double>(op_min >> ((op_min > 0) ? sh_max : sh_min)),
                                 static_cast<double>(op_max >> ((op_max > 0) ? sh_min : sh_max)));
}

/** Trunc */
std::shared_ptr<Range> taffo::handleTrunc(const std::shared_ptr<Range> op, const Type* dest) {
  using namespace llvm;
  if (!op)
    return nullptr;
  const IntegerType* itype = cast<IntegerType>(dest);

  APSInt imin(64U, true), imax(64U, true);
  bool isExact;
  APFloat(op->min).convertToInteger(imin, APFloatBase::roundingMode::TowardNegative, &isExact);
  APFloat(op->max).convertToInteger(imax, APFloatBase::roundingMode::TowardPositive, &isExact);
  APSInt new_imin(imin.trunc(itype->getBitWidth()));
  APSInt new_imax(imax.trunc(itype->getBitWidth()));

  return std::make_shared<Range>(new_imin.getExtValue(), new_imax.getExtValue());
}

/** CastToUInteger */
std::shared_ptr<Range> taffo::handleCastToUI(const std::shared_ptr<Range> op) {
  if (!op)
    return nullptr;
  const double r1 = static_cast<double>(static_cast<unsigned long>(op->min));
  const double r2 = static_cast<double>(static_cast<unsigned long>(op->max));
  return std::make_shared<Range>(r1, r2);
}

/** CastToUInteger */
std::shared_ptr<Range> taffo::handleCastToSI(const std::shared_ptr<Range> op) {
  if (!op)
    return nullptr;
  const double r1 = static_cast<double>(static_cast<long>(op->min));
  const double r2 = static_cast<double>(static_cast<long>(op->max));
  return std::make_shared<Range>(r1, r2);
}

/** FPTrunc */
std::shared_ptr<Range> taffo::handleFPTrunc(const std::shared_ptr<Range> gop, const Type* dest) {
  if (!gop)
    return nullptr;
  assert(dest && dest->isFloatingPointTy() && "Non-floating-point destination Type.");

  APFloat apmin(gop->min);
  APFloat apmax(gop->max);
  // Convert with most conservative rounding mode
  bool losesInfo;
  apmin.convert(dest->getFltSemantics(), APFloatBase::rmTowardNegative, &losesInfo);
  apmax.convert(dest->getFltSemantics(), APFloatBase::rmTowardPositive, &losesInfo);

  // Convert back to double
  apmin.convert(APFloat::IEEEdouble(), APFloatBase::rmTowardNegative, &losesInfo);
  apmax.convert(APFloat::IEEEdouble(), APFloatBase::rmTowardPositive, &losesInfo);
  return std::make_shared<Range>(apmin.convertToDouble(), apmax.convertToDouble());
}

/** boolean Xor instruction */
std::shared_ptr<Range> taffo::handleBooleanXor(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return getGenericBoolRange();
  if (!op1->cross() && !op2->cross())
    return getAlwaysFalse();
  if (op1->isConstant() && op2->isConstant())
    return getAlwaysFalse();
  return getGenericBoolRange();
}

/** boolean And instruction */
std::shared_ptr<Range> taffo::handleBooleanAnd(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return getGenericBoolRange();
  if (!op1->cross() && !op2->cross())
    return getAlwaysTrue();
  if (op1->isConstant() && op2->isConstant())
    return getAlwaysFalse();
  return getGenericBoolRange();
}

/** boolean Or instruction */
std::shared_ptr<Range> taffo::handleBooleanOr(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1 || !op2)
    return getGenericBoolRange();
  if (!op1->cross() || !op2->cross())
    return getAlwaysTrue();
  if (op1->isConstant() && op2->isConstant())
    return getAlwaysFalse();
  return getGenericBoolRange();
}

/** deep copy of range */
std::shared_ptr<ValueInfoWithRange> taffo::copyRange(const std::shared_ptr<ValueInfoWithRange> op) {
  if (!op)
    return nullptr;

  if (const std::shared_ptr<ScalarInfo> op_s = std::dynamic_ptr_cast<ScalarInfo>(op))
    return std::static_ptr_cast<ValueInfoWithRange>(op_s->clone());

  const std::shared_ptr<StructInfo> op_s = std::static_ptr_cast<StructInfo>(op);
  SmallVector<std::shared_ptr<ValueInfo>, 4> new_fields;
  unsigned num_fields = op_s->getNumFields();
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; i++) {
    if (std::shared_ptr<ValueInfo> field = op_s->getField(i))
      if (std::shared_ptr<PointerInfo> ptr_field = std::dynamic_ptr_cast_or_null<PointerInfo>(field))
        new_fields.push_back(std::make_shared<PointerInfo>(ptr_field->getPointed()));
      else
        new_fields.push_back(copyRange(std::static_ptr_cast<ValueInfoWithRange>(field)));
    else
      new_fields.push_back(nullptr);
  }
  return std::make_shared<StructInfo>(new_fields);
}

std::shared_ptr<Range> taffo::copyRange(const std::shared_ptr<Range> op) {
  if (!op)
    return nullptr;
  return std::static_ptr_cast<Range>(op->clone());
}

/** create a generic boolean range */
std::shared_ptr<Range> taffo::getGenericBoolRange() {
  std::shared_ptr<Range> res = std::make_shared<Range>(static_cast<double>(0), static_cast<double>(1));
  return res;
}

/** create a always false boolean range */
std::shared_ptr<Range> taffo::getAlwaysFalse() {
  std::shared_ptr<Range> res = std::make_shared<Range>(static_cast<double>(0), static_cast<double>(0));
  return res;
}

/** create a always false boolean range */
std::shared_ptr<Range> taffo::getAlwaysTrue() {
  std::shared_ptr<Range> res = std::make_shared<Range>(static_cast<double>(1), static_cast<double>(1));
  return res;
}

/** create a union between ranges */
std::shared_ptr<Range> taffo::getUnionRange(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2) {
  if (!op1)
    return copyRange(op2);
  if (!op2)
    return copyRange(op1);
  const double min = std::min({op1->min, op2->min});
  const double max = std::max({op1->max, op2->max});
  return std::make_shared<Range>(min, max);
}

std::shared_ptr<ValueInfoWithRange> taffo::getUnionRange(const std::shared_ptr<ValueInfoWithRange> op1,
                                                         const std::shared_ptr<ValueInfoWithRange> op2) {
  if (!op1)
    return copyRange(op2);
  if (!op2)
    return copyRange(op1);

  if (const std::shared_ptr<ScalarInfo> sop1 = std::dynamic_ptr_cast<ScalarInfo>(op1)) {
    const std::shared_ptr<ScalarInfo> sop2 = std::static_ptr_cast<ScalarInfo>(op2);
    return std::make_shared<ScalarInfo>(nullptr, getUnionRange(sop1->range, sop2->range));
  }

  const std::shared_ptr<StructInfo> op1_s = std::static_ptr_cast<StructInfo>(op1);
  const std::shared_ptr<StructInfo> op2_s = std::static_ptr_cast<StructInfo>(op2);
  unsigned num_fields = std::max(op1_s->getNumFields(), op2_s->getNumFields());
  SmallVector<std::shared_ptr<ValueInfo>, 4U> new_fields;
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; ++i) {
    const std::shared_ptr<ValueInfo> op1_f = op1_s->getField(i);
    if (op1_f && std::isa_ptr<PointerInfo>(op1_f)) {
      new_fields.push_back(op1_f);
    }
    else {
      new_fields.push_back(getUnionRange(std::static_ptr_cast<ValueInfoWithRange>(op1_f),
                                         std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(op2_s->getField(i))));
    }
  }
  return std::make_shared<StructInfo>(new_fields);
}

std::shared_ptr<ValueInfoWithRange> taffo::fillRangeHoles(const std::shared_ptr<ValueInfoWithRange>& src,
                                                          const std::shared_ptr<ValueInfoWithRange>& dst) {
  if (!src)
    return copyRange(dst);
  if (!dst || std::isa_ptr<ScalarInfo>(src))
    return copyRange(src);
  const std::shared_ptr<StructInfo> src_s = std::static_ptr_cast<StructInfo>(src);
  const std::shared_ptr<StructInfo> dst_s = std::static_ptr_cast<StructInfo>(dst);
  SmallVector<std::shared_ptr<ValueInfo>, 4U> new_fields;
  unsigned num_fields = src_s->getNumFields();
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; ++i) {
    if (const std::shared_ptr<PointerInfo> ptr_field = std::dynamic_ptr_cast_or_null<PointerInfo>(src_s->getField(i))) {
      new_fields.push_back(std::make_shared<PointerInfo>(ptr_field->getPointed()));
    }
    else if (i < dst_s->getNumFields()) {
      new_fields.push_back(fillRangeHoles(std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(src_s->getField(i)),
                                          std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(dst_s->getField(i))));
    }
  }
  return std::make_shared<StructInfo>(new_fields);
}
