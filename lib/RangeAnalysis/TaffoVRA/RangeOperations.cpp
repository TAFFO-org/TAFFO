#include "RangeOperations.hpp"
#include "RangeOperationsCallWhitelist.hpp"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <assert.h>
#include <limits>
#include <map>

#define DEBUG_TYPE "taffo-vra"

using namespace taffo;

//-----------------------------------------------------------------------------
// Wrappers
//-----------------------------------------------------------------------------

/** Handle binary instructions */
range_ptr_t
taffo::handleBinaryInstruction(const range_ptr_t op1,
                               const range_ptr_t op2,
                               const unsigned OpCode)
{
  switch (OpCode) {
  case llvm::Instruction::Add:
  case llvm::Instruction::FAdd:
    return handleAdd(op1, op2);
    break;
  case llvm::Instruction::Sub:
  case llvm::Instruction::FSub:
    return handleSub(op1, op2);
    break;
  case llvm::Instruction::Mul:
  case llvm::Instruction::FMul:
    return handleMul(op1, op2);
    break;
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::FDiv:
    return handleDiv(op1, op2);
    break;
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
  case llvm::Instruction::FRem:
    return handleRem(op1, op2);
    break;
  case llvm::Instruction::Shl:
    return handleShl(op1, op2);
  case llvm::Instruction::LShr: // TODO implement
  case llvm::Instruction::AShr:
    return handleAShr(op1, op2);
  case llvm::Instruction::And: // TODO implement
  case llvm::Instruction::Or:  // TODO implement
  case llvm::Instruction::Xor: // TODO implement
    break;
  default:
    assert(false); // unsupported operation
    break;
  }
  return nullptr;
}

range_ptr_t
taffo::handleUnaryInstruction(const range_ptr_t op,
                              const unsigned OpCode)
{
  if (!op)
    return nullptr;

  switch (OpCode) {
  case llvm::Instruction::FNeg:
    return make_range(-op->max(), -op->min());
    break;
  default:
    assert(false); // unsupported operation
    break;
  }
  return nullptr;
}

/** Cast instructions */
range_ptr_t
taffo::handleCastInstruction(const range_ptr_t scalar,
                             const unsigned OpCode,
                             const llvm::Type *dest)
{
  switch (OpCode) {
  case llvm::Instruction::Trunc:
    return handleTrunc(scalar, dest);
    break;
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
    return copyRange(scalar);
    break;
  case llvm::Instruction::FPToUI:
    return handleCastToUI(scalar);
    break;
  case llvm::Instruction::FPToSI:
    return handleCastToSI(scalar);
    break;
  case llvm::Instruction::UIToFP:
  case llvm::Instruction::SIToFP:
    return copyRange(scalar);
    break;
  case llvm::Instruction::FPTrunc:
    return handleFPTrunc(scalar, dest);
  case llvm::Instruction::FPExt:
    return copyRange(scalar);
    break;
  case llvm::Instruction::PtrToInt:
  case llvm::Instruction::IntToPtr:
    return handleCastToSI(scalar);
    break;
  case llvm::Instruction::BitCast: // TODO check
    return copyRange(scalar);
    break;
  case llvm::Instruction::AddrSpaceCast:
    return copyRange(scalar);
    break;
  default:
    assert(false); // unsupported operation
    break;
  }
  return nullptr;
}

//FIXME: better way to handle name.extension
//now we just check the prefix 
/** Return true if this function call can be handled by taffo::handleMathCallInstruction */
bool taffo::isMathCallInstruction(const std::string &function)
{
  return std::any_of(functionWhiteList.cbegin(), functionWhiteList.cend(),
                     [&function](const std::pair<const std::string, map_value_t> &whiteList) { return function.find(whiteList.first) == 0; });
}

/** Handle call to known math functions. Return nullptr if unknown */
range_ptr_t
taffo::handleMathCallInstruction(const std::list<range_ptr_t> &ops,
                                 const std::string &function)
{
  const auto it = std::find_if(functionWhiteList.cbegin(), functionWhiteList.cend(),
                               [&function](const std::pair<const std::string, map_value_t> &whiteList) { return function.find(whiteList.first) == 0; });

  if (it != functionWhiteList.cend()) {
    return it->second(ops);
  }
  return nullptr;
}

/** Handle call to known math functions. Return nullptr if unknown */
range_ptr_t
taffo::handleCompare(const std::list<range_ptr_t> &ops,
                     const llvm::CmpInst::Predicate pred)
{
  switch (pred) {
  case llvm::CmpInst::Predicate::FCMP_FALSE:
    return getAlwaysFalse();
  case llvm::CmpInst::Predicate::FCMP_TRUE:
    return getAlwaysTrue();
  default:
    break;
  }

  // from now on only 2 operators compare
  assert(ops.size() > 1 && "too few operators in compare instruction");
  assert(ops.size() <= 2 && "too many operators in compare instruction");

  // extract values for easy access
  range_ptr_t lhs = ops.front();
  range_ptr_t rhs = ops.back();
  // if unavailable data, nothing can be said
  if (!lhs || !rhs) {
    return getGenericBoolRange();
  }

  // NOTE: not dealing with Ordered / Unordered variants
  switch (pred) {
  case llvm::CmpInst::Predicate::FCMP_OEQ:
  case llvm::CmpInst::Predicate::FCMP_UEQ:
  case llvm::CmpInst::Predicate::ICMP_EQ:
    if (lhs->min() == lhs->max() && rhs->min() == rhs->max() && lhs->min() == rhs->min()) {
      return getAlwaysTrue();
    } else if (lhs->max() < rhs->min() || rhs->max() < lhs->min()) {
      return getAlwaysFalse();
    } else {
      return getGenericBoolRange();
    }
    break;
  case llvm::CmpInst::Predicate::FCMP_OGT:
  case llvm::CmpInst::Predicate::FCMP_UGT:
  case llvm::CmpInst::Predicate::ICMP_UGT:
  case llvm::CmpInst::Predicate::ICMP_SGT:
    if (lhs->min() > rhs->max()) {
      return getAlwaysTrue();
    } else if (lhs->max() <= rhs->min()) {
      return getAlwaysFalse();
    } else {
      return getGenericBoolRange();
    }
    break;
  case llvm::CmpInst::Predicate::FCMP_OGE:
  case llvm::CmpInst::Predicate::FCMP_UGE:
  case llvm::CmpInst::Predicate::ICMP_UGE:
  case llvm::CmpInst::Predicate::ICMP_SGE:
    if (lhs->min() >= rhs->max()) {
      return getAlwaysTrue();
    } else if (lhs->max() < rhs->min()) {
      return getAlwaysFalse();
    } else {
      return getGenericBoolRange();
    }
    break;
  case llvm::CmpInst::Predicate::FCMP_OLT:
  case llvm::CmpInst::Predicate::FCMP_ULT:
  case llvm::CmpInst::Predicate::ICMP_ULT:
  case llvm::CmpInst::Predicate::ICMP_SLT:
    if (lhs->max() < rhs->min()) {
      return getAlwaysTrue();
    } else if (lhs->min() >= rhs->max()) {
      return getAlwaysFalse();
    } else {
      return getGenericBoolRange();
    }
    break;
  case llvm::CmpInst::Predicate::FCMP_OLE:
  case llvm::CmpInst::Predicate::FCMP_ULE:
  case llvm::CmpInst::Predicate::ICMP_ULE:
  case llvm::CmpInst::Predicate::ICMP_SLE:
    if (lhs->max() <= rhs->min()) {
      return getAlwaysTrue();
    } else if (lhs->min() > rhs->max()) {
      return getAlwaysFalse();
    } else {
      return getGenericBoolRange();
    }
    break;
  case llvm::CmpInst::Predicate::FCMP_ONE:
  case llvm::CmpInst::Predicate::FCMP_UNE:
  case llvm::CmpInst::Predicate::ICMP_NE:
    if (lhs->min() == lhs->max() && rhs->min() == rhs->max() && lhs->min() == rhs->min()) {
      return getAlwaysFalse();
    } else if (lhs->max() < rhs->min() || rhs->max() < lhs->min()) {
      return getAlwaysTrue();
    } else {
      return getGenericBoolRange();
    }
    break;
  case llvm::CmpInst::Predicate::FCMP_ORD: // none of the operands is NaN
  case llvm::CmpInst::Predicate::FCMP_UNO: // one of the operand is NaN
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
range_ptr_t
taffo::handleAdd(const range_ptr_t op1, const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return nullptr;
  }
  num_t a = op1->min() + op2->min();
  num_t b = op1->max() + op2->max();
  return make_range(a, b);
}

/** operator- */
range_ptr_t
taffo::handleSub(const range_ptr_t op1, const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return nullptr;
  }
  num_t a = op1->min() - op2->max();
  num_t b = op1->max() - op2->min();
  return make_range(a, b);
}

/** operator* */
range_ptr_t
taffo::handleMul(const range_ptr_t op1, const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return nullptr;
  }
  if (op1 == op2) {
    // This is a square.
    num_t a = op1->min() * op1->min();
    num_t b = op1->max() * op1->max();
    num_t r1 = (op1->min() <= 0.0 && op1->max() >= 0) ? 0.0 : std::min(a, b);
    num_t r2 = std::max(a, b);
    return make_range(r1, r2);
  }
  num_t a = op1->min() * op2->min();
  num_t b = op1->max() * op2->max();
  num_t c = op1->min() * op2->max();
  num_t d = op1->max() * op2->min();
  const num_t r1 = std::min({a, b, c, d});
  const num_t r2 = std::max({a, b, c, d});
  return make_range(r1, r2);
}

/** operator/ */
range_ptr_t
taffo::handleDiv(const range_ptr_t op1, const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return nullptr;
  }
  num_t op2_min, op2_max;
  // Avoid division by 0
#define DIV_EPS (static_cast<num_t>(1e-8))
  if (op2->max() <= 0) {
    op2_min = std::min(op2->min(), -DIV_EPS);
    op2_max = std::min(op2->max(), -DIV_EPS);
  } else if (op2->min() < 0) {
    op2_min = -DIV_EPS;
    op2_max = +DIV_EPS;
  } else {
    op2_min = std::max(op2->min(), +DIV_EPS);
    op2_max = std::max(op2->max(), +DIV_EPS);
  }
  num_t a = op1->min() / op2_min;
  num_t b = op1->max() / op2_max;
  num_t c = op1->min() / op2_max;
  num_t d = op1->max() / op2_min;
  const num_t r1 = std::min({a, b, c, d});
  const num_t r2 = std::max({a, b, c, d});
  return make_range(r1, r2);
}

num_t getRemMin(num_t op1_min, num_t op1_max, num_t op2_min, num_t op2_max);
num_t getRemMax(num_t op1_min, num_t op1_max, num_t op2_min, num_t op2_max);

/** operator% */
range_ptr_t
taffo::handleRem(const range_ptr_t op1, const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return nullptr;
  }
  const num_t min_value = getRemMin(op1->min(), op1->max(), op2->min(), op2->max());
  const num_t max_value = getRemMax(op1->min(), op1->max(), op2->min(), op2->max());
  return make_range(min_value, max_value);
}

num_t getRemMin(num_t op1_min, num_t op1_max, num_t op2_min, num_t op2_max)
{
  // the sign of the second operand does not affect the result, we always mirror negative into positive
  if (op2_max < 0)
    return getRemMin(op1_min, op1_max, -op2_max, -op2_min);
  if (op2_min < 0) {
    // we have to split second operand range into negative and positive parts and calculate min separately
    num_t neg = getRemMin(op1_min, op1_max, 1, -op2_min);
    num_t pos = getRemMin(op1_min, op1_max, 0, op2_max);
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
  } else {
    if (op1_max < 0) {
      // this is the case when % will always return negative result
      // mirror the interval into positives and calculate max with "-" sign as the minimum
      num_t neg = -getRemMax(-op1_max, -op1_min, op2_min, op2_max);
      return neg;
    } else {
      // we need to split the interval into the negative and positive parts
      // first, we take the negative part of the interval [op1_min, -1]
      // we mirror it to [1, -op1_min], which is going to be positive
      // we calculate the max and take it with the "-" sign as the minimum value
      num_t neg = -getRemMax(1.0, -op1_min, op2_min, op2_max);
      // for the positive part we calculate it the standard way
      num_t pos = getRemMin(0.0, op1_max, op2_min, op2_max);
      return std::min(neg, pos);
    }
  }
}

num_t getRemMax(num_t op1_min, num_t op1_max, num_t op2_min, num_t op2_max)
{
  if (op1_min >= 0) {
    // this is the case when % will always return non-negative result
    // the range might include n*op2_max+(op2_max-1) value that will be the max
    if (op1_max >= op2_max)
      return op2_max - 1;
    // op1_max < op2_max, so op1_max % op2_max = op1_max
    return op1_max;
  } else {
    if (op1_max < 0) {
      // this is the case when remainder will always return a negative result, we need to choose the highest max
      // mirror the interval and calculate the min, take it with "-" sign as max
      return -getRemMin(-op1_max, -op1_min, op2_min, op2_max);
    } else {
      // we can ignore the negative part of the interval as it always will be lower than the positive
      return getRemMax(0.0, op1_max, op2_min, op2_max);
    }
  }
}

range_ptr_t
taffo::handleShl(const range_ptr_t op1, const range_ptr_t op2)
{
  // FIXME: it only works if no overflow occurs.
  if (!op1 || !op2) {
    return nullptr;
  }
  const unsigned sh_min = static_cast<unsigned>(op2->min());
  const unsigned sh_max = static_cast<unsigned>(op2->max());
  const long op_min = static_cast<long>(op1->min());
  const long op_max = static_cast<long>(op1->max());
  return make_range(static_cast<num_t>(op_min << ((op_min < 0) ? sh_max : sh_min)),
                    static_cast<num_t>(op_max << ((op_max < 0) ? sh_min : sh_max)));
}

range_ptr_t
taffo::handleAShr(const range_ptr_t op1, const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return nullptr;
  }
  const unsigned sh_min = static_cast<unsigned>(op2->min());
  const unsigned sh_max = static_cast<unsigned>(op2->max());
  const long op_min = static_cast<long>(op1->min());
  const long op_max = static_cast<long>(op1->max());
  return make_range(static_cast<num_t>(op_min >> ((op_min > 0) ? sh_max : sh_min)),
                    static_cast<num_t>(op_max >> ((op_max > 0) ? sh_min : sh_max)));
}

/** Trunc */
range_ptr_t
taffo::handleTrunc(const range_ptr_t op,
                   const llvm::Type *dest)
{
  using namespace llvm;
  if (!op)
    return nullptr;
  const IntegerType *itype = cast<IntegerType>(dest);

  APSInt imin(64U, true), imax(64U, true);
  bool isExact;
  APFloat(op->min()).convertToInteger(imin,
                                      llvm::APFloatBase::roundingMode::TowardNegative,
                                      &isExact);
  APFloat(op->max()).convertToInteger(imax,
                                      llvm::APFloatBase::roundingMode::TowardPositive,
                                      &isExact);
  APSInt new_imin(imin.trunc(itype->getBitWidth()));
  APSInt new_imax(imax.trunc(itype->getBitWidth()));

  return make_range(new_imin.getExtValue(), new_imax.getExtValue());
}

/** CastToUInteger */
range_ptr_t
taffo::handleCastToUI(const range_ptr_t op)
{
  if (!op) {
    return nullptr;
  }
  const num_t r1 = static_cast<num_t>(static_cast<unsigned long>(op->min()));
  const num_t r2 = static_cast<num_t>(static_cast<unsigned long>(op->max()));
  return make_range(r1, r2);
}

/** CastToUInteger */
range_ptr_t
taffo::handleCastToSI(const range_ptr_t op)
{
  if (!op) {
    return nullptr;
  }
  const num_t r1 = static_cast<num_t>(static_cast<long>(op->min()));
  const num_t r2 = static_cast<num_t>(static_cast<long>(op->max()));
  return make_range(r1, r2);
}

/** FPTrunc */
range_ptr_t
taffo::handleFPTrunc(const range_ptr_t gop,
                     const llvm::Type *dest)
{
  if (!gop) {
    return nullptr;
  }
  assert(dest && dest->isFloatingPointTy() && "Non-floating-point destination Type.");

  llvm::APFloat apmin(gop->min());
  llvm::APFloat apmax(gop->max());
  // Convert with most conservative rounding mode
  bool losesInfo;
  apmin.convert(dest->getFltSemantics(),
                llvm::APFloatBase::rmTowardNegative,
                &losesInfo);
  apmax.convert(dest->getFltSemantics(),
                llvm::APFloatBase::rmTowardPositive,
                &losesInfo);

  // Convert back to double
  apmin.convert(llvm::APFloat::IEEEdouble(),
                llvm::APFloatBase::rmTowardNegative,
                &losesInfo);
  apmax.convert(llvm::APFloat::IEEEdouble(),
                llvm::APFloatBase::rmTowardPositive,
                &losesInfo);
  return make_range(apmin.convertToDouble(), apmax.convertToDouble());
}

/** boolean Xor instruction */
range_ptr_t
taffo::handleBooleanXor(const range_ptr_t op1,
                        const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return getGenericBoolRange();
  }
  if (!op1->cross() && !op2->cross()) {
    return getAlwaysFalse();
  }
  if (op1->isConstant() && op2->isConstant()) {
    return getAlwaysFalse();
  }
  return getGenericBoolRange();
}

/** boolean And instruction */
range_ptr_t
taffo::handleBooleanAnd(const range_ptr_t op1,
                        const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return getGenericBoolRange();
  }
  if (!op1->cross() && !op2->cross()) {
    return getAlwaysTrue();
  }
  if (op1->isConstant() && op2->isConstant()) {
    return getAlwaysFalse();
  }
  return getGenericBoolRange();
}

/** boolean Or instruction */
range_ptr_t
taffo::handleBooleanOr(const range_ptr_t op1,
                       const range_ptr_t op2)
{
  if (!op1 || !op2) {
    return getGenericBoolRange();
  }
  if (!op1->cross() || !op2->cross()) {
    return getAlwaysTrue();
  }
  if (op1->isConstant() && op2->isConstant()) {
    return getAlwaysFalse();
  }
  return getGenericBoolRange();
}

/** deep copy of range */
RangeNodePtrT
taffo::copyRange(const RangeNodePtrT op)
{
  if (!op)
    return nullptr;

  if (const std::shared_ptr<VRAScalarNode> op_s =
          std::dynamic_ptr_cast<VRAScalarNode>(op)) {
    return std::make_shared<VRAScalarNode>(copyRange(op_s->getRange()));
  }

  const std::shared_ptr<VRAStructNode> op_s = std::static_ptr_cast<VRAStructNode>(op);
  llvm::SmallVector<NodePtrT, 4U> new_fields;
  unsigned num_fields = op_s->getNumFields();
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; ++i) {
    if (const NodePtrT field = op_s->getNodeAt(i)) {
      if (const std::shared_ptr<VRAPtrNode> ptr_field =
              std::dynamic_ptr_cast_or_null<VRAPtrNode>(field)) {
        new_fields.push_back(std::make_shared<VRAPtrNode>(ptr_field->getParent()));
      } else {
        new_fields.push_back(copyRange(std::static_ptr_cast<VRARangeNode>(field)));
      }
    } else {
      new_fields.push_back(nullptr);
    }
  }
  return std::make_shared<VRAStructNode>(new_fields);
}

range_ptr_t
taffo::copyRange(const range_ptr_t op)
{
  if (!op) {
    return nullptr;
  }
  return std::static_ptr_cast<range_t>(op->clone());
}

/** create a generic boolean range */
range_ptr_t
taffo::getGenericBoolRange()
{
  range_ptr_t res = make_range(static_cast<num_t>(0), static_cast<num_t>(1));
  return res;
}

/** create a always false boolean range */
range_ptr_t
taffo::getAlwaysFalse()
{
  range_ptr_t res = make_range(static_cast<num_t>(0), static_cast<num_t>(0));
  return res;
}

/** create a always false boolean range */
range_ptr_t
taffo::getAlwaysTrue()
{
  range_ptr_t res = make_range(static_cast<num_t>(1), static_cast<num_t>(1));
  return res;
}

/** create a union between ranges */
range_ptr_t
taffo::getUnionRange(const range_ptr_t op1, const range_ptr_t op2)
{
  if (!op1) {
    return copyRange(op2);
  }
  if (!op2) {
    return copyRange(op1);
  }
  const num_t min = std::min({op1->min(), op2->min()});
  const num_t max = std::max({op1->max(), op2->max()});
  return make_range(min, max);
}

RangeNodePtrT
taffo::getUnionRange(const RangeNodePtrT op1,
                     const RangeNodePtrT op2)
{
  if (!op1)
    return copyRange(op2);
  if (!op2)
    return copyRange(op1);

  if (const std::shared_ptr<VRAScalarNode> sop1 =
          std::dynamic_ptr_cast<VRAScalarNode>(op1)) {
    const std::shared_ptr<VRAScalarNode> sop2 =
        std::static_ptr_cast<VRAScalarNode>(op2);
    return std::make_shared<VRAScalarNode>(getUnionRange(sop1->getRange(), sop2->getRange()));
  }

  const std::shared_ptr<VRAStructNode> op1_s = std::static_ptr_cast<VRAStructNode>(op1);
  const std::shared_ptr<VRAStructNode> op2_s = std::static_ptr_cast<VRAStructNode>(op2);
  unsigned num_fields = std::max(op1_s->getNumFields(), op2_s->getNumFields());
  llvm::SmallVector<NodePtrT, 4U> new_fields;
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; ++i) {
    const NodePtrT op1_f = op1_s->getNodeAt(i);
    if (op1_f && std::isa_ptr<VRAPtrNode>(op1_f)) {
      new_fields.push_back(op1_f);
    } else {
      new_fields.push_back(getUnionRange(std::static_ptr_cast<VRARangeNode>(op1_f),
                                         std::dynamic_ptr_cast_or_null<VRARangeNode>(op2_s->getNodeAt(i))));
    }
  }
  return std::make_shared<VRAStructNode>(new_fields);
}

RangeNodePtrT
taffo::fillRangeHoles(const RangeNodePtrT src,
                      const RangeNodePtrT dst)
{
  if (!src)
    return copyRange(dst);
  if (!dst || std::isa_ptr<VRAScalarNode>(src)) {
    return copyRange(src);
  }
  const std::shared_ptr<VRAStructNode> src_s = std::static_ptr_cast<VRAStructNode>(src);
  const std::shared_ptr<VRAStructNode> dst_s = std::static_ptr_cast<VRAStructNode>(dst);
  llvm::SmallVector<NodePtrT, 4U> new_fields;
  unsigned num_fields = src_s->getNumFields();
  new_fields.reserve(num_fields);
  for (unsigned i = 0; i < num_fields; ++i) {
    if (const std::shared_ptr<VRAPtrNode> ptr_field =
            std::dynamic_ptr_cast_or_null<VRAPtrNode>(src_s->getNodeAt(i))) {
      new_fields.push_back(std::make_shared<VRAPtrNode>(ptr_field->getParent()));
    } else if (i < dst_s->getNumFields()) {
      new_fields.push_back(fillRangeHoles(std::dynamic_ptr_cast_or_null<VRARangeNode>(src_s->getNodeAt(i)),
                                          std::dynamic_ptr_cast_or_null<VRARangeNode>(dst_s->getNodeAt(i))));
    }
  }
  return std::make_shared<VRAStructNode>(new_fields);
}
