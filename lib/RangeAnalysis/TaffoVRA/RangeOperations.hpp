#ifndef TAFFO_RANGE_OPERATIONS_HPP
#define TAFFO_RANGE_OPERATIONS_HPP

#include <list>
#include <string>

#include "TaffoInfo/ValueInfo.hpp"
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>

#define DEBUG_TYPE "taffo-vra"

namespace taffo {

//-----------------------------------------------------------------------------
// Wrappers
//-----------------------------------------------------------------------------
/** Handle binary instructions */
std::shared_ptr<Range> handleBinaryInstruction(const std::shared_ptr<Range> op1,
                                    const std::shared_ptr<Range> op2,
                                    const unsigned OpCode);

/** Handle unary instructions */
std::shared_ptr<Range> handleUnaryInstruction(const std::shared_ptr<Range> op,
                                   const unsigned OpCode);

/** Handle cast instructions */
std::shared_ptr<Range> handleCastInstruction(const std::shared_ptr<Range> op,
                                  const unsigned OpCode,
                                  const llvm::Type *dest);

/** Return true if this function call can be handled by taffo::handleMathCallInstruction */
bool isMathCallInstruction(const std::string &function);

/** Handle call to known math functions. Return nullptr if unknown */
std::shared_ptr<Range> handleMathCallInstruction(const std::list<std::shared_ptr<Range>> &ops,
                                      const std::string &function);

std::shared_ptr<Range> handleCompare(const std::list<std::shared_ptr<Range>> &ops,
                          const llvm::CmpInst::Predicate pred);

//-----------------------------------------------------------------------------
// Arithmetic
//-----------------------------------------------------------------------------
/** operator+ */
std::shared_ptr<Range> handleAdd(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** operator- */
std::shared_ptr<Range> handleSub(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** operator* */
std::shared_ptr<Range> handleMul(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** operator/ */
std::shared_ptr<Range> handleDiv(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** operator% */
std::shared_ptr<Range> handleRem(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** operator<< */
std::shared_ptr<Range> handleShl(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** operator>> */
std::shared_ptr<Range> handleAShr(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

//-----------------------------------------------------------------------------
// Cast
//-----------------------------------------------------------------------------
/** Trunc */
std::shared_ptr<Range> handleTrunc(const std::shared_ptr<Range> gop, const llvm::Type *dest);

/** Cast To Unsigned Integer */
std::shared_ptr<Range> handleCastToUI(const std::shared_ptr<Range> op);

/** Cast To Signed Integer */
std::shared_ptr<Range> handleCastToSI(const std::shared_ptr<Range> op);

/** FPTrunc */
std::shared_ptr<Range> handleFPTrunc(const std::shared_ptr<Range> op, const llvm::Type *dest);

//-----------------------------------------------------------------------------
// Boolean
//-----------------------------------------------------------------------------
/** boolean Xor instruction */
std::shared_ptr<Range> handleBooleanXor(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** boolean And instruction */
std::shared_ptr<Range> handleBooleanAnd(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

/** boolean Or instruction */
std::shared_ptr<Range> handleBooleanOr(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

//-----------------------------------------------------------------------------
// Range helpers
//-----------------------------------------------------------------------------
/** deep copy of range */
std::shared_ptr<ValueInfoWithRange> copyRange(const std::shared_ptr<ValueInfoWithRange> op);
std::shared_ptr<Range> copyRange(const std::shared_ptr<Range> op);

/** create a generic boolean range */
std::shared_ptr<Range> getGenericBoolRange();

/** create a always false boolean range */
std::shared_ptr<Range> getAlwaysFalse();

/** create a always false boolean range */
std::shared_ptr<Range> getAlwaysTrue();

std::shared_ptr<Range> getUnionRange(const std::shared_ptr<Range> op1, const std::shared_ptr<Range> op2);

std::shared_ptr<ValueInfoWithRange> getUnionRange(const std::shared_ptr<ValueInfoWithRange> op1, const std::shared_ptr<ValueInfoWithRange> op2);

std::shared_ptr<ValueInfoWithRange> fillRangeHoles(const std::shared_ptr<ValueInfoWithRange> src, const std::shared_ptr<ValueInfoWithRange> dst);

} // namespace taffo

#undef DEBUG_TYPE

#endif /* end of include guard: TAFFO_RANGE_OPERATIONS_HPP */
