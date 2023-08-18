#include "RangeOperationsCallWhitelist.hpp"
#include "RangeOperations.hpp"
#include "Range.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <list>
#include <string>

#define DEBUG_TYPE "taffo-vra"

using namespace taffo;

#define PI 0x1.921FB54442D18p+1
#define PIO2 0x1.921FB54442D18p+0

static range_ptr_t
handleCallToCeil(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function ceil");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  return make_range(static_cast<num_t>(ceil(static_cast<double>(op->min()))),
                    static_cast<num_t>(ceil(static_cast<double>(op->max()))));
}

static range_ptr_t
handleCallToFloor(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function floor");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  return make_range(static_cast<num_t>(floor(static_cast<double>(op->min()))),
                    static_cast<num_t>(floor(static_cast<double>(op->max()))));
}

static range_ptr_t
handleCallToFabs(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function fabs");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  num_t min = static_cast<num_t>(fabs(static_cast<double>(op->min())));
  num_t max = static_cast<num_t>(fabs(static_cast<double>(op->max())));
  if (min <= max) {
    return make_range(min, max);
  }
  return make_range(max, min);
}

static range_ptr_t
handleCallToLog(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Log");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  if (op->max() < 0.0) {
    return make_range(std::numeric_limits<num_t>::quiet_NaN(),
                      std::numeric_limits<num_t>::quiet_NaN());
  }
  num_t min = (op->min() < 0) ? std::numeric_limits<num_t>::epsilon() : op->min();
  min = static_cast<num_t>(log(static_cast<double>(min)));
  num_t max = static_cast<num_t>(log(static_cast<double>(op->max())));
  return make_range(min, max);
}

static range_ptr_t
handleCallToLog10(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Log10");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  assert(op->max() >= 0);
  num_t min = (op->min() < 0) ? std::numeric_limits<num_t>::epsilon() : op->min();
  min = static_cast<num_t>(log10(static_cast<double>(min)));
  num_t max = static_cast<num_t>(log10(static_cast<double>(op->max())));
  return make_range(min, max);
}

static range_ptr_t
handleCallToLog2f(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Log2f");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  assert(op->max() >= 0);
  num_t min = (op->min() < 0) ? std::numeric_limits<num_t>::epsilon() : op->min();
  min = static_cast<num_t>(log2f(static_cast<double>(min)));
  num_t max = static_cast<num_t>(log2f(static_cast<double>(op->max())));
  return make_range(min, max);
}

static range_ptr_t
handleCallToSqrt(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Sqrt");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  assert(op->max() >= 0);
  num_t min = (op->min() < 0) ? 0 : op->min();
  min = static_cast<num_t>(sqrt(static_cast<double>(min)));
  num_t max = static_cast<num_t>(sqrt(static_cast<double>(op->max())));
  if (min <= max) {
    return make_range(min, max);
  }
  return make_range(max, min);
}

static range_ptr_t
handleCallToExp(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Exp");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  num_t min = static_cast<num_t>(exp(static_cast<double>(op->min())));
  num_t max = static_cast<num_t>(exp(static_cast<double>(op->max())));
  return make_range(min, max);
}

static range_ptr_t
handleCallToSin(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Sin");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }

  // TODO: better range reduction
  if (op->min() >= -PIO2 && op->max() <= PIO2) {
    return make_range(std::sin(op->min()), std::sin(op->max()));
  }

  return make_range(-1.0, 1.0);
}

static range_ptr_t
handleCallToCos(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Cos");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }

  // TODO: better range reduction
  if (op->min() >= -PI && op->max() <= 0.0) {
    return make_range(std::cos(op->min()), std::cos(op->max()));
  }
  if (op->min() >= 0.0 && op->max() <= PI) {
    return make_range(std::cos(op->max()), std::cos(op->min()));
  }

  return make_range(-1.0, 1.0);
}

static range_ptr_t
handleCallToAcos(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function acos");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  return make_range(std::acos(std::max(op->min(), -1.0)),
                    std::acos(std::min(op->max(), 1.0)));
}

static range_ptr_t
handleCallToAsin(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function asin");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  return make_range(std::asin(std::max(op->min(), -1.0)),
                    std::asin(std::min(op->max(), 1.0)));
}

static range_ptr_t
handleCallToTanh(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 1 && "too many operands in function tanh");
  range_ptr_t op = operands.front();
  if (!op) {
    return nullptr;
  }
  /* tanh is a monotonic increasing function */
  return make_range(std::tanh(op->min()), std::tanh(op->max()));
}

static range_ptr_t
handleCallToRand(const std::list<range_ptr_t> &operands)
{
  // FIXME: RAND_MAX is implementation defined!
  return make_range(0, RAND_MAX);
}

static range_ptr_t
handleCallToFMA(const std::list<range_ptr_t> &operands)
{
  assert(operands.size() == 3 && "Wrong number of operands in FMA");
  range_ptr_t op1 = operands.front();
  range_ptr_t op2 = *(++operands.begin());
  range_ptr_t op3 = operands.back();
  if (!op1 || !op2 || !op3)
    return nullptr;
  return handleAdd(handleMul(op1, op2), op3);
}

const std::map<const std::string, map_value_t> taffo::functionWhiteList = {
    CMATH_WHITELIST_FUN("ceil", &handleCallToCeil),
    CMATH_WHITELIST_FUN("floor", &handleCallToFloor),
    CMATH_WHITELIST_FUN("fabs", &handleCallToFabs),
    CMATH_WHITELIST_FUN("log", &handleCallToLog),
    CMATH_WHITELIST_FUN("log10", &handleCallToLog10),
    CMATH_WHITELIST_FUN("log2", &handleCallToLog2f),
    CMATH_WHITELIST_FUN("sqrt", &handleCallToSqrt),
    CMATH_WHITELIST_FUN("exp", &handleCallToExp),
    CMATH_WHITELIST_FUN("sin", &handleCallToSin),
    CMATH_WHITELIST_FUN("cos", &handleCallToCos),
    CMATH_WHITELIST_FUN("acos", &handleCallToAcos),
    CMATH_WHITELIST_FUN("asin", &handleCallToAsin),
    CMATH_WHITELIST_FUN("tanh", &handleCallToTanh),
    CMATH_WHITELIST_FUN("rand", &handleCallToRand),
    CMATH_WHITELIST_FUN("fma", &handleCallToFMA),
    INTRINSIC_WHITELIST_FUN("fmuladd", &handleCallToFMA)};
