#include "RangeOperationsCallWhitelist.hpp"
#include "RangeOperations.hpp"
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

static std::shared_ptr<Range>
handleCallToCeil(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function ceil");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(ceil(op->Min), ceil(op->Max));
}

static std::shared_ptr<Range>
handleCallToFloor(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function floor");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(floor(op->Min), floor(op->Max));
}

static std::shared_ptr<Range>
handleCallToFabs(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function fabs");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  double min = fabs(op->Min);
  double max = fabs(op->Max);
  if (min <= max) {
    return std::make_shared<Range>(min, max);
  }
  return std::make_shared<Range>(max, min);
}

static std::shared_ptr<Range>
handleCallToLog(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Log");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  if (op->Max < 0.0) {
    return std::make_shared<Range>(std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::quiet_NaN());
  }
  double min = op->Min < 0 ? std::numeric_limits<double>::epsilon() : op->Min;
  min = log(min);
  double max = log(op->Max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range>
handleCallToLog10(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Log10");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  assert(op->Max >= 0);
  double min = op->Min < 0 ? std::numeric_limits<double>::epsilon() : op->Min;
  min = log10(min);
  double max = log10(op->Max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range>
handleCallToLog2f(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Log2f");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  assert(op->Max >= 0);
  double min = op->Min < 0 ? std::numeric_limits<double>::epsilon() : op->Min;
  min = static_cast<double>(log2f(min));
  double max = log2f(op->Max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range>
handleCallToSqrt(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Sqrt");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  assert(op->Max >= 0);
  double min = op->Min < 0 ? 0 : op->Min;
  min = sqrt(min);
  double max = sqrt(op->Max);
  if (min <= max) {
    return std::make_shared<Range>(min, max);
  }
  return std::make_shared<Range>(max, min);
}

static std::shared_ptr<Range>
handleCallToExp(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Exp");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  double min = exp(op->Min);
  double max = exp(op->Max);
  return std::make_shared<Range>(min, max);
}

static std::shared_ptr<Range>
handleCallToSin(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Sin");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;

  // TODO: better range reduction
  if (op->Min >= -PIO2 && op->Max <= PIO2) {
    return std::make_shared<Range>(std::sin(op->Min), std::sin(op->Max));
  }

  return std::make_shared<Range>(-1.0, 1.0);
}

static std::shared_ptr<Range>
handleCallToCos(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function Cos");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;

  // TODO: better range reduction
  if (op->Min >= -PI && op->Max <= 0.0) {
    return std::make_shared<Range>(std::cos(op->Min), std::cos(op->Max));
  }
  if (op->Min >= 0.0 && op->Max <= PI) {
    return std::make_shared<Range>(std::cos(op->Max), std::cos(op->Min));
  }

  return std::make_shared<Range>(-1.0, 1.0);
}

static std::shared_ptr<Range>
handleCallToAcos(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function acos");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(std::acos(std::max(op->Min, -1.0)),
                    std::acos(std::min(op->Max, 1.0)));
}

static std::shared_ptr<Range>
handleCallToAsin(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function asin");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(std::asin(std::max(op->Min, -1.0)),
                    std::asin(std::min(op->Max, 1.0)));
}

static std::shared_ptr<Range>
handleCallToAtan(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function atan");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  return std::make_shared<Range>(std::atan(op->Min), std::atan(op->Max));
}

static std::shared_ptr<Range>
handleCallToTanh(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 1 && "too many operands in function tanh");
  std::shared_ptr<Range> op = operands.front();
  if (!op)
    return nullptr;
  /* tanh is a monotonic increasing function */
  return std::make_shared<Range>(std::tanh(op->Min), std::tanh(op->Max));
}

static std::shared_ptr<Range>
handleCallToRand(const std::list<std::shared_ptr<Range>> &operands)
{
  // FIXME: RAND_MAX is implementation defined!
  return std::make_shared<Range>(0, RAND_MAX);
}

static std::shared_ptr<Range>
handleCallToFMA(const std::list<std::shared_ptr<Range>> &operands)
{
  assert(operands.size() == 3 && "Wrong number of operands in FMA");
  std::shared_ptr<Range> op1 = operands.front();
  std::shared_ptr<Range> op2 = *(++operands.begin());
  std::shared_ptr<Range> op3 = operands.back();
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
    CMATH_WHITELIST_FUN("atan", &handleCallToAtan),
    CMATH_WHITELIST_FUN("tanh", &handleCallToTanh),
    CMATH_WHITELIST_FUN("rand", &handleCallToRand),
    CMATH_WHITELIST_FUN("fma", &handleCallToFMA),
    INTRINSIC_WHITELIST_FUN("fmuladd", &handleCallToFMA)};
