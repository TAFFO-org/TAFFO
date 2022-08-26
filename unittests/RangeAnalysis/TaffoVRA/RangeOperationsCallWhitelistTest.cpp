#include "TaffoVRA/RangeOperationsCallWhitelist.hpp"
#include "gtest/gtest.h"

namespace
{

using namespace taffo;


class RangeOperationsCallWhitelistTest : public testing::Test
{
protected:
  std::list<range_ptr_t> operands;
  range_ptr_t result;

  RangeOperationsCallWhitelistTest()
  {
    operands.clear();
    result = nullptr;
  }
};

TEST_F(RangeOperationsCallWhitelistTest, handleCeil)
{
  operands.push_back(make_range(-0.5, 0.5));
  result = functionWhiteList.find("ceil")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), 0);
  EXPECT_DOUBLE_EQ(result->max(), 1);
}

TEST_F(RangeOperationsCallWhitelistTest, handleFloor)
{
  operands.push_back(make_range(-0.5, 0.5));
  result = functionWhiteList.find("floor")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), -1);
  EXPECT_DOUBLE_EQ(result->max(), 0);
}

TEST_F(RangeOperationsCallWhitelistTest, handleFabs)
{
  operands.push_back(make_range(-0.5, 0.6));
  result = functionWhiteList.find("fabs")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), 0.5);
  EXPECT_DOUBLE_EQ(result->max(), 0.6);
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog)
{
  // average use
  operands.push_back(make_range(0.5, 4));
  result = functionWhiteList.find("log")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), log(0.5));
  EXPECT_DOUBLE_EQ(result->max(), log(4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog_negativeMax)
{
  // TODO: check if this behavior is to be expected also from log10 and log2f
  operands.push_back(make_range(0.5, -4));
  result = functionWhiteList.find("log")->second(operands);
  EXPECT_TRUE(isnanf(result->min()));
  EXPECT_TRUE(isnanf(result->max()));
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog_negativeMin)
{
  operands.push_back(make_range(-0.5, 4));
  result = functionWhiteList.find("log")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), log(std::numeric_limits<num_t>::epsilon()));
  EXPECT_DOUBLE_EQ(result->max(), log(4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog_minGTmax)
{
  operands.push_back(make_range(4, 0.5));
  result = functionWhiteList.find("log")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), log(0.5));
  EXPECT_DOUBLE_EQ(result->max(), log(4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog10)
{
  // all positive
  operands.push_back(make_range(0.1, 100));
  result = functionWhiteList.find("log10")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), -1);
  EXPECT_DOUBLE_EQ(result->max(), 2);
}
TEST_F(RangeOperationsCallWhitelistTest, handleLog10_negativeMin)
{
  operands.push_back(make_range(-0.1, 100));
  result = functionWhiteList.find("log10")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), log10(std::numeric_limits<num_t>::epsilon()));
  EXPECT_DOUBLE_EQ(result->max(), 2);
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog10_minGTmax)
{

  operands.push_back(make_range(100, 0.1));
  result = functionWhiteList.find("log10")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), -1);
  EXPECT_DOUBLE_EQ(result->max(), 2);
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog2f)
{
  // all positive
  operands.push_back(make_range(0.5, 4));
  result = functionWhiteList.find("log2f")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), -1);
  EXPECT_DOUBLE_EQ(result->max(), 2);
}

TEST_F(RangeOperationsCallWhitelistTest, handleLog2f_negativeMin)
{

  operands.push_back(make_range(-0.5, 4));
  result = functionWhiteList.find("log2")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), log2(std::numeric_limits<num_t>::epsilon()));
  EXPECT_DOUBLE_EQ(result->max(), 2);
}
TEST_F(RangeOperationsCallWhitelistTest, handleLog2f_minGTmax)
{
  operands.push_back(make_range(4, 0.5));
  result = functionWhiteList.find("log2")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), -1);
  EXPECT_DOUBLE_EQ(result->max(), 2);
}

TEST_F(RangeOperationsCallWhitelistTest, handleSqrt)
{
  operands.push_back(make_range(0.5, 4));
  result = functionWhiteList.find("sqrt")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), sqrt(0.5));
  EXPECT_DOUBLE_EQ(result->max(), 2);
}

TEST_F(RangeOperationsCallWhitelistTest, handleSqrt_negativeMin)
{
  operands.clear();
  operands.push_back(make_range(-0.5, 4));
  result = functionWhiteList.find("sqrt")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), 0);
  EXPECT_DOUBLE_EQ(result->max(), 2);
}

TEST_F(RangeOperationsCallWhitelistTest, handleSqrt_minGTmax)
{
  operands.clear();
  operands.push_back(make_range(4, 0.5));
  result = functionWhiteList.find("sqrt")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), sqrt(0.5));
  EXPECT_DOUBLE_EQ(result->max(), 2);
}

TEST_F(RangeOperationsCallWhitelistTest, handleExp)
{
  operands.push_back(make_range(-0.5, 4));
  result = functionWhiteList.find("exp")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), exp(-0.5));
  EXPECT_DOUBLE_EQ(result->max(), exp(4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleExp_minGTmax)
{
  operands.push_back(make_range(4, 0.5));
  result = functionWhiteList.find("exp")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), exp(0.5));
  EXPECT_DOUBLE_EQ(result->max(), exp(4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleSin_PosPos)
{
  // both positive
  operands.push_back(make_range(M_PI_4, M_PI_2 - 0.5));
  result = functionWhiteList.find("sin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), sin(M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), sin(M_PI_2 - 0.5));
}
TEST_F(RangeOperationsCallWhitelistTest, handleSin_PosPos_max1)
{
  operands.push_back(make_range(M_PI_4, 3 * M_PI_4));
  result = functionWhiteList.find("sin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), sin(M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), 1);
}
TEST_F(RangeOperationsCallWhitelistTest, handleSin_NegPos)
{
  operands.push_back(make_range(-M_PI_4, M_PI_4));
  result = functionWhiteList.find("sin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), sin(-M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), sin(M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleSin_PosNeg)
{
  operands.push_back(make_range(3 * M_PI_4, 5 * M_PI_4));
  result = functionWhiteList.find("sin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), sin(5 * M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), sin(3 * M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleSin_NegNeg)
{

  operands.push_back(make_range(5 * M_PI_4, 6 * M_PI_4 - 0.5));
  result = functionWhiteList.find("sin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), sin(6 * M_PI_4 - 0.5));
  EXPECT_DOUBLE_EQ(result->max(), sin(5 * M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleSin_NegNeg_minNeg1)
{
  operands.push_back(make_range(5 * M_PI_4, 7 * M_PI_4));
  result = functionWhiteList.find("sin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), -1);
  EXPECT_DOUBLE_EQ(result->max(), sin(7 * M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleCos_PosPos)
{
  // both positive
  operands.push_back(make_range(-M_PI_4, 0 - 0.5));
  result = functionWhiteList.find("cos")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), cos(-M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), cos(0 - 0.5));
}

TEST_F(RangeOperationsCallWhitelistTest, handleCos_PosPos_max1)
{
  operands.push_back(make_range(-M_PI_4, M_PI_4));
  result = functionWhiteList.find("cos")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), cos(-M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), 1);
}

TEST_F(RangeOperationsCallWhitelistTest, handleCos_NegPos)
{
  operands.push_back(make_range(-3 * M_PI_4, -M_PI_4));
  result = functionWhiteList.find("cos")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), cos(-3 * M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), cos(-M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleCos_PosNeg)
{
  operands.push_back(make_range(M_PI_4, 3 * M_PI_4));
  result = functionWhiteList.find("cos")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), cos(3 * M_PI_4));
  EXPECT_DOUBLE_EQ(result->max(), cos(M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleCos_NegNeg)
{
  operands.push_back(make_range(3 * M_PI_4, 4 * M_PI_4 - 0.5));
  result = functionWhiteList.find("cos")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), cos(4 * M_PI_4 - 0.5));
  EXPECT_DOUBLE_EQ(result->max(), cos(3 * M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleCos_NegNeg_minNeg1)
{
  operands.push_back(make_range(3 * M_PI_4, 5 * M_PI_4));
  result = functionWhiteList.find("cos")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), -1);
  EXPECT_DOUBLE_EQ(result->max(), cos(5 * M_PI_4));
}

TEST_F(RangeOperationsCallWhitelistTest, handleACos)
{
  operands.push_back(make_range(-0.8, 0.8));
  result = functionWhiteList.find("acos")->second(operands);
  // TODO: should these be swapped?
  EXPECT_DOUBLE_EQ(result->min(), acos(-0.8));
  EXPECT_DOUBLE_EQ(result->max(), acos(0.8));
}

TEST_F(RangeOperationsCallWhitelistTest, handleACos_minGTmax)
{
  operands.push_back(make_range(0.8, -0.8));
  result = functionWhiteList.find("acos")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), acos(-0.8));
  EXPECT_DOUBLE_EQ(result->max(), acos(0.8));
}

TEST_F(RangeOperationsCallWhitelistTest, handleACos_outOfRange)
{
  operands.push_back(make_range(-2, 2));
  result = functionWhiteList.find("acos")->second(operands);
  // TODO: should these be swapped?
  EXPECT_DOUBLE_EQ(result->min(), acos(-1));
  EXPECT_DOUBLE_EQ(result->max(), acos(1));
}

TEST_F(RangeOperationsCallWhitelistTest, handleASin)
{
  operands.push_back(make_range(-0.8, 0.8));
  result = functionWhiteList.find("asin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), asin(-0.8));
  EXPECT_DOUBLE_EQ(result->max(), asin(0.8));
}

TEST_F(RangeOperationsCallWhitelistTest, handleASin_minGTmax)
{
  operands.push_back(make_range(0.8, -0.8));
  result = functionWhiteList.find("asin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), asin(-0.8));
  EXPECT_DOUBLE_EQ(result->max(), asin(0.8));
}

TEST_F(RangeOperationsCallWhitelistTest, handleASin_outOfRange)
{
  operands.push_back(make_range(-2, 2));
  result = functionWhiteList.find("asin")->second(operands);
  EXPECT_DOUBLE_EQ(result->min(), asin(-1));
  EXPECT_DOUBLE_EQ(result->max(), asin(1));
}

TEST_F(RangeOperationsCallWhitelistTest, handleTanh)
{
  // TODO: enable once the feature is implemented
  // operands.push_back(make_range(-0.8, 0.8));
  // result = functionWhiteList.find("tanh")->second(operands);
  // EXPECT_DOUBLE_EQ(result->min(), tanh(-0.8));
  // EXPECT_DOUBLE_EQ(result->max(), tanh(0.8));
}

TEST_F(RangeOperationsCallWhitelistTest, handleTanh_minGTmax)
{
  // TODO: enable once the feature is implemented
  // operands.push_back(make_range(0.8, -0.8));
  // result = functionWhiteList.find("tanh")->second(operands);
  // EXPECT_DOUBLE_EQ(result->min(), tanh(-0.8));
  // EXPECT_DOUBLE_EQ(result->max(), tanh(0.8));
}

TEST_F(RangeOperationsCallWhitelistTest, nullOperand)
{
  operands.push_back(nullptr);
  EXPECT_EQ(functionWhiteList.find("ceil")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("floor")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("fabs")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("log")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("log10")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("log2f")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("sqrt")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("exp")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("sin")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("cos")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("acos")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("asin")->second(operands), nullptr);
  EXPECT_EQ(functionWhiteList.find("tanh")->second(operands), nullptr);
}
} // namespace
