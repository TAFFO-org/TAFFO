#include "RangeAnalysis/TaffoVRA/RangeOperations.hpp"
#include "RangeAnalysis/TaffoVRA/Range.hpp"
#include <gtest/gtest.h>

namespace
{

using namespace taffo;

// ADD
TEST(RangeOperationsTest, AddPositive)
{
  auto op1 = make_range(2.0, 11.0);
  auto op2 = make_range(10.0, 100.0);
  auto result = handleAdd(op1, op2);
  EXPECT_EQ(result->min(), 12.0);
  EXPECT_EQ(result->max(), 111.0);
}

TEST(RangeOperationsTest, AddNegative)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(-100.0, -1.0);
  auto result = handleAdd(op1, op2);
  EXPECT_EQ(result->min(), -120.0);
  EXPECT_EQ(result->max(), -11.0);
}

TEST(RangeOperationsTest, AddMixed)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(100.0, 110.0);
  auto result = handleAdd(op1, op2);
  EXPECT_EQ(result->min(), 80.0);
  EXPECT_EQ(result->max(), 100.0);
}

// SUB
TEST(RangeOperationsTest, SubPositive)
{
  auto op1 = make_range(2.0, 11.0);
  auto op2 = make_range(10.0, 100.0);
  auto result = handleSub(op1, op2);
  EXPECT_EQ(result->min(), -98.0);
  EXPECT_EQ(result->max(), 1.0);
}

TEST(RangeOperationsTest, SubNegative)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(-100.0, -1.0);
  auto result = handleSub(op1, op2);
  EXPECT_EQ(result->min(), -19.0);
  EXPECT_EQ(result->max(), 90.0);
}

TEST(RangeOperationsTest, SubMixed)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(100.0, 110.0);
  auto result = handleSub(op1, op2);
  EXPECT_EQ(result->min(), -130.0);
  EXPECT_EQ(result->max(), -110.0);
}

// MUL
TEST(RangeOperationsTest, MulPositive)
{
  auto op1 = make_range(2.0, 11.0);
  auto op2 = make_range(10.0, 100.0);
  auto result = handleMul(op1, op2);
  EXPECT_EQ(result->min(), 20.0);
  EXPECT_EQ(result->max(), 1100.0);
}

TEST(RangeOperationsTest, MulNegative)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(-100.0, -1.0);
  auto result = handleMul(op1, op2);
  EXPECT_EQ(result->min(), 10.0);
  EXPECT_EQ(result->max(), 2000.0);
}

TEST(RangeOperationsTest, MulMixed)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(100.0, 110.0);
  auto result = handleMul(op1, op2);
  EXPECT_EQ(result->min(), -2200.0);
  EXPECT_EQ(result->max(), -1000.0);
}

// DIV
TEST(RangeOperationsTest, DivPositive)
{
  auto op1 = make_range(2.0, 11.0);
  auto op2 = make_range(10.0, 100.0);
  auto result = handleDiv(op1, op2);
  EXPECT_EQ(result->min(), 0.02);
  EXPECT_EQ(result->max(), 1.1);
}

TEST(RangeOperationsTest, DivNegative)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(-100.0, -1.0);
  auto result = handleDiv(op1, op2);
  EXPECT_EQ(result->min(), 0.1);
  EXPECT_EQ(result->max(), 20.0);
}

TEST(RangeOperationsTest, DivMixed)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(100.0, 110.0);
  auto result = handleDiv(op1, op2);
  EXPECT_EQ(result->min(), -0.2);
  EXPECT_EQ(result->max(), -10.0 / 110.0);
}

// REM
TEST(RangeOperationsTest, RemPositive)
{
  auto op1 = make_range(2.0, 11.0);
  auto op2 = make_range(10.0, 100.0);
  auto result = handleRem(op1, op2);
  EXPECT_EQ(result->min(), 0.0);
  EXPECT_EQ(result->max(), 11.0);
}

TEST(RangeOperationsTest, RemNegative)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(-100.0, -1.0);
  auto result = handleRem(op1, op2);
  EXPECT_EQ(result->min(), -20.0);
  EXPECT_EQ(result->max(), 0.0);
}

TEST(RangeOperationsTest, RemMixed)
{
  auto op1 = make_range(-20.0, -10.0);
  auto op2 = make_range(100.0, 110.0);
  auto result = handleRem(op1, op2);
  EXPECT_EQ(result->min(), -20.0);
  EXPECT_EQ(result->max(), -10.0);
}

// SHL
TEST(RangeOperationsTest, ShlPositive)
{
  auto op1 = make_range(2.0, 256.0);
  auto op2 = make_range(1.0, 16.0);
  auto result = handleShl(op1, op2);
  EXPECT_EQ(result->min(), 4.0);
  EXPECT_EQ(result->max(), static_cast<double>(256 << 16));
}

// ASHR
TEST(RangeOperationsTest, AShrPositive)
{
  auto op1 = make_range(2.0, 2L << 20);
  auto op2 = make_range(1.0, 16.0);
  auto result = handleAShr(op1, op2);
  EXPECT_EQ(result->min(), 0.0);
  EXPECT_EQ(result->max(), 2L << 19);
}

TEST(RangeOperationsTest, AShrNegative)
{
  auto op1 = make_range(-(2L << 20), -2.0);
  auto op2 = make_range(1.0, 16.0);
  auto result = handleAShr(op1, op2);
  EXPECT_EQ(result->min(), static_cast<double>((-(2L << 20)) >> 1));
  EXPECT_EQ(result->max(), -1.0);
}

TEST(RangeOperationsTest, AShrMixed)
{
  auto op1 = make_range(-2.0, (2L << 20));
  auto op2 = make_range(1.0, 16.0);
  auto result = handleAShr(op1, op2);
  EXPECT_EQ(result->min(), -1.0);
  EXPECT_EQ(result->max(), 2L << 19);
}

}; // namespace
