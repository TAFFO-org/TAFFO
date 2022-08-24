#include "TaffoVRA/RangeOperations.hpp"
#include "TaffoVRA/Range.hpp"
#include "gtest/gtest.h"

namespace
{

using namespace taffo;


class RangeOperationsTest : public testing::Test
{
protected:
  range_ptr_t op1;
  range_ptr_t op2;
  range_ptr_t result;
};

// ADD
TEST_F(RangeOperationsTest, AddPositive)
{
  op1 = make_range(2.0, 11.0);
  op2 = make_range(10.0, 100.0);
  result = handleAdd(op1, op2);
  EXPECT_EQ(result->min(), 12.0);
  EXPECT_EQ(result->max(), 111.0);
}

TEST_F(RangeOperationsTest, AddNegative)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(-100.0, -1.0);
  result = handleAdd(op1, op2);
  EXPECT_EQ(result->min(), -120.0);
  EXPECT_EQ(result->max(), -11.0);
}

TEST_F(RangeOperationsTest, AddMixed)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(100.0, 110.0);
  result = handleAdd(op1, op2);
  EXPECT_EQ(result->min(), 80.0);
  EXPECT_EQ(result->max(), 100.0);
}

// SUB
TEST_F(RangeOperationsTest, SubPositive)
{
  op1 = make_range(2.0, 11.0);
  op2 = make_range(10.0, 100.0);
  result = handleSub(op1, op2);
  EXPECT_EQ(result->min(), -98.0);
  EXPECT_EQ(result->max(), 1.0);
}

TEST_F(RangeOperationsTest, SubNegative)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(-100.0, -1.0);
  result = handleSub(op1, op2);
  EXPECT_EQ(result->min(), -19.0);
  EXPECT_EQ(result->max(), 90.0);
}

TEST_F(RangeOperationsTest, SubMixed)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(100.0, 110.0);
  result = handleSub(op1, op2);
  EXPECT_EQ(result->min(), -130.0);
  EXPECT_EQ(result->max(), -110.0);
}

// MUL
TEST_F(RangeOperationsTest, MulPositive)
{
  op1 = make_range(2.0, 11.0);
  op2 = make_range(10.0, 100.0);
  result = handleMul(op1, op2);
  EXPECT_EQ(result->min(), 20.0);
  EXPECT_EQ(result->max(), 1100.0);
}

TEST_F(RangeOperationsTest, MulNegative)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(-100.0, -1.0);
  result = handleMul(op1, op2);
  EXPECT_EQ(result->min(), 10.0);
  EXPECT_EQ(result->max(), 2000.0);
}

TEST_F(RangeOperationsTest, MulMixed)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(100.0, 110.0);
  result = handleMul(op1, op2);
  EXPECT_EQ(result->min(), -2200.0);
  EXPECT_EQ(result->max(), -1000.0);
}

// MUL square
TEST_F(RangeOperationsTest, MulSameOpPositive)
{
  op1 = make_range(2.0, 11.0);
  result = handleMul(op1, op1);
  EXPECT_EQ(result->min(), 4.0);
  EXPECT_EQ(result->max(), 121.0);
}

TEST_F(RangeOperationsTest, MulSameOpNegative)
{
  op1 = make_range(-20.0, -10.0);
  result = handleMul(op1, op1);
  EXPECT_EQ(result->min(), 100.0);
  EXPECT_EQ(result->max(), 400.0);
}

// DIV
TEST_F(RangeOperationsTest, DivPositive)
{
  op1 = make_range(2.0, 11.0);
  op2 = make_range(10.0, 100.0);
  result = handleDiv(op1, op2);
  EXPECT_EQ(result->min(), 0.02);
  EXPECT_EQ(result->max(), 1.1);
}

TEST_F(RangeOperationsTest, DivNegative)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(-100.0, -1.0);
  result = handleDiv(op1, op2);
  EXPECT_EQ(result->min(), 0.1);
  EXPECT_EQ(result->max(), 20.0);
}

TEST_F(RangeOperationsTest, DivMixed)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(100.0, 110.0);
  result = handleDiv(op1, op2);
  EXPECT_EQ(result->min(), -0.2);
  EXPECT_EQ(result->max(), -10.0 / 110.0);
}

TEST_F(RangeOperationsTest, DivMaxPosMinNeg)
{
  op1 = make_range(10.0, 20.0);
  op2 = make_range(-1.0, 100.0);
  result = handleDiv(op1, op2);
  EXPECT_EQ(result->min(), -20); // TODO: fix implementation
  EXPECT_EQ(result->max(), 0.2); // TODO: fix implementation
}

// REM
TEST_F(RangeOperationsTest, RemPositive)
{
  op1 = make_range(2.0, 11.0);
  op2 = make_range(10.0, 100.0);
  result = handleRem(op1, op2);
  EXPECT_EQ(result->min(), 0.0);
  EXPECT_EQ(result->max(), 11.0);
}

TEST_F(RangeOperationsTest, RemNegative)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(-100.0, -1.0);
  result = handleRem(op1, op2);
  EXPECT_EQ(result->min(), -20.0);
  EXPECT_EQ(result->max(), 0.0);
}

TEST_F(RangeOperationsTest, RemMixed)
{
  op1 = make_range(-20.0, -10.0);
  op2 = make_range(100.0, 110.0);
  result = handleRem(op1, op2);
  EXPECT_EQ(result->min(), -20.0);
  EXPECT_EQ(result->max(), -10.0);
}

// SHL
TEST_F(RangeOperationsTest, ShlPositive)
{
  op1 = make_range(2.0, 256.0);
  op2 = make_range(1.0, 16.0);
  result = handleShl(op1, op2);
  EXPECT_EQ(result->min(), 4.0);
  EXPECT_EQ(result->max(), static_cast<double>(256 << 16));
}

// ASHR
TEST_F(RangeOperationsTest, AShrPositive)
{
  op1 = make_range(2.0, 2L << 20);
  op2 = make_range(1.0, 16.0);
  result = handleAShr(op1, op2);
  EXPECT_EQ(result->min(), 0.0);
  EXPECT_EQ(result->max(), 2L << 19);
}

TEST_F(RangeOperationsTest, AShrNegative)
{
  op1 = make_range(-(2L << 20), -2.0);
  op2 = make_range(1.0, 16.0);
  result = handleAShr(op1, op2);
  EXPECT_EQ(result->min(), static_cast<double>((-(2L << 20)) >> 1));
  EXPECT_EQ(result->max(), -1.0);
}

TEST_F(RangeOperationsTest, AShrMixed)
{
  op1 = make_range(-2.0, (2L << 20));
  op2 = make_range(1.0, 16.0);
  result = handleAShr(op1, op2);
  EXPECT_EQ(result->min(), -1.0);
  EXPECT_EQ(result->max(), 2L << 19);
}

// Truncate - lose info about decimal digits
TEST_F(RangeOperationsTest, Trunc) // TODO: check if truncation should apply also to signed values
{
  op1 = make_range(2.718, 10.3256);
  result = handleTrunc(op1, llvm::Type::getInt32Ty(Context));
    EXPECT_EQ(result->min(), 2);
    EXPECT_EQ(result->max(), 10); // FIXME: this should be correct but the test fails,  check implementation
}

TEST_F(RangeOperationsTest, FPTrunc) {
  double Dbound = 1.0000000000000002; // smallest double < 1
  float Fbound = 1.0000001192092896; // smallest float < 1
    op1 = make_range(Dbound, Dbound);
    result = handleFPTrunc(op1, llvm::Type::getFloatTy(Context));
    EXPECT_DOUBLE_EQ(result->min(), 1); // conservative bound
    EXPECT_DOUBLE_EQ(result->max(), Fbound);
}

// Cast - lose info about decimal digits
TEST_F(RangeOperationsTest, CastToUI) {
  op1 = make_range(2.4345, 10.56);
  result = handleCastToUI(op1);
  EXPECT_EQ(result->min(), 2);
    EXPECT_EQ(result->max(), 10);
}

TEST_F(RangeOperationsTest, CastToSI)
{
  op1 = make_range(-2.4345, 10.56);
  result = handleCastToSI(op1);
  EXPECT_EQ(result->min(), -2);
  EXPECT_EQ(result->max(), 10);
}
}; // namespace
