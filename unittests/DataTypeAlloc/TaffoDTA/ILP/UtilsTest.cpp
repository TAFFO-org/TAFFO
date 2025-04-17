#include "TaffoDTA/ILP/Utils.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

namespace {

using namespace llvm;
using namespace taffo_test;

class UtilsTest : public taffo_test::Test {};

TEST_F(UtilsTest, uniqueID_instr) {
  auto F = genFunction(*M, "functionName", Type::getVoidTy(Context), {});
  auto BB = BasicBlock::Create(Context, "basicblock", F);
  auto I = BinaryOperator::Create(Instruction::Add,
                                  ConstantInt::get(Type::getInt32Ty(Context), 1),
                                  ConstantInt::get(Type::getInt32Ty(Context), 2),
                                  "instructionName",
                                  BB);

  std::string id = tuner::uniqueIDForValue(I);

  std::string expected = "instr_functionName_instructionName_";
  EXPECT_EQ(id.substr(0, expected.size()), expected);
  std::string rem = id.substr(expected.size());
  unsigned long addr = std::stol(rem);
  EXPECT_EQ(addr, (intptr_t) I);
}

TEST_F(UtilsTest, uniqueID_instrNoName) {
  auto F = genFunction(*M, "functionName", Type::getVoidTy(Context), {});
  auto BB = BasicBlock::Create(Context, "basicblock", F);
  auto I = BinaryOperator::Create(Instruction::Add,
                                  ConstantInt::get(Type::getInt32Ty(Context), 1),
                                  ConstantInt::get(Type::getInt32Ty(Context), 2),
                                  "",
                                  BB);

  std::string id = tuner::uniqueIDForValue(I);

  std::string expected = "instr_functionName__0_"; // instr is %0, % is sanitized into _
  EXPECT_EQ(id.substr(0, expected.size()), expected);
  std::string rem = id.substr(expected.size());
  unsigned long addr = std::stol(rem);
  EXPECT_EQ(addr, (intptr_t) I);
}

TEST_F(UtilsTest, uniqueID_funarg) {
  auto F = genFunction(*M, "functionName", Type::getVoidTy(Context), {Type::getInt32Ty(Context)});
  auto arg = F->args().begin();
  arg->setName("argName");
  std::string id = tuner::uniqueIDForValue(arg);

  std::string expected = "funarg_functionName_argName_";
  EXPECT_EQ(id.substr(0, expected.size()), expected);
  std::string rem = id.substr(expected.size());
  unsigned long addr = std::stol(rem);
  EXPECT_EQ(addr, (intptr_t) arg);
}

TEST_F(UtilsTest, uniqueID_funargNoName) {
  auto F =
    genFunction(*M, "functionName", Type::getVoidTy(Context), {Type::getInt32Ty(Context), Type::getInt32Ty(Context)});
  auto arg = F->args().begin();
  std::string id = tuner::uniqueIDForValue(arg);

  std::string expected = "funarg_functionName__0_"; // first arg is %0
  EXPECT_EQ(id.substr(0, expected.size()), expected);
  std::string rem = id.substr(expected.size());
  unsigned long addr = std::stol(rem);
  EXPECT_EQ(addr, (intptr_t) arg);

  arg++;
  id = tuner::uniqueIDForValue(arg);
  expected = "funarg_functionName__1_"; // second arg is %1
  EXPECT_EQ(id.substr(0, expected.size()), expected);
  rem = id.substr(expected.size());
  addr = std::stol(rem);
  EXPECT_EQ(addr, (intptr_t) arg);
}

TEST_F(UtilsTest, uniqueID_const) {
  auto C = ConstantInt::get(Type::getInt32Ty(Context), 42);
  std::string id = tuner::uniqueIDForValue(C);

  std::string expected = "const_42_";
  EXPECT_EQ(id.substr(0, expected.size()), expected);
  std::string rem = id.substr(expected.size());
  unsigned long addr = std::stol(rem);
  EXPECT_EQ(addr, (intptr_t) C);
}
}; // namespace
