#include "gtest/gtest.h"
#include <memory>

#include "TaffoVRA/Range.hpp"
#include "TaffoVRA/VRAFunctionStore.hpp"
#include "TaffoVRA/VRAGlobalStore.hpp"
#include "TaffoVRA/VRAnalyzer.hpp"
#include "TestUtils.h"

namespace
{

using namespace llvm;
using namespace taffo;

class VRAFunctionStoreTest : public testing::Test
{
private:
  VRAGlobalStore GlobalStore = VRAGlobalStore();
  Pass *Pass;

protected:
  CodeInterpreter CI = CodeInterpreter(reinterpret_cast<llvm::Pass &>(Pass), std::make_shared<VRAGlobalStore>(GlobalStore));
  VRAFunctionStore VRAfs = VRAFunctionStore(CI);

  LLVMContext Context;
  std::shared_ptr<Module> M;
  Function *F;
  std::vector<Type *> args;
  std::list<NodePtrT> argsRanges;
  NodePtrT retval;
  NodePtrT ret;

  VRAFunctionStoreTest()
  {
    M = std::make_unique<Module>("test", Context);
  }
};

/*
 * more in-depth testing on convexMerge is done in VRAStoreTest.cpp,
 * here we test only the sameScalar case
 */
TEST_F(VRAFunctionStoreTest, convexMerge_VRAnalyzer) {
  VRAnalyzer Other(CI);

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRAfs.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  //Other.setNode(V1, std::make_shared<VRAScalarNode>(*N2));
  // TODO: determine how to run setNode on VRAnalyzer

  VRAfs.convexMerge(Other);

  auto node = VRAfs.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  /*EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
  */

}

TEST_F(VRAFunctionStoreTest, convexMerge_VRAGlobalStore) {
  VRAGlobalStore Other;

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRAfs.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  Other.setNode(V1, std::make_shared<VRAScalarNode>(*N2));

  VRAfs.convexMerge(Other);

  auto node = VRAfs.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAFunctionStoreTest, convexMerge_VRAFunctionStore) {
  VRAFunctionStore Other(CI);

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRAfs.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  Other.setNode(V1, std::make_shared<VRAScalarNode>(*N2));

  VRAfs.convexMerge(Other);

  auto node = VRAfs.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAFunctionStoreTest, setRetVal_new)
{
  retval = std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{0, 10}));
  VRAfs.setRetVal(retval);

  ret = VRAfs.getRetVal();
  EXPECT_NE(ret, nullptr);
  auto retcast = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  EXPECT_NE(retcast, nullptr);
  EXPECT_EQ(retcast->getRange()->min(), 0);
  EXPECT_EQ(retcast->getRange()->max(), 10);
}

TEST_F(VRAFunctionStoreTest, setRetVal_null)
{
  retval = nullptr;
  VRAfs.setRetVal(retval);

  ret = VRAfs.getRetVal();
  EXPECT_EQ(ret, nullptr);
}

TEST_F(VRAFunctionStoreTest, setRetVal_union)
{
  retval = std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{0, 10}));
  VRAfs.setRetVal(retval);
  retval = std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{5, 15}));
  VRAfs.setRetVal(retval);

  ret = VRAfs.getRetVal();
  EXPECT_NE(ret, nullptr);
  auto retcast = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  EXPECT_NE(retcast, nullptr);
  EXPECT_EQ(retcast->getRange()->min(), 0);
  EXPECT_EQ(retcast->getRange()->max(), 15);
}

TEST_F(VRAFunctionStoreTest, setArgumentRanges)
{
  args = {Type::getInt32Ty(Context), Type::getInt32Ty(Context)};
  argsRanges = {std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{0, 10})), std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{5, 15}))};
  F = genFunction(*M, Type::getVoidTy(Context), args);
  VRAfs.setArgumentRanges(*F, argsRanges);

  auto it = F->args().begin();
  auto arg = std::dynamic_ptr_cast_or_null<VRAScalarNode>(VRAfs.getNode(it));
  ASSERT_NE(arg, nullptr);
  EXPECT_EQ(arg->getRange()->min(), 0);
  EXPECT_EQ(arg->getRange()->max(), 10);
  it++;
  arg = std::dynamic_ptr_cast_or_null<VRAScalarNode>(VRAfs.getNode(it));
  ASSERT_NE(arg, nullptr);
  EXPECT_EQ(arg->getRange()->min(), 5);
  EXPECT_EQ(arg->getRange()->max(), 15);
  it++;
  auto end = VRAfs.getNode(it);
  ASSERT_EQ(end, nullptr);
}
} // namespace
