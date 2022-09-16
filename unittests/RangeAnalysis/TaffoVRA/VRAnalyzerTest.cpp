#include "gtest/gtest.h"
#include <memory>

#include "TaffoVRA/Range.hpp"
#include "TaffoVRA/VRAGlobalStore.hpp"
#include "TaffoVRA/VRAnalyzer.hpp"
#include "TestUtils.h"

namespace
{

using namespace llvm;
using namespace taffo;


class VRAnalyzerTest : public testing::Test
{
private:
  Pass *Pass;
  Function *F0; // acts like a main from which instructions are called

protected:
  std::shared_ptr<VRAGlobalStore> GlobalStore = std::make_shared<VRAGlobalStore>(*new VRAGlobalStore());
  CodeInterpreter CI = CodeInterpreter(reinterpret_cast<llvm::Pass &>(Pass), GlobalStore);
  VRAnalyzer VRA = VRAnalyzer(CI);

  LLVMContext Context;
  std::shared_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  Instruction *I;


  VRAnalyzerTest()
  {
    M = std::make_unique<Module>("test", Context);
    F0 = genFunction(*M, "main", Type::getVoidTy(Context), {});
    BB = BasicBlock::Create(Context, "entry", F0);
  }
};

/*
 * more in-depth testing on convexMerge is done in VRAStoreTest.cpp,
 * here we test only the sameScalar case
 */
TEST_F(VRAnalyzerTest, convexMerge_VRAnalyzer)
{
  VRAnalyzer Other(CI);

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRA.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  Other.setNode(V1, std::make_shared<VRAScalarNode>(*N2));

  VRA.convexMerge(Other);

  auto node = VRA.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAnalyzerTest, convexMerge_VRAGlobalStore)
{
  VRAGlobalStore Other;

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRA.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  Other.setNode(V1, std::make_shared<VRAScalarNode>(*N2));

  VRA.convexMerge(Other);

  auto node = VRA.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAnalyzerTest, convexMerge_VRAFunctionStore)
{
  VRAFunctionStore Other(CI);

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRA.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  Other.setNode(V1, std::make_shared<VRAScalarNode>(*N2));

  VRA.convexMerge(Other);

  auto node = VRA.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

} // namespace
