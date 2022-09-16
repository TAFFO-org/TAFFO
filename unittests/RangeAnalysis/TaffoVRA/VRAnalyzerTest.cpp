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

/*
 * handling of instructions is tested by proxy via analyzeInstruction
 */
TEST_F(VRAnalyzerTest, handleBinaryInstr)
{
  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto V2 = ConstantInt::get(Type::getInt32Ty(Context), 2);
  I = BinaryOperator::Create(Instruction::Add, V1, V2, "add", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 3);
  EXPECT_EQ(scalar->getRange()->max(), 3);
}

TEST_F(VRAnalyzerTest, handleUnaryInstr)
{
  auto V1 = ConstantFP::get(Type::getFloatTy(Context), 3.1415);
  I = UnaryOperator::Create(Instruction::FNeg, V1, "neg", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_FLOAT_EQ(scalar->getRange()->min(), -3.1415);
  EXPECT_FLOAT_EQ(scalar->getRange()->max(), -3.1415);
}

TEST_F(VRAnalyzerTest, handleAllocaInstr_scalarNoAnno)
{
  I = new AllocaInst(Type::getInt32Ty(Context), 0, "alloca", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  EXPECT_EQ(scalarNode, nullptr);
}

TEST_F(VRAnalyzerTest, handleAllocaInstr_structNoAnno)
{
  auto S = StructType::create({Type::getInt32Ty(Context), Type::getInt32Ty(Context)});
  I = new AllocaInst(S, 0, "alloca", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(node);
  EXPECT_EQ(structNode->fields().size(), 0);
}

TEST_F(VRAnalyzerTest, handleAllocaInstr_struct)
{
  auto S = StructType::create({Type::getInt32Ty(Context), Type::getInt32Ty(Context)});
  I = new AllocaInst(S, 0, "alloca", BB);
  // populate UserInput
  auto *SI = new mdutils::StructInfo(2);
  SI->setField(0, std::make_shared<mdutils::InputInfo>(*genII(1, 2)));
  SI->setField(1, std::make_shared<mdutils::InputInfo>(*genII(3, 4)));
  mdutils::MetadataManager::setMDInfoMetadata(I, SI);
  GlobalStore->harvestMetadata(*M);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(node);
  EXPECT_NE(structNode, nullptr);
  ASSERT_EQ(structNode->fields().size(), 2);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structNode->getNodeAt(0));
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
  scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structNode->getNodeAt(1));
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 3);
  EXPECT_EQ(scalarNode->getRange()->max(), 4);
}

TEST_F(VRAnalyzerTest, handleAllocaInstr_scalar)
{
  I = new AllocaInst(Type::getInt32Ty(Context), 0, "alloca", BB);
  mdutils::MetadataManager::setMDInfoMetadata(I, genII(1, 2));
  GlobalStore->harvestMetadata(*M);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(node);
  ASSERT_NE(ptrNode, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent());
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
}

TEST_F(VRAnalyzerTest, handleStoreInstr_value) {
  auto val = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto ptr = ConstantPointerNull::get(Type::getInt32PtrTy(Context));
  I = new StoreInst(val, ptr, BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(ptr);
  ASSERT_NE(node, nullptr);
  auto ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(node);
  ASSERT_NE(ptrNode, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent());
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 1);
}

TEST_F(VRAnalyzerTest, handleStoreInstr_ptr) {
  auto val = new PtrToIntInst(ConstantPointerNull::get(Type::getInt32PtrTy(Context)), Type::getInt32Ty(Context), "", BB);
  auto ptr = ConstantPointerNull::get(Type::getInt32PtrTy(Context));
  I = new StoreInst(val, ptr, BB);
  auto scalar = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  VRA.setNode(I, std::make_shared<VRAScalarNode>(*scalar));

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(ptr);
  ASSERT_NE(node, nullptr);
    auto ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(node);
    ASSERT_NE(ptrNode, nullptr);
    auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent());
    EXPECT_EQ(scalarNode->getRange()->min(), 1);
    EXPECT_EQ(scalarNode->getRange()->max(), 2);
}

} // namespace
