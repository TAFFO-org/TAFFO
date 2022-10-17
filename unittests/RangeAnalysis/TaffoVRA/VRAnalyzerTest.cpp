#include "TestUtils.h"

#include "TaffoVRA/Range.hpp"
#include "TaffoVRA/VRAGlobalStore.hpp"
#include "TaffoVRA/VRAnalyzer.hpp"

namespace
{

using namespace llvm;
using namespace taffo;
using namespace taffo_test;


class VRAnalyzerTest : public taffo_test::Test
{
private:
  Pass *P;
  Function *F0; // acts like a main from which instructions are called

protected:
  std::shared_ptr<VRAGlobalStore> GlobalStore = std::make_shared<VRAGlobalStore>(*new VRAGlobalStore());
  CodeInterpreter CI = CodeInterpreter(reinterpret_cast<llvm::Pass &>(P), GlobalStore);
  VRAnalyzer VRA = VRAnalyzer(CI);

  Function *F;
  BasicBlock *BB;
  Instruction *I;


  VRAnalyzerTest()
  {
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
TEST_F(VRAnalyzerTest, handleMathCallInstruction)
{
  F = genFunction(*M, "ceil", Type::getVoidTy(Context), {Type::getInt32Ty(Context)});
  I = InvokeInst::Create(F, BB, BB, {ConstantInt::get(Type::getInt32Ty(Context), 1)}, "", BB);
  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 1);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAnalyzerTest, handleMallocCall_calloc)
{
  F = genFunction(*M, "calloc", Type::getVoidTy(Context), {Type::getInt32Ty(Context)});
  I = InvokeInst::Create(F, BB, BB, {ConstantInt::get(Type::getInt32Ty(Context), 1)}, "", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(node);
  ASSERT_NE(ptrNode, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent());
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 0);
}

TEST_F(VRAnalyzerTest, handleMallocCall_scalar)
{
  F = genFunction(*M, "malloc", Type::getVoidTy(Context), {Type::getInt32Ty(Context)});
  I = InvokeInst::Create(F, BB, BB, {ConstantInt::get(Type::getInt32Ty(Context), 1)}, "", BB);
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

TEST_F(VRAnalyzerTest, handleMallocCall_scalarNoAnno)
{
  F = genFunction(*M, "malloc", Type::getVoidTy(Context), {Type::getInt32Ty(Context)});
  I = InvokeInst::Create(F, BB, BB, {ConstantInt::get(Type::getInt32Ty(Context), 1)}, "", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(node);
  ASSERT_NE(ptrNode, nullptr);
  EXPECT_EQ(ptrNode->getParent(), nullptr);
}

TEST_F(VRAnalyzerTest, handleMallocCall_struct)
{
  auto ST = StructType::create({Type::getInt32Ty(Context), Type::getInt32Ty(Context)});
  F = genFunction(*M, "malloc", PointerType::get(ST, 0), {Type::getInt32Ty(Context)});
  I = InvokeInst::Create(F, BB, BB, {ConstantInt::get(Type::getInt32Ty(Context), 1)}, "", BB);
  auto BC = BitCastInst::Create(Instruction::BitCast, I, PointerType::get(ST, 0), "", BB);
  auto *SI = new mdutils::StructInfo(2);
  SI->setField(0, std::make_shared<mdutils::InputInfo>(*genII(1, 2)));
  SI->setField(1, std::make_shared<mdutils::InputInfo>(*genII(3, 4)));
  mdutils::MetadataManager::setMDInfoMetadata(I, SI);
  GlobalStore->harvestMetadata(*M);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(node);
  ASSERT_NE(structNode, nullptr);
  ASSERT_EQ(structNode->fields().size(), 2);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structNode->fields()[0]);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
  scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structNode->fields()[1]);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 3);
  EXPECT_EQ(scalarNode->getRange()->max(), 4);
}

TEST_F(VRAnalyzerTest, handleMallocCall_structNoAnno)
{
  auto ST = StructType::create({Type::getInt32Ty(Context), Type::getInt32Ty(Context)});
  F = genFunction(*M, "malloc", PointerType::get(ST, 0), {Type::getInt32Ty(Context)});
  I = InvokeInst::Create(F, BB, BB, {ConstantInt::get(Type::getInt32Ty(Context), 1)}, "", BB);
  auto BC = BitCastInst::Create(Instruction::BitCast, I, PointerType::get(ST, 0), "", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(node);
  ASSERT_NE(structNode, nullptr);
  ASSERT_EQ(structNode->fields().size(), 0);
}

TEST_F(VRAnalyzerTest, handleMallocCall_pointer)
{
  F = genFunction(*M, "malloc", PointerType::get(Type::getInt32Ty(Context), 0), {Type::getInt32Ty(Context)});
  I = InvokeInst::Create(F, BB, BB, {ConstantInt::get(Type::getInt32Ty(Context), 1)}, "", BB);
  auto BC = BitCastInst::Create(Instruction::BitCast, I, PointerType::get(PointerType::get(Type::getInt32Ty(Context), 0), 0), "", BB);
  mdutils::MetadataManager::setMDInfoMetadata(I, genII(1, 2));
  GlobalStore->harvestMetadata(*M);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(node);
  ASSERT_NE(ptrNode, nullptr);
  EXPECT_EQ(ptrNode->getParent(), nullptr);
}

TEST_F(VRAnalyzerTest, handleIntrinsic_memcpy)
{
  auto dst = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto src = ConstantInt::get(Type::getInt32Ty(Context), 2);
  auto dst_bitcast = new BitCastInst(dst, Type::getInt32Ty(Context), "", BB);
  auto src_bitcast = new BitCastInst(src, Type::getInt32Ty(Context), "", BB);

  F = genFunction(*M, "llvm.memcpy", Type::getVoidTy(Context), {dst_bitcast->getType(), src_bitcast->getType()});
  I = InvokeInst::Create(F, BB, BB, {dst_bitcast, src_bitcast}, "", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(dst);
  ASSERT_NE(node, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 1);
}

TEST_F(VRAnalyzerTest, handleReturn)
{
  // FIXME: needs a non-null Pass
  /*
  auto F0 = genFunction(*M, "f0", Type::getInt32Ty(Context), {});
  CI.interpretFunction(F0); // needed to create the FunctionStore in the CodeInterpreter

    F = genFunction(*M, "foo", Type::getInt32Ty(Context), {Type::getInt32Ty(Context)});
    I = ReturnInst::Create(Context, ConstantInt::get(Type::getInt32Ty(Context), 1), BB);

    VRA.analyzeInstruction(I);
*/
}

TEST_F(VRAnalyzerTest, handleCast)
{
  auto op = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto scalar = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2}));
  VRA.setNode(op, std::make_shared<VRAScalarNode>(*scalar));
  I = CastInst::Create(Instruction::CastOps::SExt, op, Type::getInt64Ty(Context), "cast", BB);

  VRA.analyzeInstruction(I);
  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
}

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

TEST_F(VRAnalyzerTest, handleLoadInstr)
{
  // FIXME: needs a non-null Pass
}

TEST_F(VRAnalyzerTest, handleStoreInstr_value)
{
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

TEST_F(VRAnalyzerTest, handleStoreInstr_ptr)
{
  auto val = new PtrToIntInst(ConstantPointerNull::get(Type::getInt32PtrTy(Context)), Type::getInt32Ty(Context), "", BB);
  auto ptr = ConstantPointerNull::get(Type::getInt32PtrTy(Context));
  I = new StoreInst(val, ptr, BB);
  auto scalar = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2}));
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

TEST_F(VRAnalyzerTest, handleGEPInstr)
{
  auto ptr = ConstantPointerNull::get(Type::getInt32PtrTy(Context));
  auto idx = ConstantInt::get(Type::getInt32Ty(Context), 1);
  I = GetElementPtrInst::Create(nullptr, ptr, {idx}, "", BB);
  auto scalar = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2}));
  VRA.setNode(ptr, std::make_shared<VRAScalarNode>(*scalar));

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto GEPNode = std::dynamic_ptr_cast_or_null<VRAGEPNode>(node);
  ASSERT_NE(GEPNode, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(GEPNode->getParent());
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
}

TEST_F(VRAnalyzerTest, handleBitCastInstr)
{
  auto ptr = new PtrToIntInst(ConstantPointerNull::get(Type::getInt32PtrTy(Context)), Type::getInt32Ty(Context), "", BB);
  I = new BitCastInst(ptr, Type::getInt32Ty(Context), "", BB);
  auto scalar = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2}));
  VRA.setNode(ptr, std::make_shared<VRAScalarNode>(*scalar));

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
}

TEST_F(VRAnalyzerTest, handleCmpInstr)
{
  auto lhs = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto rhs = ConstantInt::get(Type::getInt32Ty(Context), 2);
  I = new ICmpInst(*BB, CmpInst::ICMP_EQ, lhs, rhs);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 0);
}

TEST_F(VRAnalyzerTest, handlePhiNode_scalar)
{
  auto lhs = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto rhs = ConstantInt::get(Type::getInt32Ty(Context), 2);
  auto phi = PHINode::Create(Type::getInt32Ty(Context), 2, "", BB);
  phi->addIncoming(lhs, BB);
  phi->addIncoming(rhs, BB);
  I = phi;

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
}

TEST_F(VRAnalyzerTest, handlePhiNode_pointer)
{
  auto lhs_val = ConstantInt::get(Type::getInt32Ty(Context), 5);
  auto rhs_val = ConstantInt::get(Type::getInt32Ty(Context), 6);
  auto lhs = ConstantExpr::getIntToPtr(lhs_val, Type::getInt32PtrTy(Context));
  auto rhs = ConstantExpr::getIntToPtr(rhs_val, Type::getInt32PtrTy(Context));
  auto s1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2}));
  auto s2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4}));
  VRA.setNode(lhs, std::make_shared<VRAPtrNode>(*new VRAPtrNode(std::make_shared<VRAScalarNode>(*s1))));
  VRA.setNode(rhs, std::make_shared<VRAPtrNode>(*new VRAPtrNode(std::make_shared<VRAScalarNode>(*s2))));
  auto phi = PHINode::Create(Type::getInt32PtrTy(Context), 2, "", BB);
  phi->addIncoming(lhs, BB);
  phi->addIncoming(rhs, BB);
  I = phi;

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(node);
  ASSERT_NE(ptrNode, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent());
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 4); // TODO: check if correct
}

TEST_F(VRAnalyzerTest, handleSelect)
{
  auto cond = ConstantInt::get(Type::getInt1Ty(Context), 1);
  auto lhs = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto rhs = ConstantInt::get(Type::getInt32Ty(Context), 2);
  I = SelectInst::Create(cond, lhs, rhs, "", BB);

  VRA.analyzeInstruction(I);

  auto node = VRA.getNode(I);
  ASSERT_NE(node, nullptr);
  auto scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 1);
  EXPECT_EQ(scalarNode->getRange()->max(), 2);
}

} // namespace
