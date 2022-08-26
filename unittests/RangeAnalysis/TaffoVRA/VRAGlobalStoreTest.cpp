#include "gtest/gtest.h"
#include <memory>

#include "TaffoVRA/Range.hpp"
#include "TaffoVRA/VRAGlobalStore.hpp"
#include "TestUtils.h"

namespace
{

using namespace llvm;
using namespace taffo;


class VRAGlobalStoreTest : public testing::Test
{
protected:
  VRAGlobalStore VRAgs;

  LLVMContext Context;
  std::shared_ptr<Module> M;
  Function *F;
  BasicBlock *BB;

  VRAGlobalStoreTest()
  {
    M = std::make_unique<Module>("test", Context);
  }
};

TEST_F(VRAGlobalStoreTest, InvalidRange)
{
  mdutils::Range *range;
  double NaN = std::numeric_limits<double>::quiet_NaN();

  range = nullptr;
  EXPECT_FALSE(VRAgs.isValidRange(range));

  range = new mdutils::Range(NaN, 5);
  EXPECT_FALSE(VRAgs.isValidRange(range));

  range = new mdutils::Range(5, NaN);
  EXPECT_FALSE(VRAgs.isValidRange(range));

  range = new mdutils::Range(NaN, NaN);
  EXPECT_FALSE(VRAgs.isValidRange(range));
}

TEST_F(VRAGlobalStoreTest, ValidRange)
{
  mdutils::Range *range;
  double inf = std::numeric_limits<double>::infinity();

  range = new mdutils::Range(5, 5);
  EXPECT_TRUE(VRAgs.isValidRange(range));

  range = new mdutils::Range(-5, 5);
  EXPECT_TRUE(VRAgs.isValidRange(range));

  range = new mdutils::Range(5, -5);
  EXPECT_TRUE(VRAgs.isValidRange(range));

  range = new mdutils::Range(-inf, inf);
  EXPECT_TRUE(VRAgs.isValidRange(range));
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_Scalar)
{
  Type *T = Type::getFloatTy(Context);
  auto retval = VRAgs.harvestStructMD(genII(0, 5, true), T);

  auto *scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get();
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 5);
  EXPECT_TRUE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_Array)
{
  int arraySize = 5;
  auto *T = ArrayType::get(Type::getFloatTy(Context), arraySize);
  auto retval = VRAgs.harvestStructMD(genII(0, 5, true), T);

  auto *scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get();
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 5);
  EXPECT_TRUE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_ScalarPointer)
{
  auto *T = PointerType::get(Type::getFloatTy(Context), 0);
  auto retval = VRAgs.harvestStructMD(genII(0, 5, true), T);

  auto *ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get();
  ASSERT_NE(ptrNode, nullptr);
  auto *scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent()).get();
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 5);
  EXPECT_TRUE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_StructPointer)
{
  auto *S = StructType::create(Context, "struct");
  auto *F1 = Type::getFloatTy(Context);
  auto *F2 = Type::getFloatTy(Context);
  S->setBody({F1, F2});
  auto *T = PointerType::get(S, 0);

  auto *SI = new mdutils::StructInfo(2);
  for (int i = 0; i < 2; i++)
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*genII(0, i, true)));

  auto retval = VRAgs.harvestStructMD(SI, T);

  auto *structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get();
  ASSERT_NE(structNode, nullptr);
  int pos = 0;
  for (auto &f : structNode->fields()) {
    auto *scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(f).get();
    EXPECT_NE(scalarNode, nullptr);
    EXPECT_EQ(scalarNode->getRange()->min(), 0);
    EXPECT_EQ(scalarNode->getRange()->max(), pos);
    EXPECT_TRUE(scalarNode->isFinal());
    pos++;
  }
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_SimpleStruct)
{
  auto *S = StructType::create(Context, "struct");
  auto *F1 = Type::getFloatTy(Context);
  auto *F2 = Type::getFloatTy(Context);
  S->setBody({F1, F2});

  auto *SI = new mdutils::StructInfo(2);
  for (int i = 0; i < 2; i++)
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*genII(0, i, true)));

  auto retval = VRAgs.harvestStructMD(SI, S);

  auto *structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get();
  ASSERT_NE(structNode, nullptr);

  int pos = 0;
  for (auto &f : structNode->fields()) {
    auto *scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(f).get();
    EXPECT_NE(scalarNode, nullptr);
    EXPECT_EQ(scalarNode->getRange()->min(), 0);
    EXPECT_EQ(scalarNode->getRange()->max(), pos);
    EXPECT_TRUE(scalarNode->isFinal());
    pos++;
  }
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_MixedStruct)
{
  auto *S_INNER = StructType::create(Context, "struct");
  auto *F1 = Type::getFloatTy(Context);
  auto *F2 = Type::getFloatTy(Context);
  auto *F3 = Type::getFloatTy(Context);
  S_INNER->setBody({F1, F2, F3});

  auto *SI_INNER = new mdutils::StructInfo(3);
  for (int i = 0; i < 3; i++)
    SI_INNER->setField(i, std::make_shared<mdutils::InputInfo>(*genII(0, i, true)));

  auto *T = StructType::create(Context, "struct");
  auto *F = Type::getFloatTy(Context);
  auto *P = PointerType::get(F, 0);
  T->setBody({S_INNER, P});

  auto *SI = new mdutils::StructInfo(2);
  SI->setField(0, std::make_shared<mdutils::StructInfo>(*SI_INNER));
  SI->setField(1, std::make_shared<mdutils::InputInfo>(*genII(0, 3, false)));

  auto retval = VRAgs.harvestStructMD(SI, T);

  auto *structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get();
  ASSERT_NE(structNode, nullptr);

  auto &innerStruct = structNode->fields()[0];
  auto *innerStructNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(innerStruct).get();
  ASSERT_NE(innerStructNode, nullptr);
  int pos = 0;
  for (auto &f : innerStructNode->fields()) {
    auto *scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(f).get();
    EXPECT_NE(scalarNode, nullptr);
    EXPECT_EQ(scalarNode->getRange()->min(), 0);
    EXPECT_EQ(scalarNode->getRange()->max(), pos);
    EXPECT_TRUE(scalarNode->isFinal());
    pos++;
  }

  auto &outerStruct = structNode->fields()[1];
  auto *ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(outerStruct).get();
  ASSERT_NE(ptrNode, nullptr);

  auto *scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent()).get();
  ASSERT_NE(scalarNode, nullptr);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 3);
  EXPECT_FALSE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, toMDInfo_Scalar)
{
  range_t range = {0, 1, true};
  auto *scalarNode = new VRAScalarNode(std::make_shared<range_t>(range));
  auto retval = VRAgs.toMDInfo(std::shared_ptr<VRAScalarNode>(scalarNode));

  mdutils::InputInfo *II;
  ASSERT_NE(II = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(retval).get(), nullptr);
  EXPECT_EQ(II->IRange->Min, 0);
  EXPECT_EQ(II->IRange->Max, 1);
  // EXPECT_TRUE(II->IFinal); //TODO: check what the expected behavior should be
}

TEST_F(VRAGlobalStoreTest, toMDInfo_ScalarNoRange)
{
  auto *scalarNode = new VRAScalarNode(nullptr);
  auto retval = VRAgs.toMDInfo(std::shared_ptr<VRAScalarNode>(scalarNode));
  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, toMDInfo_Struct)
{
  auto *structInner = new VRAStructNode();
  range_t r_inner = {0, 1, true};
  auto *scalarInner = new VRAScalarNode(std::make_shared<range_t>(r_inner));
  structInner->setNodeAt(0, std::shared_ptr<VRAScalarNode>(scalarInner));
  auto *structOuter = new VRAStructNode();
  range_t r_outer = {0, 2, false};
  auto *scalarOuter = new VRAScalarNode(std::make_shared<range_t>(r_outer));
  structOuter->setNodeAt(0, std::shared_ptr<VRAScalarNode>(scalarOuter));
  structOuter->setNodeAt(1, std::shared_ptr<VRAStructNode>(structInner));
  auto retval = VRAgs.toMDInfo(std::shared_ptr<VRAStructNode>(structOuter));

  auto *SI_OUTER = std::dynamic_ptr_cast_or_null<mdutils::StructInfo>(retval).get();
  ASSERT_NE(SI_OUTER, nullptr);
  EXPECT_EQ(SI_OUTER->size(), 2);
  auto *II_OUTER = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(SI_OUTER->getField(0)).get();
  EXPECT_NE(II_OUTER, nullptr);
  EXPECT_EQ(II_OUTER->IRange->Min, 0);
  EXPECT_EQ(II_OUTER->IRange->Max, 2);
  // EXPECT_FALSE(II_OUTER->IFinal);
  auto *SI_INNER = std::dynamic_ptr_cast_or_null<mdutils::StructInfo>(SI_OUTER->getField(1)).get();
  EXPECT_NE(SI_INNER, nullptr);
  EXPECT_EQ(SI_INNER->size(), 1);
  auto *II_INNER = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(SI_INNER->getField(0)).get();
  EXPECT_NE(II_INNER, nullptr);
  EXPECT_EQ(II_INNER->IRange->Min, 0);
  EXPECT_EQ(II_INNER->IRange->Max, 1);
  // EXPECT_TRUE(II_INNER->IFinal);
}

TEST_F(VRAGlobalStoreTest, updateMDInfo_Scalar)
{
  range_t range = {0, 1, true};
  auto *scalarNode = new VRAScalarNode(std::make_shared<range_t>(range));
  auto *II = genII(3, 5, true);
  auto MDtoUpdate = std::shared_ptr<mdutils::InputInfo>(II);
  VRAgs.updateMDInfo(MDtoUpdate, std::shared_ptr<VRAScalarNode>(scalarNode));

  EXPECT_EQ(II->IRange->Min, 0);
  EXPECT_EQ(II->IRange->Max, 1);
  // EXPECT_TRUE(II->IFinal);
}

TEST_F(VRAGlobalStoreTest, updateMDInfo_Struct)
{
  auto *structInner = new VRAStructNode();
  range_t r_inner = {0, 1, true};
  auto *scalarInner = new VRAScalarNode(std::make_shared<range_t>(r_inner));
  structInner->setNodeAt(0, std::shared_ptr<VRAScalarNode>(scalarInner));
  auto *structOuter = new VRAStructNode();
  range_t r_outer = {2, 3, false};
  auto *scalarOuter = new VRAScalarNode(std::make_shared<range_t>(r_outer));
  structOuter->setNodeAt(0, std::shared_ptr<VRAScalarNode>(scalarOuter));
  structOuter->setNodeAt(1, std::shared_ptr<VRAStructNode>(structInner));

  auto *SI = new mdutils::StructInfo(2);
  SI->setField(0, std::make_shared<mdutils::InputInfo>(*genII(4, 5, true)));
  auto *SI_INNER = new mdutils::StructInfo(1);
  SI_INNER->setField(0, std::make_shared<mdutils::InputInfo>(*genII(6, 7, true)));
  SI->setField(1, std::make_shared<mdutils::StructInfo>(*SI_INNER));
  auto MDtoUpdate = std::shared_ptr<mdutils::StructInfo>(SI);

  VRAgs.updateMDInfo(MDtoUpdate, std::shared_ptr<VRAStructNode>(structOuter));

  EXPECT_EQ(MDtoUpdate->size(), 2);
  mdutils::InputInfo *II_OUTER;
  EXPECT_NE(II_OUTER = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(MDtoUpdate->getField(0)).get(), nullptr);
  EXPECT_EQ(II_OUTER->IRange->Min, 2);
  EXPECT_EQ(II_OUTER->IRange->Max, 3);
  // EXPECT_FALSE(II_OUTER->IFinal);
  mdutils::StructInfo *SI_INNER_NEW;
  EXPECT_NE(SI_INNER_NEW = std::dynamic_ptr_cast_or_null<mdutils::StructInfo>(MDtoUpdate->getField(1)).get(), nullptr);
  EXPECT_EQ(SI_INNER_NEW->size(), 1);
  mdutils::InputInfo *II_INNER;
  EXPECT_NE(II_INNER = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(SI_INNER_NEW->getField(0)).get(), nullptr);
  EXPECT_EQ(II_INNER->IRange->Min, 0);
  EXPECT_EQ(II_INNER->IRange->Max, 1);
  // EXPECT_TRUE(II_INNER->IFinal);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_Integer)
{
  auto *k = ConstantInt::get(Type::getInt32Ty(Context), 42);
  auto retval = VRAgs.fetchConstant(k);

  auto *kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get();
  ASSERT_NE(kval, nullptr);
  EXPECT_EQ(kval->getRange()->min(), 42);
  EXPECT_EQ(kval->getRange()->max(), 42);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_FP)
{
  auto *k = ConstantFP::get(Type::getFloatTy(Context), 3.1415);
  auto retval = VRAgs.fetchConstant(k);

  auto *kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get();
  ASSERT_NE(kval, nullptr);
  EXPECT_FLOAT_EQ(kval->getRange()->min(), 3.1415);
  EXPECT_FLOAT_EQ(kval->getRange()->max(), 3.1415);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_TokenNone)
{
  auto *k = ConstantTokenNone::get(Context);
  auto retval = VRAgs.fetchConstant(k);

  auto *kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get();
  ASSERT_NE(kval, nullptr);
  EXPECT_EQ(kval->getRange()->min(), 0);
  EXPECT_EQ(kval->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_PointerNull)
{
  auto *k = ConstantPointerNull::get(Type::getInt32Ty(Context)->getPointerTo());
  auto retval = VRAgs.fetchConstant(k);

  auto *kval = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get();
  ASSERT_NE(kval, nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_UndefValue)
{
  auto *k = UndefValue::get(Type::getInt32Ty(Context));
  auto retval = VRAgs.fetchConstant(k);
  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateZeroStruct)
{
  auto *s_inner = StructType::create(Context, {Type::getInt32Ty(Context)});
  auto *s = StructType::create(Context, {Type::getInt32Ty(Context), s_inner});
  auto *k = ConstantAggregateZero::get(s);
  auto retval = VRAgs.fetchConstant(k);

  VRAStructNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get(), nullptr);
  auto *outerScalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(kval->fields()[0]).get();
  EXPECT_NE(outerScalar, nullptr);
  EXPECT_EQ(outerScalar->getRange()->min(), 0);
  EXPECT_EQ(outerScalar->getRange()->max(), 0);
  auto *structInner = std::dynamic_ptr_cast_or_null<VRAStructNode>(kval->fields()[1]).get();
  EXPECT_NE(structInner, nullptr);
  EXPECT_EQ(structInner->fields().size(), 1);
  auto *scalarInner = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structInner->fields()[0]).get();
  EXPECT_NE(scalarInner, nullptr);
  EXPECT_EQ(scalarInner->getRange()->min(), 0);
  EXPECT_EQ(scalarInner->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateZeroArray)
{
  auto *v = ArrayType::get(Type::getInt32Ty(Context), 2);
  auto *k = ConstantAggregateZero::get(v);
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 0);
  EXPECT_EQ(kval->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateZeroVector)
{
  auto *v = VectorType::get(Type::getInt32Ty(Context), 2, false);
  auto *k = ConstantAggregateZero::get(v);
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 0);
  EXPECT_EQ(kval->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_DataSequential)
{
  /*
  auto *v = ArrayType::get(Type::getInt32Ty(Context), 2);
  //auto *v_init = ConstantArray::get(v, {ConstantFP::get(Type::getFloatTy(Context), 3.1415), ConstantFP::get(Type::getFloatTy(Context), 2.718)});
  float init[2] = {3.1415, 2.718};
  auto arrayRef = new ArrayRef<float>(init, 2);
  auto *k = ConstantDataArray::get(Context, arrayRef);
  auto retval = VRAgs.fetchConstant(k);


  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 2.718);
  EXPECT_EQ(kval->getRange()->max(), 3.1415);
   */
  // FIXME understand how to create a ConstantDataArray object without crying
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GEPExpr)
{
  // TODO: write test
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateStruct)
{
  // FIXME: write once implemented
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateArray)
{
  auto *v = ArrayType::get(Type::getInt32Ty(Context), 2);
  auto *k = ConstantArray::get(v, {ConstantInt::get(Type::getInt32Ty(Context), 1), ConstantInt::get(Type::getInt32Ty(Context), 2)});
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 1);
  EXPECT_EQ(kval->getRange()->max(), 2);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalVariable)
{
  auto globalVar = genGlobalVariable(*M, Type::getInt32Ty(Context), 42);
  auto retval = VRAgs.fetchConstant(globalVar);

  VRAPtrNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get(), nullptr);
  VRAScalarNode *scalar;
  EXPECT_NE(scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(kval->getParent()).get(), nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 42);
  EXPECT_EQ(scalar->getRange()->max(), 42);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalVariableNotInit)
{
  auto globalVar = genGlobalVariable(*M, Type::getInt32Ty(Context));
  auto retval = VRAgs.fetchConstant(globalVar);

  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalAlias)
{
  auto k = genGlobalVariable(*M, Type::getInt32Ty(Context), 42);
  auto alias = GlobalAlias::create(Type::getInt32Ty(Context), 0U, GlobalValue::ExternalLinkage, "alias", k);
  auto retval = VRAgs.fetchConstant(alias);

  VRAPtrNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get(), nullptr);
  VRAScalarNode *scalar;
  EXPECT_NE(scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(kval->getParent()).get(), nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 42);
  EXPECT_EQ(scalar->getRange()->max(), 42);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_Function)
{
  F = genFunction(*M, Type::getVoidTy(Context));
  auto retval = VRAgs.fetchConstant(F);

  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalIFunc)
{
  F = genFunction(*M, Type::getInt32Ty(Context));
  auto ifunc = GlobalIFunc::create(Type::getInt32Ty(Context), 0U, GlobalValue::ExternalLinkage, "ifunc", F, M.get());
  auto retval = VRAgs.fetchConstant(ifunc);

  ASSERT_EQ(retval, nullptr);
}


TEST_F(VRAGlobalStoreTest, setConstRangeMD_noFP)
{
  auto &MDM = mdutils::MetadataManager::getMetadataManager();
  F = genFunction(*M, Type::getVoidTy(Context));
  BB = BasicBlock::Create(Context, "", F);
  auto op1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto op2 = ConstantInt::get(Type::getInt32Ty(Context), 2);
  auto I = BinaryOperator::Create(Instruction::Add, op1, op2, "sum", BB);
  VRAGlobalStore::setConstRangeMetadata(MDM, *I);

  SmallVector<mdutils::InputInfo *, 2U> II;
  MDM.retrieveConstInfo(*I, II);
  ASSERT_EQ(II.size(), 2);
  EXPECT_EQ(II[0], nullptr);
  EXPECT_EQ(II[1], nullptr);
}

TEST_F(VRAGlobalStoreTest, setConstRangeMD_FP)
{
  auto &MDM = mdutils::MetadataManager::getMetadataManager();
  F = genFunction(*M, Type::getVoidTy(Context));
  BB = BasicBlock::Create(Context, "", F);
  auto op1 = ConstantFP::get(Type::getFloatTy(Context), 3.1415);
  auto op2 = ConstantFP::get(Type::getFloatTy(Context), 2.7182);
  auto I = BinaryOperator::Create(Instruction::FAdd, op1, op2, "fsum", BB);
  VRAGlobalStore::setConstRangeMetadata(MDM, *I);

  SmallVector<mdutils::InputInfo *, 2U> II;
  MDM.retrieveConstInfo(*I, II);
  ASSERT_EQ(II.size(), 2);
  EXPECT_FLOAT_EQ(II[0]->IRange->Min, 3.1415);
  EXPECT_FLOAT_EQ(II[0]->IRange->Max, 3.1415);
  EXPECT_FLOAT_EQ(II[1]->IRange->Min, 2.7182);
  EXPECT_FLOAT_EQ(II[1]->IRange->Max, 2.7182);
}


TEST_F(VRAGlobalStoreTest, harvestMD_globalScalar)
{
  auto globalVar = genGlobalVariable(*M, Type::getInt32Ty(Context));
  auto *II = genII(0, 1, true);
  mdutils::MetadataManager::setInputInfoMetadata(*globalVar, *II);
  VRAgs.harvestMetadata(*M);

  auto retUI = VRAgs.getUserInput(globalVar);
  auto UI = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retUI);
  ASSERT_NE(UI, nullptr);
  EXPECT_EQ(UI->getRange()->min(), 0);
  EXPECT_EQ(UI->getRange()->max(), 1);
  EXPECT_TRUE(UI->isFinal());

  auto retDR = VRAgs.getNode(globalVar);
  ASSERT_NE(retDR, nullptr);
  auto DR = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retDR);
  ASSERT_NE(DR, nullptr);
  auto range = std::dynamic_ptr_cast_or_null<VRAScalarNode>(DR->getParent());
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->getRange()->min(), 0);
  EXPECT_EQ(range->getRange()->max(), 1);
  EXPECT_TRUE(range->isFinal());
}

TEST_F(VRAGlobalStoreTest, harvestMD_globalStruct)
{
  auto *T = StructType::create(Context, "struct");
  auto *F1 = Type::getFloatTy(Context);
  auto *F2 = Type::getFloatTy(Context);
  T->setBody({F1, F2});

  auto *SI = new mdutils::StructInfo(2);
  for (int i = 0; i < 2; i++)
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*genII(0, i, true)));

  auto globalVar = genGlobalVariable(*M, T);
  mdutils::MetadataManager::setStructInfoMetadata(*globalVar, *SI);
  VRAgs.harvestMetadata(*M);

  auto retUI = VRAgs.getUserInput(globalVar);
  ASSERT_NE(retUI, nullptr);
  auto UI = std::dynamic_ptr_cast_or_null<VRAStructNode>(retUI);
  ASSERT_NE(UI, nullptr);
  ASSERT_EQ(UI->fields().size(), 2);
  auto field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(UI->fields()[0]);
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 0);
  EXPECT_EQ(field->getRange()->max(), 0);
  EXPECT_TRUE(field->isFinal());
  field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(UI->fields()[1]);
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 0);
  EXPECT_EQ(field->getRange()->max(), 1);
  EXPECT_TRUE(field->isFinal());

  auto retDR = VRAgs.getNode(globalVar);
  auto DR = std::dynamic_ptr_cast_or_null<VRAStructNode>(retDR);
  ASSERT_NE(DR, nullptr);
  ASSERT_EQ(DR->fields().size(), 2);
  field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(DR->fields()[0]);
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 0);
  EXPECT_EQ(field->getRange()->max(), 0);
  EXPECT_TRUE(field->isFinal());
  field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(DR->fields()[1]);
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 0);
  EXPECT_EQ(field->getRange()->max(), 1);
  EXPECT_TRUE(field->isFinal());
}

TEST_F(VRAGlobalStoreTest, harvestMD_globalStructDerived)
{
  auto *T = StructType::create(Context, "struct");
  auto *F1 = Type::getFloatTy(Context);
  auto *F2 = Type::getFloatTy(Context);
  T->setBody({F1, F2});
  auto *SI = new mdutils::StructInfo(2);
  for (int i = 0; i < 2; i++)
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*genII(0, i)));

  auto globalVar = genGlobalVariable(*M, Type::getInt32Ty(Context));
  VRAgs.harvestMetadata(*M);

  auto retUI = VRAgs.getUserInput(globalVar);
  auto UI = std::dynamic_ptr_cast_or_null<VRAStructNode>(retUI);
  ASSERT_EQ(UI, nullptr);

  auto retDR = VRAgs.getNode(globalVar);
  auto DR = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retDR);
  ASSERT_NE(DR, nullptr);
  ASSERT_EQ(DR->getParent(), nullptr);
}

TEST_F(VRAGlobalStoreTest, harvestMD_globalScalarDerived)
{
  auto globalVar = genGlobalVariable(*M, Type::getInt32Ty(Context));
  VRAgs.harvestMetadata(*M);

  auto retUI = VRAgs.getUserInput(globalVar);
  ASSERT_EQ(retUI, nullptr);

  auto retDR = VRAgs.getNode(globalVar);
  ASSERT_NE(retDR, nullptr);
  auto DR = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retDR);
  EXPECT_NE(DR, nullptr);
}

TEST_F(VRAGlobalStoreTest, harvestMD_globalScalarConstant)
{
  auto value = ConstantInt::get(Type::getInt32Ty(Context), 42);
  auto globalVar = genGlobalVariable(*M, Type::getInt32Ty(Context), value);
  VRAgs.harvestMetadata(*M);

  auto retUI = VRAgs.getUserInput(globalVar);
  ASSERT_EQ(retUI, nullptr);
  auto retDR = VRAgs.getNode(globalVar);
  ASSERT_NE(retDR, nullptr);
  auto DRptr = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retDR);
  ASSERT_NE(DRptr, nullptr);
  auto DR = std::dynamic_ptr_cast_or_null<VRAScalarNode>(DRptr->getParent());
  EXPECT_EQ(DR->getRange()->min(), 42);
  EXPECT_EQ(DR->getRange()->max(), 42);
}

TEST_F(VRAGlobalStoreTest, harvestMD_globalScalarConstant2)
{
  // TODO: check if it is actually possible to have a GlobalValue of kind VRAScalarNodeK
}

TEST_F(VRAGlobalStoreTest, harvestMD_functionParametersNoWeight)
{
  auto args = std::vector<Type *>{Type::getInt32Ty(Context), Type::getInt32Ty(Context)};
  F = genFunction(*M, Type::getVoidTy(Context), args);
  BB = BasicBlock::Create(Context, "", F);
  auto argsMD = std::vector<mdutils::MDInfo *>{genII(0, 1), genII(0, 2)};
  mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, argsMD);
  VRAgs.harvestMetadata(*M);

  int ctr = 1;
  for (auto &arg : F->args()) {
    auto retUI = VRAgs.getUserInput(&arg);
    ASSERT_NE(retUI, nullptr);
    auto UI = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retUI);
    ASSERT_NE(UI, nullptr);
    EXPECT_EQ(UI->getRange()->min(), 0);
    EXPECT_EQ(UI->getRange()->max(), ctr);
    ++ctr;
  }
}

TEST_F(VRAGlobalStoreTest, harvestMD_functionParametersInitWeight)
{
  auto args = std::vector<Type *>{Type::getInt32Ty(Context), Type::getInt32Ty(Context)};
  F = genFunction(*M, Type::getVoidTy(Context), args);
  BB = BasicBlock::Create(Context, "", F);
  auto argsMD = std::vector<mdutils::MDInfo *>{genII(0, 1), genII(0, 2)};
  mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, argsMD);
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(F, std::vector<int>{1, 2});
  VRAgs.harvestMetadata(*M);

  auto arg = F->args().begin();
  auto retUI = VRAgs.getUserInput(arg);
  ASSERT_NE(retUI, nullptr);
  auto UI = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retUI);
  ASSERT_NE(UI, nullptr);
  EXPECT_EQ(UI->getRange()->min(), 0);
  EXPECT_EQ(UI->getRange()->max(), 1);
  arg++;
  retUI = VRAgs.getUserInput(arg);
  EXPECT_EQ(retUI, nullptr);
}

TEST_F(VRAGlobalStoreTest, harvestMD_instructionNoWeightScalar)
{
  M = std::make_unique<Module>("test", Context);
  auto args = std::vector<Type *>{Type::getInt32Ty(Context), Type::getInt32Ty(Context)};
  F = Function::Create(FunctionType::get(Type::getVoidTy(Context), args, false), Function::ExternalLinkage, "func", M.get());
  BB = BasicBlock::Create(Context, "", F);
  auto argsMD = std::vector<mdutils::MDInfo *>{new mdutils::InputInfo(nullptr, std::make_shared<mdutils::Range>(0, 1), nullptr, false, true), new mdutils::InputInfo(nullptr, std::make_shared<mdutils::Range>(0, 2), nullptr, false, true)};
  mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, argsMD);
  VRAgs.harvestMetadata(*M);

  int ctr = 1;
  for (auto &arg : F->args()) {
    auto retUI = VRAgs.getUserInput(&arg);
    ASSERT_NE(retUI, nullptr);
    auto UI = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retUI);
    ASSERT_NE(UI, nullptr);
    EXPECT_EQ(UI->getRange()->min(), 0);
    EXPECT_EQ(UI->getRange()->max(), ctr);
    ctr++;
  }
}

TEST_F(VRAGlobalStoreTest, harvestMD_instructionNoWeightStruct)
{
  // TODO: find an instruction which has a struct as operand type
}

TEST_F(VRAGlobalStoreTest, harvestMD_instructionWithWeight)
{
  auto args = std::vector<Type *>{Type::getInt32Ty(Context)};
  F = genFunction(*M, Type::getVoidTy(Context), args);
  BB = BasicBlock::Create(Context, "", F);
  auto *I = BinaryOperator::Create(Instruction::Add, F->args().begin(), ConstantInt::get(Type::getInt32Ty(Context), 42), "", BB);
  mdutils::MetadataManager::setMDInfoMetadata(I, genII(0, 1, true));
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(I, 2);
  mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, std::vector<mdutils::MDInfo *>{genII(2, 3, true)});
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(F, std::vector<int>{3});

  VRAgs.harvestMetadata(*M);
  auto retUI = VRAgs.getUserInput(I);
  ASSERT_NE(retUI, nullptr);
  auto UI = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retUI);
  ASSERT_NE(UI, nullptr);
  EXPECT_EQ(UI->getRange()->min(), 0);
  EXPECT_EQ(UI->getRange()->max(), 1);
}

TEST_F(VRAGlobalStoreTest, harvestMD_instructionWithLargerWeightThanParent)
{
  auto args = std::vector<Type *>{Type::getInt32Ty(Context)};
  F = genFunction(*M, Type::getVoidTy(Context), args);
  BB = BasicBlock::Create(Context, "", F);
  auto *I = BinaryOperator::Create(Instruction::Add, F->args().begin(), F->args().begin(), "", BB);
  auto MD = genII(0, 1);
  mdutils::MetadataManager::setMDInfoMetadata(I, MD);
  mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, {MD});
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(I, 4);
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(F, std::vector<int>{3});

  VRAgs.harvestMetadata(*M);
  auto retUI = VRAgs.getUserInput(I);
  ASSERT_EQ(retUI, nullptr);
}

TEST_F(VRAGlobalStoreTest, harvestMD_instructionWithInstrOp)
{
  auto args = std::vector<Type *>{Type::getInt32Ty(Context)};
  F = genFunction(*M, Type::getVoidTy(Context), args);
  BB = BasicBlock::Create(Context, "", F);
  auto MD = genII(0, 1);
  auto *I0 = BinaryOperator::Create(Instruction::Add, ConstantInt::get(Type::getInt32Ty(Context), 42), ConstantInt::get(Type::getInt32Ty(Context), 42), "", BB);
  auto *I = BinaryOperator::Create(Instruction::Add, I0, I0, "", BB);
  mdutils::MetadataManager::setMDInfoMetadata(I0, MD);
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(I0, 3);
  mdutils::MetadataManager::setMDInfoMetadata(I, MD);
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(I, 2);
  mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, std::vector<mdutils::MDInfo *>{genII(4, 5, true)});

  VRAgs.harvestMetadata(*M);
  auto retUI = VRAgs.getUserInput(I);
  ASSERT_NE(retUI, nullptr);
  auto UI = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retUI);
  ASSERT_NE(UI, nullptr);
  EXPECT_EQ(UI->getRange()->min(), 0);
  EXPECT_EQ(UI->getRange()->max(), 1);
}

TEST_F(VRAGlobalStoreTest, harvestMD_instructionAlloca)
{
  F = genFunction(*M, Type::getVoidTy(Context), {});
  BB = BasicBlock::Create(Context, "", F);
  auto *I = new AllocaInst(Type::getInt32Ty(Context), 0, "", BB);
  mdutils::MetadataManager::setMDInfoMetadata(I, genII(0, 1));
  mdutils::MetadataManager::setInputInfoInitWeightMetadata(I, 2);

  VRAgs.harvestMetadata(*M);
  auto retUI = VRAgs.getUserInput(I);
  ASSERT_EQ(retUI, nullptr);
}


} // namespace
