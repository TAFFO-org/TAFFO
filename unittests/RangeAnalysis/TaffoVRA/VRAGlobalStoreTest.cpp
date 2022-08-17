#include "TaffoVRA/VRAGlobalStore.hpp"

#include <memory>
#include "TaffoVRA/Range.hpp"
#include "gtest/gtest.h"

namespace
{

using namespace llvm;
using namespace taffo;


class VRAGlobalStoreTest : public testing::Test
{
protected:
  VRAGlobalStore VRAgs;
  llvm::LLVMContext Context;
};

TEST_F(VRAGlobalStoreTest, InvalidRange) {
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

TEST_F(VRAGlobalStoreTest, ValidRange) {
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

TEST_F(VRAGlobalStoreTest, HarvestStructMD_Scalar) {
  llvm::Type *T = llvm::Type::getFloatTy(Context);
  auto IRange = std::make_shared<mdutils::Range>(0, 5);
  auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
  auto retval = VRAgs.harvestStructMD(II, T);

  VRAScalarNode *scalarNode;
  ASSERT_NE(scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(scalarNode->getKind(), VRANode::VRAScalarNodeK);

  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 5);
  EXPECT_TRUE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_Array) {
  int arraySize = 5;
  auto *T = llvm::ArrayType::get(llvm::Type::getFloatTy(Context), arraySize);
  auto IRange = std::make_shared<mdutils::Range>(0, 5);
  auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
  auto retval = VRAgs.harvestStructMD(II, T);

  VRAScalarNode *scalarNode;
  ASSERT_NE(scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(scalarNode->getKind(), VRANode::VRAScalarNodeK);

  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 5);
  EXPECT_TRUE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_ScalarPointer) {
  auto *T = llvm::PointerType::get(llvm::Type::getFloatTy(Context), 0);
  auto IRange = std::make_shared<mdutils::Range>(0, 5);
  auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
  auto retval = VRAgs.harvestStructMD(II, T);

  VRAPtrNode *ptrNode;
  ASSERT_NE(ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get(), nullptr);
  EXPECT_EQ(ptrNode->getKind(), VRANode::VRAPtrNodeK);

  VRAScalarNode *scalarNode;
  ASSERT_NE(scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent()).get(), nullptr);
  EXPECT_EQ(scalarNode->getKind(), VRANode::VRAScalarNodeK);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 5);
  EXPECT_TRUE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_StructPointer) {
  auto *S = llvm::StructType::create(Context, "struct");
  auto *F1 = llvm::Type::getFloatTy(Context);
  auto *F2 = llvm::Type::getFloatTy(Context);
  S->setBody({F1, F2});
  auto *T = llvm::PointerType::get(S, 0);

  auto *SI = new mdutils::StructInfo(2);
  for (int i = 0; i < 2; i++) {
    auto IRange = std::make_shared<mdutils::Range>(0, i);
    auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*II));
  }
  auto retval = VRAgs.harvestStructMD(SI, T);

  VRAStructNode *structNode;
  ASSERT_NE(structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get(), nullptr);
  EXPECT_EQ(structNode->getKind(), VRANode::VRAStructNodeK);

  int pos = 0;
  for (auto &f : structNode->fields()) {
    VRAScalarNode *scalarNode;
    EXPECT_NE(scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(f).get(), nullptr);
    EXPECT_EQ(scalarNode->getRange()->min(), 0);
    EXPECT_EQ(scalarNode->getRange()->max(), pos);
    EXPECT_TRUE(scalarNode->isFinal());
    EXPECT_EQ(scalarNode->getKind(), VRANode::VRAScalarNodeK);
    pos++;
  }
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_SimpleStruct) {
  auto *T = llvm::StructType::create(Context, "struct");
  auto *F1 = llvm::Type::getFloatTy(Context);
  auto *F2 = llvm::Type::getFloatTy(Context);
  T->setBody({F1, F2});

  auto *SI = new mdutils::StructInfo(2);
  for (int i = 0; i < 2; i++) {
    auto IRange = std::make_shared<mdutils::Range>(0, i);
    auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*II));
  }
  auto retval = VRAgs.harvestStructMD(SI, T);

  VRAStructNode *structNode;
  ASSERT_NE(structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get(), nullptr);
  EXPECT_EQ(structNode->getKind(), VRANode::VRAStructNodeK);

  int pos = 0;
  for (auto &f : structNode->fields()) {
    VRAScalarNode *scalarNode;
    EXPECT_NE(scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(f).get(), nullptr);
    EXPECT_EQ(scalarNode->getRange()->min(), 0);
    EXPECT_EQ(scalarNode->getRange()->max(), pos);
    EXPECT_TRUE(scalarNode->isFinal());
    EXPECT_EQ(scalarNode->getKind(), VRANode::VRAScalarNodeK);
    pos++;
  }
}

TEST_F(VRAGlobalStoreTest, HarvestStructMD_MixedStruct) {
  auto *S_INNER = llvm::StructType::create(Context, "struct");
  auto *F1 = llvm::Type::getFloatTy(Context);
  auto *F2 = llvm::Type::getFloatTy(Context);
  auto *F3 = llvm::Type::getFloatTy(Context);
  S_INNER->setBody({F1, F2, F3});

  auto *SI_INNER = new mdutils::StructInfo(3);
  for (int i = 0; i < 3; i++) {
    auto IRange = std::make_shared<mdutils::Range>(0, i);
    auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
    SI_INNER->setField(i, std::make_shared<mdutils::InputInfo>(*II));
  }

  auto *T = llvm::StructType::create(Context, "struct");
  auto *F = llvm::Type::getFloatTy(Context);
  auto *P = llvm::PointerType::get(F, 0);
  T->setBody({S_INNER, P});

  auto IRange = std::make_shared<mdutils::Range>(0, 3);
  auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, false);
  auto *SI = new mdutils::StructInfo(2);
  SI->setField(0, std::make_shared<mdutils::StructInfo>(*SI_INNER));
  SI->setField(1, std::make_shared<mdutils::InputInfo>(*II));

  auto retval = VRAgs.harvestStructMD(SI, T);

  VRAStructNode *structNode;
  ASSERT_NE(structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get(), nullptr);
  EXPECT_EQ(structNode->getKind(), VRANode::VRAStructNodeK);

  auto &innerStruct = structNode->fields()[0];
  VRAStructNode *innerStructNode;
  ASSERT_NE(innerStructNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(innerStruct).get(), nullptr);
  EXPECT_EQ(innerStructNode->getKind(), VRANode::VRAStructNodeK);
  int pos = 0;
  for (auto &f : innerStructNode->fields()) {
    VRAScalarNode *scalarNode;
    EXPECT_NE(scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(f).get(), nullptr);
    EXPECT_EQ(scalarNode->getRange()->min(), 0);
    EXPECT_EQ(scalarNode->getRange()->max(), pos);
    EXPECT_TRUE(scalarNode->isFinal());
    EXPECT_EQ(scalarNode->getKind(), VRANode::VRAScalarNodeK);
    pos++;
  }

  auto &outerStruct = structNode->fields()[1];
  VRAPtrNode *ptrNode;
  ASSERT_NE(ptrNode = std::dynamic_ptr_cast_or_null<VRAPtrNode>(outerStruct).get(), nullptr);
  EXPECT_EQ(ptrNode->getKind(), VRANode::VRAPtrNodeK);

  VRAScalarNode *scalarNode;
  ASSERT_NE(scalarNode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ptrNode->getParent()).get(), nullptr);
  EXPECT_EQ(scalarNode->getKind(), VRANode::VRAScalarNodeK);
  EXPECT_EQ(scalarNode->getRange()->min(), 0);
  EXPECT_EQ(scalarNode->getRange()->max(), 3);
  EXPECT_FALSE(scalarNode->isFinal());
}

TEST_F(VRAGlobalStoreTest, toMDInfo_Scalar) {
  range_t range = {0, 1, true};
  auto *scalarNode = new VRAScalarNode(std::make_shared<range_t>(range));
  auto retval = VRAgs.toMDInfo(std::shared_ptr<VRAScalarNode>(scalarNode));

  mdutils::InputInfo *II;
  ASSERT_NE(II = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(retval).get(), nullptr);
  EXPECT_EQ(II->IRange->Min, 0);
  EXPECT_EQ(II->IRange->Max, 1);
  // EXPECT_TRUE(II->IFinal); //TODO: check what the expected behavior should be
}

TEST_F(VRAGlobalStoreTest, toMDInfo_ScalarNoRange) {
  auto *scalarNode = new VRAScalarNode(nullptr);
  auto retval = VRAgs.toMDInfo(std::shared_ptr<VRAScalarNode>(scalarNode));
  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, toMDInfo_Struct) {
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

  mdutils::StructInfo *SI_OUTER;
  ASSERT_NE(SI_OUTER = std::dynamic_ptr_cast_or_null<mdutils::StructInfo>(retval).get(), nullptr);
  EXPECT_EQ(SI_OUTER->size(), 2);
  mdutils::InputInfo *II_OUTER;
  EXPECT_NE(II_OUTER = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(SI_OUTER->getField(0)).get(), nullptr);
  EXPECT_EQ(II_OUTER->IRange->Min, 0);
  EXPECT_EQ(II_OUTER->IRange->Max, 2);
  // EXPECT_FALSE(II_OUTER->IFinal);
  mdutils::StructInfo *SI_INNER;
  EXPECT_NE(SI_INNER = std::dynamic_ptr_cast_or_null<mdutils::StructInfo>(SI_OUTER->getField(1)).get(), nullptr);
  EXPECT_EQ(SI_INNER->size(), 1);
  mdutils::InputInfo *II_INNER;
  EXPECT_NE(II_INNER = std::dynamic_ptr_cast_or_null<mdutils::InputInfo>(SI_INNER->getField(0)).get(), nullptr);
  EXPECT_EQ(II_INNER->IRange->Min, 0);
  EXPECT_EQ(II_INNER->IRange->Max, 1);
  // EXPECT_TRUE(II_INNER->IFinal);
}

TEST_F(VRAGlobalStoreTest, updateMDInfo_Scalar)
{
  range_t range = {0, 1, true};
  auto *scalarNode = new VRAScalarNode(std::make_shared<range_t>(range));
  auto IRange = std::make_shared<mdutils::Range>(3, 5);
  auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
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
  SI->setField(0, std::make_shared<mdutils::InputInfo>(nullptr, std::make_shared<mdutils::Range>(4, 5), nullptr, false, true));
  auto *SI_INNER = new mdutils::StructInfo(1);
  SI_INNER->setField(0, std::make_shared<mdutils::InputInfo>(nullptr, std::make_shared<mdutils::Range>(6, 7), nullptr, false, true));
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

TEST_F(VRAGlobalStoreTest, fetchConstant_Integer) {
  auto *k = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 42);
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 42);
  EXPECT_EQ(kval->getRange()->max(), 42);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_FP) {
  auto *k = llvm::ConstantFP::get(llvm::Type::getFloatTy(Context), 3.1415);
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_FLOAT_EQ(kval->getRange()->min(), 3.1415);
  EXPECT_FLOAT_EQ(kval->getRange()->max(), 3.1415);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_TokenNone) {
    auto *k = llvm::ConstantTokenNone::get(Context);
    auto retval = VRAgs.fetchConstant(k);

    VRAScalarNode *kval;
    ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
    EXPECT_EQ(kval->getRange()->min(), 0);
    EXPECT_EQ(kval->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_PointerNull) {
    auto *k = llvm::ConstantPointerNull::get(llvm::Type::getInt32Ty(Context)->getPointerTo());
    auto retval = VRAgs.fetchConstant(k);

    VRAPtrNode *kval;
    ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get(), nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_UndefValue) {
  auto *k = llvm::UndefValue::get(llvm::Type::getInt32Ty(Context));
  auto retval = VRAgs.fetchConstant(k);
  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateZeroStruct) {
  auto *s_inner = llvm::StructType::create(Context, {llvm::Type::getInt32Ty(Context)});
  auto *s = llvm::StructType::create(Context, {llvm::Type::getInt32Ty(Context), s_inner});
  auto *k = llvm::ConstantAggregateZero::get(s);
  auto retval = VRAgs.fetchConstant(k);

  VRAStructNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get(), nullptr);
  VRAScalarNode *outerScalar;
  EXPECT_NE(outerScalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(kval->fields()[0]).get(), nullptr);
  EXPECT_EQ(outerScalar->getRange()->min(), 0);
  EXPECT_EQ(outerScalar->getRange()->max(), 0);
  VRAStructNode *structInner;
  EXPECT_NE(structInner = std::dynamic_ptr_cast_or_null<VRAStructNode>(kval->fields()[1]).get(), nullptr);
  EXPECT_EQ(structInner->fields().size(), 1);
  VRAScalarNode *scalarInner;
  EXPECT_NE(scalarInner = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structInner->fields()[0]).get(), nullptr);
  EXPECT_EQ(scalarInner->getRange()->min(), 0);
  EXPECT_EQ(scalarInner->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateZeroArray) {
  auto *v = llvm::ArrayType::get(llvm::Type::getInt32Ty(Context), 2);
  auto *k = llvm::ConstantAggregateZero::get(v);
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 0);
  EXPECT_EQ(kval->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateZeroVector) {
  auto *v = llvm::VectorType::get(llvm::Type::getInt32Ty(Context), 2, false);
  auto *k = llvm::ConstantAggregateZero::get(v);
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 0);
  EXPECT_EQ(kval->getRange()->max(), 0);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_DataSequential) {
  /*
  auto *v = llvm::ArrayType::get(llvm::Type::getInt32Ty(Context), 2);
  //auto *v_init = llvm::ConstantArray::get(v, {llvm::ConstantFP::get(llvm::Type::getFloatTy(Context), 3.1415), llvm::ConstantFP::get(llvm::Type::getFloatTy(Context), 2.718)});
  float init[2] = {3.1415, 2.718};
  auto arrayRef = new llvm::ArrayRef<float>(init, 2);
  auto *k = llvm::ConstantDataArray::get(Context, arrayRef);
  auto retval = VRAgs.fetchConstant(k);


  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 2.718);
  EXPECT_EQ(kval->getRange()->max(), 3.1415);
   */
  //FIXME understand how to create a ConstantDataArray object without crying
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GEPExpr) {
  // TODO: write test
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateStruct) {
  //FIXME: write once implemented
}

TEST_F(VRAGlobalStoreTest, fetchConstant_AggregateArray) {
  auto *v = llvm::ArrayType::get(llvm::Type::getInt32Ty(Context), 2);
  auto *k = llvm::ConstantArray::get(v, {llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 1), llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 2)});
  auto retval = VRAgs.fetchConstant(k);

  VRAScalarNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval).get(), nullptr);
  EXPECT_EQ(kval->getRange()->min(), 1);
  EXPECT_EQ(kval->getRange()->max(), 2);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalVariable) {
  auto M = std::make_unique<llvm::Module>("test", Context);
  auto globalVar = new llvm::GlobalVariable(*M, llvm::Type::getInt32Ty(Context), false, llvm::GlobalValue::ExternalLinkage, llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 42), "globalVar");
  auto retval = VRAgs.fetchConstant(globalVar);

  VRAPtrNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get(), nullptr);
  VRAScalarNode *scalar;
  EXPECT_NE(scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(kval->getParent()).get(), nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 42);
  EXPECT_EQ(scalar->getRange()->max(), 42);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalVariableNotInit) {
  auto M = std::make_unique<llvm::Module>("test", Context);
  auto globalVar = new llvm::GlobalVariable(*M, llvm::Type::getInt32Ty(Context), false, llvm::GlobalValue::ExternalLinkage, nullptr, "globalVar");
  auto retval = VRAgs.fetchConstant(globalVar);

  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalAlias) {
  auto M = std::make_unique<llvm::Module>("test", Context);
  auto k = new llvm::GlobalVariable(*M, llvm::Type::getInt32Ty(Context), false, llvm::GlobalValue::ExternalLinkage, llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 42), "globalVar");
  auto alias = llvm::GlobalAlias::create(llvm::Type::getInt32Ty(Context), 0U, llvm::GlobalValue::ExternalLinkage, "alias", k);
  auto retval = VRAgs.fetchConstant(alias);

  VRAPtrNode *kval;
  ASSERT_NE(kval = std::dynamic_ptr_cast_or_null<VRAPtrNode>(retval).get(), nullptr);
  VRAScalarNode *scalar;
  EXPECT_NE(scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(kval->getParent()).get(), nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 42);
  EXPECT_EQ(scalar->getRange()->max(), 42);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_Function) {
  auto M = std::make_unique<llvm::Module>("test", Context);
  auto func = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(Context), false), llvm::GlobalValue::ExternalLinkage, "func", M.get());
  auto retval = VRAgs.fetchConstant(func);

  ASSERT_EQ(retval, nullptr);
}

TEST_F(VRAGlobalStoreTest, fetchConstant_GlobalIFunc) {
  auto M = std::make_unique<llvm::Module>("test", Context);
  auto func = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getInt32Ty(Context), false), llvm::GlobalValue::ExternalLinkage, "func", M.get());
  auto ifunc = llvm::GlobalIFunc::create(llvm::Type::getInt32Ty(Context), 0U, llvm::GlobalValue::ExternalLinkage, "ifunc", func, M.get());
  auto retval = VRAgs.fetchConstant(ifunc);

  ASSERT_EQ(retval, nullptr);
}


TEST_F(VRAGlobalStoreTest, setConstRangeMD_noFP) {
  auto &MDM = mdutils::MetadataManager::getMetadataManager();
  auto M = std::make_unique<llvm::Module>("test", Context);
  auto F = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(Context), false), llvm::GlobalValue::ExternalLinkage, "func", M.get());
  auto BB = llvm::BasicBlock::Create(Context, "", F);
  auto op1 = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 1);
  auto op2 = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 2);
  auto i = llvm::BinaryOperator::Create(llvm::Instruction::Add, op1, op2, "sum", BB);
  VRAGlobalStore::setConstRangeMetadata(MDM, *i);

  SmallVector<mdutils::InputInfo*, 2U> II;
  MDM.retrieveConstInfo(*i, II);
  ASSERT_EQ(II.size(), 2);
  EXPECT_EQ(II[0], nullptr);
  EXPECT_EQ(II[1], nullptr);
}

TEST_F(VRAGlobalStoreTest, setConstRangeMD_FP) {
  auto &MDM = mdutils::MetadataManager::getMetadataManager();
  auto M = std::make_unique<llvm::Module>("test", Context);
  auto F = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(Context), false), llvm::GlobalValue::ExternalLinkage, "func", M.get());
  auto BB = llvm::BasicBlock::Create(Context, "", F);
  auto op1 = llvm::ConstantFP::get(llvm::Type::getFloatTy(Context), 3.1415);
  auto op2 = llvm::ConstantFP::get(llvm::Type::getFloatTy(Context), 2.7182);
  auto i = llvm::BinaryOperator::Create(llvm::Instruction::FAdd, op1, op2, "fsum", BB);
  VRAGlobalStore::setConstRangeMetadata(MDM, *i);

  SmallVector<mdutils::InputInfo*, 2U> II;
  MDM.retrieveConstInfo(*i, II);
  ASSERT_EQ(II.size(), 2);
  EXPECT_FLOAT_EQ(II[0]->IRange->Min, 3.1415);
  EXPECT_FLOAT_EQ(II[0]->IRange->Max, 3.1415);
  EXPECT_FLOAT_EQ(II[1]->IRange->Min, 2.7182);
  EXPECT_FLOAT_EQ(II[1]->IRange->Max, 2.7182);
}

}
