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

  range = new mdutils::Range(NaN,5);
  EXPECT_FALSE(VRAgs.isValidRange(range));

  range = new mdutils::Range(5,NaN);
  EXPECT_FALSE(VRAgs.isValidRange(range));

  range = new mdutils::Range(NaN,NaN);
  EXPECT_FALSE(VRAgs.isValidRange(range));
}

TEST_F(VRAGlobalStoreTest, ValidRange) {
  mdutils::Range *range;
  double inf = std::numeric_limits<double>::infinity();

  range = new mdutils::Range(5,5);
  EXPECT_TRUE(VRAgs.isValidRange(range));

  range = new mdutils::Range(-5,5);
  EXPECT_TRUE(VRAgs.isValidRange(range));

  range = new mdutils::Range(5,-5);
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
  for(int i = 0; i < 2; i++) {
    auto IRange = std::make_shared<mdutils::Range>(0, i);
    auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*II));
  }
  auto retval = VRAgs.harvestStructMD(SI, T);

  VRAStructNode *structNode;
  ASSERT_NE(structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get(), nullptr);
  EXPECT_EQ(structNode->getKind(), VRANode::VRAStructNodeK);

  int pos = 0;
  for(auto & f : structNode->fields()) {
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
  for(int i = 0; i < 2; i++) {
    auto IRange = std::make_shared<mdutils::Range>(0, i);
    auto *II = new mdutils::InputInfo(nullptr, IRange, nullptr, false, true);
    SI->setField(i, std::make_shared<mdutils::InputInfo>(*II));
  }
  auto retval = VRAgs.harvestStructMD(SI, T);

  VRAStructNode *structNode;
  ASSERT_NE(structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(retval).get(), nullptr);
  EXPECT_EQ(structNode->getKind(), VRANode::VRAStructNodeK);

  int pos = 0;
  for(auto & f : structNode->fields()) {
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
  for(int i = 0; i < 3; i++) {
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
  for(auto & f : innerStructNode->fields()) {
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


}
