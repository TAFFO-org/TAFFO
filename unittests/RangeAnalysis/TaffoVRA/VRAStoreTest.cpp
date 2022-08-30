#include "gtest/gtest.h"
#include <memory>

#include "TaffoVRA/Range.hpp"
#include "TaffoVRA/VRAGlobalStore.hpp"
#include "TestUtils.h"

namespace
{

using namespace llvm;
using namespace taffo;

class VRAStoreStub : public VRAStore
{
public:
  VRAStoreStub()
      : VRAStore(VRASK_VRAGlobalStore, std::make_shared<VRALogger>()) {}
};

class VRAStoreTest : public testing::Test
{
protected:
  VRAStoreStub VRAs;

  LLVMContext Context;
  std::shared_ptr<Module> M;
  VRAStoreTest()
  {
    M = std::make_unique<Module>("test", Context);
  }
};

TEST_F(VRAStoreTest, convexMerge_sameScalar)
{
  VRAStoreStub other;

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRAs.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  other.setNode(V1, std::make_shared<VRAScalarNode>(*N2));

  VRAs.convexMerge(other);

  auto node = VRAs.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAStoreTest, convexMerge_distinctScalars)
{
  VRAStoreStub other;

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto V2 = ConstantInt::get(Type::getInt32Ty(Context), 2);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  auto N2 = new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, false}));
  VRAs.setNode(V1, std::make_shared<VRAScalarNode>(*N1));
  other.setNode(V2, std::make_shared<VRAScalarNode>(*N2));

  VRAs.convexMerge(other);

  auto node = VRAs.getNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 2);
  EXPECT_FALSE(scalar->isFinal());

  node = VRAs.getNode(V2);
  ASSERT_NE(node, nullptr);
  scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  EXPECT_EQ(scalar->getRange()->min(), 3);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAStoreTest, convexMerge_sameStruct)
{
  VRAStoreStub other;

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAStructNode();
  N1->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  auto N2 = new VRAStructNode();
  N2->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{3, 4, false})));
  VRAs.setNode(V1, std::make_shared<VRAStructNode>(*N1));
  other.setNode(V1, std::make_shared<VRAStructNode>(*N2));

  VRAs.convexMerge(other);

  auto structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(VRAs.getNode(V1));
  ASSERT_NE(structNode, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structNode->getNodeAt(0));
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAStoreTest, convexMerge_distinctStruct)
{
  VRAStoreStub other;

  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto V2 = ConstantInt::get(Type::getInt32Ty(Context), 2);
  auto N1 = new VRAStructNode();
  N1->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  auto N2 = new VRAStructNode();
  N2->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{3, 4, false})));
  VRAs.setNode(V1, std::make_shared<VRAStructNode>(*N1));
  other.setNode(V2, std::make_shared<VRAStructNode>(*N2));

  VRAs.convexMerge(other);

  auto structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(VRAs.getNode(V1));
  ASSERT_NE(structNode, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structNode->getNodeAt(0));
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 2);
  EXPECT_FALSE(scalar->isFinal());

  structNode = std::dynamic_ptr_cast_or_null<VRAStructNode>(VRAs.getNode(V2));
  ASSERT_NE(structNode, nullptr);
  scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(structNode->getNodeAt(0));
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 3);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAStoreTest, fetchRange_scalar)
{
  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  VRAs.setNode(V1, std::make_shared<VRAScalarNode>(*N1));

  auto range = VRAs.fetchRange(V1);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->min(), 1);
  EXPECT_EQ(range->max(), 2);
  EXPECT_FALSE(range->isFinal());
}

TEST_F(VRAStoreTest, fetchRange_struct)
{
  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAStructNode();
  N1->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  VRAs.setNode(V1, std::make_shared<VRAStructNode>(*N1));

  auto range = VRAs.fetchRange(V1);
  ASSERT_EQ(range, nullptr);
}

TEST_F(VRAStoreTest, fetchRangeNode_scalar)
{
  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
  VRAs.setNode(V1, std::make_shared<VRAScalarNode>(*N1));

  auto node = VRAs.fetchRangeNode(V1);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 2);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAStoreTest, fetchRangeNode_scalarPtr)
{
  // TODO: how do I get a pointer value?
  /*
    auto V1 = ConstantInt::get(Type::getInt32PtrTy(Context), 1);
    auto parent = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, false}));
    auto N1 = new VRAPtrNode(std::make_shared<VRAScalarNode>(*parent));
    VRAs.setNode(V1, std::make_shared<VRAPtrNode>(*N1));

    auto node = VRAs.fetchRangeNode(V1);
    ASSERT_NE(node, nullptr);
    auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
    ASSERT_NE(scalar, nullptr);
    EXPECT_EQ(scalar->getRange()->min(), 1);
    EXPECT_EQ(scalar->getRange()->max(), 2);
    EXPECT_FALSE(scalar->isFinal());
    */
}

TEST_F(VRAStoreTest, fetchRangeNode_struct)
{
  auto V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto N1 = new VRAStructNode();
  N1->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  VRAs.setNode(V1, std::make_shared<VRAStructNode>(*N1));

  auto node = VRAs.fetchRangeNode(V1);
  ASSERT_EQ(node, nullptr);
}

TEST_F(VRAStoreTest, fetchRange_NodePtr_scalar)
{
  auto node = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}));

  auto ret = VRAs.fetchRange(std::make_shared<VRAScalarNode>(*node));
  auto range = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->getRange()->min(), 1);
  EXPECT_EQ(range->getRange()->max(), 2);
  EXPECT_TRUE(range->isFinal());
}

TEST_F(VRAStoreTest, fetchRange_NodePtr_struct)
{
  auto node = new VRAStructNode();
  node->setNodeAt(0, std::make_shared<VRAScalarNode>(*new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}))));

  auto ret = VRAs.fetchRange(std::make_shared<VRAStructNode>(*node));
  auto range = std::dynamic_ptr_cast_or_null<VRAStructNode>(ret);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->fields().size(), 1);
  auto field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(range->getNodeAt(0));
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 1);
  EXPECT_EQ(field->getRange()->max(), 2);
  EXPECT_TRUE(field->getRange()->isFinal());
}

TEST_F(VRAStoreTest, fetchRange_NodePtr_GEP)
{
  auto node = new VRAStructNode();
  node->setNodeAt(0, std::make_shared<VRAScalarNode>(*new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}))));
  node->setNodeAt(1, std::make_shared<VRAScalarNode>(*new VRAScalarNode(std::make_shared<range_t>(range_t{3, 4, true}))));
  unsigned offset[2] = {1, 1};
  auto GEP = new VRAGEPNode(std::make_shared<VRAStructNode>(*node), offset);

  auto ret = VRAs.fetchRange(std::make_shared<VRAGEPNode>(*GEP));
  auto range = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->getRange()->min(), 3);
  EXPECT_EQ(range->getRange()->max(), 4);
  EXPECT_TRUE(range->getRange()->isFinal());
}

} // namespace
