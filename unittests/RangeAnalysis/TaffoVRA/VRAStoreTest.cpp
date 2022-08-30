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

} // namespace
