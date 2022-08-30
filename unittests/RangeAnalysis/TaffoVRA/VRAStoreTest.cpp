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

TEST_F(VRAStoreTest, saveValueRange_new) {
  auto V = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto range = std::make_shared<range_t>(range_t{1, 2, false});
  VRAs.saveValueRange(V, range);

  auto node = VRAs.fetchRangeNode(V);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 2);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAStoreTest, saveValueRange_union) {
  auto V = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto range = std::make_shared<range_t>(range_t{1, 2, false});
  VRAs.saveValueRange(V, range);
  range = std::make_shared<range_t>(range_t{3, 4, false});
  VRAs.saveValueRange(V, range);

  auto node = VRAs.fetchRangeNode(V);
  ASSERT_NE(node, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(node);
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 4);
  EXPECT_FALSE(scalar->isFinal());
}

TEST_F(VRAStoreTest, saveValueRange_struct) {
  auto V = ConstantInt::get(Type::getInt32Ty(Context), 1);
  auto range = new VRAStructNode();
  range->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  VRAs.saveValueRange(V, std::make_shared<VRAStructNode>(*range));

  auto node = VRAs.fetchRangeNode(V);
  ASSERT_EQ(node, nullptr);
}

TEST_F(VRAStoreTest, setNode)
{
  auto V = ConstantInt::get(Type::getInt32Ty(Context), 0);
  auto N = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}));
  VRAs.setNode(V, std::make_shared<VRAScalarNode>(*N));

  auto retval = VRAs.getNode(V);
  ASSERT_NE(retval, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retval);
  ASSERT_NE(retnode, nullptr);
  EXPECT_EQ(retnode->getRange(), N->getRange());
}

TEST_F(VRAStoreTest, loadNode_scalar)
{
  auto node = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}));

  auto ret = VRAs.loadNode(std::make_shared<VRAScalarNode>(*node));
  ASSERT_NE(ret, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  ASSERT_NE(retnode, nullptr);
  EXPECT_EQ(retnode->getRange()->min(), 1);
  EXPECT_EQ(retnode->getRange()->max(), 2);
  EXPECT_TRUE(retnode->getRange()->isFinal());
}

TEST_F(VRAStoreTest, loadNode_struct)
{
  auto node = new VRAStructNode();
  node->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, true})));

  auto ret = VRAs.loadNode(std::make_shared<VRAStructNode>(*node));
  ASSERT_NE(ret, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAStructNode>(ret);
  ASSERT_NE(retnode, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retnode->getNodeAt(0));
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 2);
  EXPECT_TRUE(scalar->getRange()->isFinal());
}

TEST_F(VRAStoreTest, loadNode_GEP)
{
  auto parent = new VRAStructNode();
  parent->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, true})));
  parent->setNodeAt(1, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{3, 4, true})));
  parent->setNodeAt(2, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{5, 6, true})));
  unsigned int offset[2] = {1, 2};
  auto node = new VRAGEPNode(std::make_shared<VRAStructNode>(*parent), offset);

  auto ret = VRAs.loadNode(std::make_shared<VRAGEPNode>(*node));
  ASSERT_NE(ret, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  ASSERT_NE(retnode, nullptr);
  // struct element in position 2 (last index of the offset array)
  EXPECT_EQ(retnode->getRange()->min(), 5);
  EXPECT_EQ(retnode->getRange()->max(), 6);
  EXPECT_TRUE(retnode->getRange()->isFinal());
}

TEST_F(VRAStoreTest, loadNode_ptr)
{
  auto parent = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}));
  auto node = new VRAPtrNode(std::make_shared<VRAScalarNode>(*parent));

  auto ret = VRAs.loadNode(std::make_shared<VRAPtrNode>(*node));
  ASSERT_NE(ret, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  ASSERT_NE(retnode, nullptr);
  EXPECT_EQ(retnode->getRange()->min(), 1);
  EXPECT_EQ(retnode->getRange()->max(), 2);
  EXPECT_TRUE(retnode->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_GEP)
{
  auto src = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}));
  auto dst = std::make_shared<VRAGEPNode>(*new VRAGEPNode(std::make_shared<VRAStructNode>(), {0}));

  VRAs.storeNode(dst, std::make_shared<VRAScalarNode>(*src));
  auto ret = dst->getParent();
  ASSERT_NE(ret, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAStructNode>(ret);
  ASSERT_NE(retnode, nullptr);
  auto scalar = std::dynamic_ptr_cast_or_null<VRAScalarNode>(retnode->getNodeAt(0));
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(scalar->getRange()->min(), 1);
  EXPECT_EQ(scalar->getRange()->max(), 2);
  EXPECT_TRUE(scalar->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_simpleStruct)
{
  auto src = std::make_shared<VRAStructNode>();
  src->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));

  auto dst = std::make_shared<VRAStructNode>();
  dst->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{0, 0, false})));

  VRAs.storeNode(dst, src);

  ASSERT_EQ(dst->fields().size(), 1);
  auto field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(dst->getNodeAt(0));
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 0);
  EXPECT_EQ(field->getRange()->max(), 2);
  EXPECT_FALSE(field->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_nestedStruct)
{
  auto src_inner = std::make_shared<VRAStructNode>();
  src_inner->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  auto src = std::make_shared<VRAStructNode>();
  src->setNodeAt(0, src_inner);
  auto dst_inner = std::make_shared<VRAStructNode>();
  dst_inner->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{0, 0, false})));
  auto dst = std::make_shared<VRAStructNode>();
  dst->setNodeAt(0, dst_inner);

  VRAs.storeNode(dst, src);

  ASSERT_EQ(dst->fields().size(), 1);
  auto field = std::dynamic_ptr_cast_or_null<VRAStructNode>(dst->getNodeAt(0));
  ASSERT_NE(field, nullptr);
  ASSERT_EQ(field->fields().size(), 1);
  auto field_inner = std::dynamic_ptr_cast_or_null<VRAScalarNode>(field->getNodeAt(0));
  ASSERT_NE(field_inner, nullptr);
  EXPECT_EQ(field_inner->getRange()->min(), 0);
  EXPECT_EQ(field_inner->getRange()->max(), 2);
  EXPECT_FALSE(field_inner->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_structLargerSrc)
{
  auto src = std::make_shared<VRAStructNode>();
  src->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  src->setNodeAt(1, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{3, 4, false})));
  auto dst = std::make_shared<VRAStructNode>();
  dst->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{0, 0, false})));

  VRAs.storeNode(dst, src);

  ASSERT_EQ(dst->fields().size(), 2);
  auto field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(dst->getNodeAt(0));
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 0);
  EXPECT_EQ(field->getRange()->max(), 2);
  EXPECT_FALSE(field->getRange()->isFinal());
  field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(dst->getNodeAt(1));
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 3);
  EXPECT_EQ(field->getRange()->max(), 4);
  EXPECT_FALSE(field->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_structLargerDst)
{
  auto src = std::make_shared<VRAStructNode>();
  src->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{0, 0, false})));
  auto dst = std::make_shared<VRAStructNode>();
  dst->setNodeAt(0, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{1, 2, false})));
  src->setNodeAt(1, std::make_shared<VRAScalarNode>(std::make_shared<range_t>(range_t{3, 4, false})));

  VRAs.storeNode(dst, src);

  ASSERT_EQ(dst->fields().size(), 2);
  auto field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(dst->getNodeAt(0));
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 0);
  EXPECT_EQ(field->getRange()->max(), 2);
  EXPECT_FALSE(field->getRange()->isFinal());
  field = std::dynamic_ptr_cast_or_null<VRAScalarNode>(dst->getNodeAt(1));
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->getRange()->min(), 3);
  EXPECT_EQ(field->getRange()->max(), 4);
  EXPECT_FALSE(field->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_scalarPtr)
{
  auto src = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}));
  auto dst = new VRAScalarNode(std::make_shared<range_t>(range_t{0, 0, false}));
  auto dst_ptr = std::make_shared<VRAPtrNode>(std::make_shared<VRAScalarNode>(*dst));

  VRAs.storeNode(dst_ptr, std::make_shared<VRAScalarNode>(*src));
  auto ret = dst_ptr->getParent();
  ASSERT_NE(ret, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  ASSERT_NE(retnode, nullptr);
  EXPECT_EQ(retnode->getRange()->min(), 0);
  EXPECT_EQ(retnode->getRange()->max(), 2);
  EXPECT_FALSE(retnode->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_scalarPtrFinalDst)
{
  auto src = new VRAScalarNode(std::make_shared<range_t>(range_t{1, 2, true}));
  auto dst = new VRAScalarNode(std::make_shared<range_t>(range_t{0, 0, true}));
  auto dst_ptr = std::make_shared<VRAPtrNode>(std::make_shared<VRAScalarNode>(*dst));

  VRAs.storeNode(dst_ptr, std::make_shared<VRAScalarNode>(*src));
  auto ret = dst_ptr->getParent();
  ASSERT_NE(ret, nullptr);
  auto retnode = std::dynamic_ptr_cast_or_null<VRAScalarNode>(ret);
  ASSERT_NE(retnode, nullptr);
  EXPECT_EQ(retnode->getRange()->min(), 0);
  EXPECT_EQ(retnode->getRange()->max(), 0);
  EXPECT_TRUE(retnode->getRange()->isFinal());
}

TEST_F(VRAStoreTest, storeNode_genericPtr)
{
  auto src = std::make_shared<VRAStructNode>(*new VRAStructNode());
  auto dst = new VRAStructNode();
  auto dst_ptr = std::make_shared<VRAPtrNode>(std::make_shared<VRAStructNode>(*dst));

  VRAs.storeNode(dst_ptr, src);
  auto ret = dst_ptr->getParent();
  ASSERT_NE(ret, nullptr);
  EXPECT_EQ(ret, src);
}

TEST_F(VRAStoreTest, storeNode_structOffset) {
  // TODO: implement
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
