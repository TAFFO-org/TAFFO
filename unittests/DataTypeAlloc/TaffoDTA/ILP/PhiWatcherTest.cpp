#include "TaffoDTA/ILP/PhiWatcher.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

namespace
{

using namespace llvm;
using namespace taffo_test;
using namespace tuner;


class PhiWatcherTest : public taffo_test::Test
{
protected:
  Function *F;
  BasicBlock *BB;

  PhiWatcher PW;

  PHINode *phiNode;
  Value *V1;
  Value *V2;

  PhiWatcherTest()
  {
    F = genFunction(*M, "functionName", Type::getVoidTy(Context), {});
    BB = BasicBlock::Create(Context, "basicblock", F);

    V1 = ConstantInt::get(Type::getInt32Ty(Context), 1);
    V2 = ConstantInt::get(Type::getInt32Ty(Context), 2);
    phiNode = PHINode::Create(Type::getInt32Ty(Context), 2, "", BB);
    phiNode->addIncoming(V1, BB);
    phiNode->addIncoming(V2, BB);
  }
};

TEST_F(PhiWatcherTest, openPhiLoop)
{
  PW.openPhiLoop(phiNode, V1);

  auto ret = PW.getPhiNodeToClose(V1);
  ASSERT_EQ(ret, phiNode);
}

TEST_F(PhiWatcherTest, closePhiLoop)
{
  PW.openPhiLoop(phiNode, V1);
  PW.openPhiLoop(phiNode, V2);
  PW.closePhiLoop(phiNode, V1);

  auto ret1 = PW.getPhiNodeToClose(V1);
  ASSERT_EQ(ret1, nullptr);
  auto ret2 = PW.getPhiNodeToClose(V2);
  ASSERT_EQ(ret2, phiNode);
}
}; // namespace
