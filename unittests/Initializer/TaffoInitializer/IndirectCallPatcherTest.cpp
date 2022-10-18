#include "TestUtils.h"

#include "TaffoInitializer/IndirectCallPatcher.h"

namespace
{
using namespace llvm;
using namespace taffo;
using namespace taffo_test;


class IndirectCallPatcherTest : public taffo_test::Test
{
  protected:
    Function *F0;
    BasicBlock *BB0;

  IndirectCallPatcherTest() {
    F0 = genFunction(*M, "caller", Type::getVoidTy(Context), {});
    BB0 = BasicBlock::Create(Context, "entry", F0);
  }
};

TEST_F(IndirectCallPatcherTest, containsUnsupportedFunction_lastFunction)
{
  auto F = genFunction(*M, "__kmpc_omp_task_fun", Type::getVoidTy(Context), {});
  auto CI = CallInst::Create(F, {}, "", BB0);
  ASSERT_TRUE(containsUnsupportedFunctions(F0, {}));

  F = genFunction(*M, "__kmpc_reduce_fun", Type::getVoidTy(Context), {});
  CI = CallInst::Create(F, {}, "", BB0);
  ASSERT_TRUE(containsUnsupportedFunctions(F0, {}));
}

TEST_F(IndirectCallPatcherTest, containsUnsupportedFunction_traversed)
{
  auto F = genFunction(*M, "supportedFunction", Type::getVoidTy(Context), {});
  auto F1 = genFunction(*M, "__kmpc_omp_task_fun", Type::getVoidTy(Context), {});
  auto BB = BasicBlock::Create(Context, "entry", F);
  auto CI = CallInst::Create(F, {}, "", BB0);
  auto CI1 = CallInst::Create(F1, {}, "", BB);
  ASSERT_TRUE(containsUnsupportedFunctions(F0, {}));
  ASSERT_TRUE(containsUnsupportedFunctions(F, {}));
}

TEST_F(IndirectCallPatcherTest, containsUnsupportedFunction_no)
{
  auto CI = CallInst::Create(F0, {}, "", BB0);
  ASSERT_FALSE(containsUnsupportedFunctions(F0, {}));
}

}