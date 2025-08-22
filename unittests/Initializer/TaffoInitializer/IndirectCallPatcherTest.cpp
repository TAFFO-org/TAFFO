#include "TaffoInitializer/IndirectCallPatcher.h"
#include "TestUtils.h"

namespace {
using namespace llvm;
using namespace taffo;
using namespace taffo_test;

class IndirectCallPatcherTest : public taffo_test::Test {
protected:
  Function* F0;
  BasicBlock* BB0;
  const std::vector<std::string> prefixBlocklist {"__kmpc_omp_task", "__kmpc_reduce"};

  IndirectCallPatcherTest() {
    F0 = genFunction(*M, "caller", Type::getVoidTy(Context), {});
    BB0 = BasicBlock::Create(Context, "entry", F0);
  }
};

TEST_F(IndirectCallPatcherTest, containsUnsupportedFunction_lastFunction) {
  for (auto prefix : prefixBlocklist) {
    auto F1 = genFunction(*M, prefix, Type::getVoidTy(Context), {});
    CallInst::Create(F1, {}, "", BB0);
    ASSERT_TRUE(containsUnsupportedFunctions(F0, {}));
  }
}

TEST_F(IndirectCallPatcherTest, containsUnsupportedFunction_traversed) {
  /*x
   * F0: entry
   * F1: supportedFunction
   * F2: unsupported function
   * Calls: F0 -> F1 -> F2
   */

  for (auto prefix : prefixBlocklist) {
    auto F1 = genFunction(*M, "supportedFunction", Type::getVoidTy(Context), {});
    auto F2 = genFunction(*M, prefix, Type::getVoidTy(Context), {});
    auto BB1 = BasicBlock::Create(Context, "entry", F1);
    CallInst::Create(F1, {}, "", BB0);
    CallInst::Create(F2, {}, "", BB1);
    ASSERT_TRUE(containsUnsupportedFunctions(F0, {}));
    ASSERT_TRUE(containsUnsupportedFunctions(F1, {}));
  }
}

TEST_F(IndirectCallPatcherTest, containsUnsupportedFunction_no) {
  CallInst::Create(F0, {}, "", BB0);
  ASSERT_FALSE(containsUnsupportedFunctions(F0, {}));
}

} // namespace
