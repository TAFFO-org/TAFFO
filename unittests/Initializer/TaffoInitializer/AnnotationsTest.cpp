#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include "TaffoInitializer/TaffoInitializerPass.h"
#include "TestUtils.h"


namespace
{
using namespace taffo;
using namespace llvm;


class AnnotationsTest : public testing::Test
{
protected:
  const std::string STARTP_NOT_INIT_ERR = "__taffo_vra_starting_function not initialized to anything or initialized incorrectly!";
  const std::string STARTP_BAD_INIT_ERR = "__taffo_vra_starting_function initialized incorrectly!";

  std::string code;
  taffo::ValueInfo info;
  llvm::Value *val{};
  llvm::LLVMContext Context;


  TaffoInitializer initializer;

  AnnotationsTest()
  {
    llvm::install_fatal_error_handler(FatalErrorHandler, nullptr);
  }

  ~AnnotationsTest() override
  {
    llvm::remove_fatal_error_handler();
  }

  /**
   * @brief Check the correctness of parsed metadata against a fixed template
   *
   * Annotation string: "target('test') scalar(range(0, 10) type(1 2) error(3.1415) disabled final)"
   *
   * @param toCheck the metadata to check
   * @return @c true if the metadata correspond to the intended values,@c false otherwise
   */
  static void checkMD(const ValueInfo &toCheck)
  {
    std::shared_ptr<mdutils::MDInfo> mdinfo = toCheck.metadata;
    auto *metadata = cast<mdutils::InputInfo>(mdinfo.get());

    // range
    auto min = metadata->IRange->Min;
    auto max = metadata->IRange->Max;
    EXPECT_EQ(min, 0);
    EXPECT_EQ(max, 10);

    // type
    auto width = cast<mdutils::FPType>(metadata->IType.get())->getWidth();
    auto pointPos = cast<mdutils::FPType>(metadata->IType.get())->getPointPos();
    auto isSigned = cast<mdutils::FPType>(metadata->IType.get())->isSigned();
    EXPECT_EQ(width, 1);
    EXPECT_EQ(pointPos, 2);
    EXPECT_TRUE(isSigned);

    auto error = *metadata->IError;
    EXPECT_EQ(error, 3.1415);

    EXPECT_EQ(toCheck.backtrackingDepthLeft, 0);
  }
};


TEST_F(AnnotationsTest, StartingPoint_None)
{
  code = R"(
    define dso_local i32 @main() #0 {
        ret i32 0
    }
    )";

  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  llvm::Function *F = initializer.findStartingPointFunctionGlobal(*M);
  ASSERT_EQ(F, nullptr);
}

TEST_F(AnnotationsTest, StartingPoint_Set)
{
  code = R"(
    @__taffo_vra_starting_function = dso_local global i8* bitcast (i32 ()* @main to i8*), align 8

    define dso_local i32 @main() #0 {
        ret i32 0
    }
    )";

  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  llvm::Function *F = initializer.findStartingPointFunctionGlobal(*M);
  ASSERT_NE(F, nullptr);
  EXPECT_EQ(F->getName(), "main");

  F = initializer.findStartingPointFunctionGlobal(*M);
  EXPECT_EQ(F, nullptr);
}

TEST_F(AnnotationsTest, StartingPoint_Unset)
{
  code = R"(
    @__taffo_vra_starting_function = dso_local global i8* null, align 8

    define dso_local i32 @main() #0 {
        ret i32 0
    }
    )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  try {
    llvm::Function *F = initializer.findStartingPointFunctionGlobal(*M);
  } catch (const std::exception &e) {
    EXPECT_STREQ(e.what(), STARTP_NOT_INIT_ERR.c_str());
  }
}

TEST_F(AnnotationsTest, StartingPoint_NotAFunction)
{
  code = R"(
    @__taffo_vra_starting_function = dso_local constant i32 0, align 4

    define dso_local i32 @main() #0 {
      ret i32 0
    }
    )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  try {
    llvm::Function *F = initializer.findStartingPointFunctionGlobal(*M);
  } catch (const std::exception &e) {
    EXPECT_STREQ(e.what(), STARTP_NOT_INIT_ERR.c_str());
  }
}


} // namespace
