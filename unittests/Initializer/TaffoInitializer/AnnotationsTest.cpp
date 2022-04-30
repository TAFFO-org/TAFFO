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


TEST_F(AnnotationsTest, ParseAnnotation_GlobalVariable)
{
  code = R"(
    @var = dso_local global float 0.000000e+00, align 4
    @.str = private unnamed_addr constant [75 x i8] c"target('test') scalar(range(0, 10) type(1 2) error(3.1415) disabled final)\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @llvm.global.annotations = appending global
    [1 x { i8*, i8*, i8*, i32, i8* }]
    [{ i8*, i8*, i8*, i32, i8* } {
      i8* bitcast (float* @var to i8*),
      i8* getelementptr inbounds ([75 x i8], [75 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 1,
      i8* null
    }], section "llvm.metadata"

    define dso_local i32 @main() #0 {
      ret i32 0
  })";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  GlobalVariable *globalVars = M->getGlobalVariable("llvm.global.annotations");
  ASSERT_NE(globalVars, nullptr);

  MultiValueMap<llvm::Value *, ValueInfo> variables;
  auto *anno = cast<ConstantStruct>(globalVars->getInitializer()->getOperand(0));
  auto *annoPtrInstr = cast<ConstantExpr>(anno->getOperand(1));
  auto *instr = cast<ConstantExpr>(anno->getOperand(0))->getOperand(0);
  bool startingPoint;
  bool res = initializer.parseAnnotation(variables, annoPtrInstr, instr, &startingPoint);

  ASSERT_TRUE(res);

  Value *first = variables.begin()->first;
  ASSERT_EQ(first, instr);
  checkMD(variables.begin()->second);
  EXPECT_TRUE(startingPoint);
}

TEST_F(AnnotationsTest, ParseAnnotation_Function)
{
  code = R"(
    @.str = private unnamed_addr constant [75 x i8] c"target('test') scalar(range(0, 10) type(1 2) error(3.1415) disabled final)\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @llvm.global.annotations = appending global
    [1 x { i8*, i8*, i8*, i32, i8* }]
    [{ i8*, i8*, i8*, i32, i8* } {
      i8* bitcast (float ()* @fun to i8*),
      i8* getelementptr inbounds ([75 x i8], [75 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 10,
      i8* null
    }], section "llvm.metadata"

    define dso_local i32 @main() #0 {
      %1 = alloca float, align 4
      %2 = call float @fun()
      store float %2, float* %1, align 4
      ret i32 0
    }

    define dso_local float @fun() #0 {
      ret float 0x400921CAC0000000
    }
  )";

  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  GlobalVariable *globalVars = M->getGlobalVariable("llvm.global.annotations");
  ASSERT_NE(globalVars, nullptr);

  MultiValueMap<llvm::Value *, ValueInfo> variables;
  auto *anno = cast<ConstantStruct>(globalVars->getInitializer()->getOperand(0));
  auto *annoPtrInstr = cast<ConstantExpr>(anno->getOperand(1));
  auto *instr = cast<ConstantExpr>(anno->getOperand(0))->getOperand(0);
  bool startingPoint;
  bool res = initializer.parseAnnotation(variables, annoPtrInstr, instr, &startingPoint);

  ASSERT_TRUE(res);

  Value *first = variables.begin()->first;
  auto instructions = M->getFunction("main")->getBasicBlockList().begin()->getInstList().begin();
  auto *user = cast<llvm::Value>(instructions.operator++());
  ASSERT_EQ(first, user);
  checkMD(variables.begin()->second);
  EXPECT_TRUE(startingPoint);

  auto fun = M->getFunction("fun");
  EXPECT_TRUE(initializer.enabledFunctions.contains(fun));
  EXPECT_EQ(initializer.enabledFunctions.size(), 1);
}

TEST_F(AnnotationsTest, ParseAnnotation_LocalVariable)
{
  code = R"(
    @.str = private unnamed_addr constant [75 x i8] c"target('test') scalar(range(0, 10) type(1 2) error(3.1415) disabled final)\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"

    define dso_local i32 @main() #0 {
      %1 = alloca float, align 4
      %2 = bitcast float* %1 to i8*
      call void @llvm.var.annotation(
        i8* %2,
        i8* getelementptr inbounds ([75 x i8], [75 x i8]* @.str, i32 0, i32 0),
        i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
        i32 3,
        i8* null
      )
      ret i32 0
    }

  declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  auto instruction = M->getFunction("main")->getBasicBlockList().begin()->getInstList().begin();
  auto *user = cast<llvm::Value>(instruction); // the register %1
  instruction = instruction.operator++().operator++();

  // check that we're picking the right instruction
  auto *callInstr = cast<llvm::CallInst>(instruction);
  ASSERT_NE(callInstr, nullptr);
  ASSERT_NE(callInstr->getCalledFunction(), nullptr);

  MultiValueMap<llvm::Value *, ValueInfo> variables;
  auto *annoPtrInstr = cast<llvm::ConstantExpr>(instruction->getOperand(1));
  auto *instr = instruction->getOperand(0);
  bool startingPoint;
  bool res = initializer.parseAnnotation(variables, annoPtrInstr, instr, &startingPoint);

  ASSERT_TRUE(res);

  Value *first = variables.begin()->first;
  ASSERT_EQ(first, user);
  checkMD(variables.begin()->second);
  EXPECT_TRUE(startingPoint);
}


TEST_F(AnnotationsTest, ReadLocalAnnos_None)
{
  code = R"(
    @var = dso_local global float 0.000000e+00, align 4
    @.str = private unnamed_addr constant [75 x i8] c"target('test') scalar(range(0, 10) type(1 2) error(3.1415) disabled final)\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @llvm.global.annotations = appending global
    [1 x { i8*, i8*, i8*, i32, i8* }]
    [{ i8*, i8*, i8*, i32, i8* } {
      i8* bitcast (float* @var to i8*),
      i8* getelementptr inbounds ([75 x i8], [75 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 1,
      i8* null
    }], section "llvm.metadata"

    define dso_local i32 @main() #0 {
      ret i32 0
    }
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> variables;
  auto fun = M->getFunction("main");
  initializer.readLocalAnnotations(*fun, variables);
  EXPECT_EQ(variables.size(), 0);
}

TEST_F(AnnotationsTest, ReadLocalAnnos_MultipleAnnos)
{
  code = R"(
    @.str = private unnamed_addr constant [21 x i8] c"target('x') scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @.str.2 = private unnamed_addr constant [21 x i8] c"target('y') scalar()\00", section "llvm.metadata"

    define dso_local i32 @main() #0 {
    %1 = alloca float, align 4
    %2 = alloca i32, align 4
    %3 = bitcast float* %1 to i8*
    call void @llvm.var.annotation(
      i8* %3,
      i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 3,
      i8* null
    )
    %4 = bitcast i32* %2 to i8*
    call void @llvm.var.annotation(
      i8* %4,
      i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.2, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 4,
      i8* null
    )
    %5 = alloca float, align 4
    ret i32 0
    }

    declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> variables;
  auto fun = M->getFunction("main");
  initializer.readLocalAnnotations(*fun, variables);
  EXPECT_EQ(variables.size(), 2);
}

TEST_F(AnnotationsTest, ReadLocalAnnos_StartingPointSet)
{
  code = R"(
    @.str = private unnamed_addr constant [21 x i8] c"target('x') scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"

    define dso_local i32 @main() #0 {
    %1 = alloca float, align 4
    %2 = alloca i32, align 4
    %3 = bitcast float* %1 to i8*
    call void @llvm.var.annotation(
      i8* %3,
      i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 3,
      i8* null
    )
    ret i32 0
    }

    declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> variables;
  auto fun = M->getFunction("main");
  initializer.readLocalAnnotations(*fun, variables);
  auto md = fun->getMetadata("taffo.start");
  EXPECT_NE(md, nullptr);
}

TEST_F(AnnotationsTest, ReadLocalAnnos_StartingPointNotSet)
{
  code = R"(
    @.str = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"

    define dso_local i32 @main() #0 {
    %1 = alloca float, align 4
    %2 = alloca i32, align 4
    %3 = bitcast float* %1 to i8*
    call void @llvm.var.annotation(
      i8* %3,
      i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 3,
      i8* null
    )
    ret i32 0
    }

    declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> variables;
  auto fun = M->getFunction("main");
  initializer.readLocalAnnotations(*fun, variables);
  auto md = fun->getMetadata("taffo.start");
  EXPECT_EQ(md, nullptr);
}


TEST_F(AnnotationsTest, ReadAllLocalAnnos) {
  code = R"(
  @.str = private unnamed_addr constant [21 x i8] c"target('x') scalar()\00", section "llvm.metadata"
  @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
  @.str.2 = private unnamed_addr constant [21 x i8] c"target('y') scalar()\00", section "llvm.metadata"
  @.str.3 = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"

  ; Function Attrs: noinline nounwind optnone sspstrong uwtable
  define dso_local i32 @main() #0 {
  ret i32 0
  }

  ; Function Attrs: noinline nounwind optnone sspstrong uwtable
  define dso_local float @fun1() #0 {
  %1 = alloca float, align 4
  %2 = bitcast float* %1 to i8*
  call void @llvm.var.annotation(i8* %2, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 7, i8* null)
  %3 = load float, float* %1, align 4
  ret float %3
  }

  ; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
  declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1

  ; Function Attrs: noinline nounwind optnone sspstrong uwtable
  define dso_local i32 @fun2() #0 {
  %1 = alloca i32, align 4
  %2 = bitcast i32* %1 to i8*
  call void @llvm.var.annotation(i8* %2, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 12, i8* null)
  %3 = load i32, i32* %1, align 4
  ret i32 %3
  }

  ; Function Attrs: noinline nounwind optnone sspstrong uwtable
  define dso_local void @fun3() #0 {
  %1 = alloca double, align 8
  %2 = bitcast double* %1 to i8*
  call void @llvm.var.annotation(i8* %2, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 17, i8* null)
  ret void
  })";

  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> variables;
  initializer.readAllLocalAnnotations(*M, variables);

  for(auto &fun : M->functions()) {
    ASSERT_EQ(fun.getMetadata(Attribute::OptimizeNone), nullptr);
  }
}

TEST_F(AnnotationsTest, ReadAllLocalAnnos_MultipleStartingPoints) {
  code = R"(
  @.str = private unnamed_addr constant [21 x i8] c"target('x') scalar()\00", section "llvm.metadata"
  @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
  @.str.2 = private unnamed_addr constant [21 x i8] c"target('y') scalar()\00", section "llvm.metadata"

  define dso_local i32 @main() #0 {
  %1 = alloca float, align 4
  %2 = bitcast float* %1 to i8*
  call void @llvm.var.annotation(
    i8* %2,
    i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i32 0, i32 0),
    i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
    i32 3,
    i8* null
  )
  ret i32 0
  }

  declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1

  define dso_local i32 @fun2() #0 {
  %1 = alloca i32, align 4
  %2 = bitcast i32* %1 to i8*
  call void @llvm.var.annotation(
    i8* %2,
    i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.2, i32 0, i32 0),
    i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
    i32 7,
    i8* null
  )
  %3 = load i32, i32* %1, align 4
  ret i32 %3
  }
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> variables;
  initializer.readAllLocalAnnotations(*M, variables);

  auto md1 = M->getFunction("main")->getMetadata("taffo.start");
  EXPECT_NE(md1, nullptr);
  auto md2 = M->getFunction("fun2")->getMetadata("taffo.start");
  EXPECT_NE(md2, nullptr);
}


TEST_F(AnnotationsTest, RemoveNoFloat_BadInstr)
{
  /*
   * TODO: implement this test
   * (probably) not possible to have annotations on such instructions in
   * a realistic use case, since the __attribute directive can be applied
   * only to declarations, which are either alloca or global
   */
}

TEST_F(AnnotationsTest, RemoveNoFloat_AllocaFloat)
{
  code = R"(
    @.str = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"

    define dso_local i32 @main() #0 {
      %1 = alloca float, align 4
      %2 = bitcast float* %1 to i8*
      call void @llvm.var.annotation(
        i8* %2,
        i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0),
        i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
        i32 3,
        i8* null
      )
      ret i32 0
    }

    declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  auto instruction = M->getFunction("main")->getBasicBlockList().begin()->getInstList().begin();
  instruction = instruction.operator++().operator++();

  MultiValueMap<llvm::Value *, ValueInfo> variables;
  auto *annoPtrInstr = cast<llvm::ConstantExpr>(instruction->getOperand(1));
  auto *instr = instruction->getOperand(0);
  bool startingPoint;
  bool res = initializer.parseAnnotation(variables, annoPtrInstr, instr, &startingPoint);
  ASSERT_TRUE(res);

  EXPECT_EQ(variables.size(), 1);
  initializer.removeNoFloatTy(variables);
  EXPECT_EQ(variables.size(), 1);
}

TEST_F(AnnotationsTest, RemoveNoFloat_AllocaNoFloat)
{
  /*
   * TODO: test this case, it is not currently possible to test
   * however, the RemoveNoFloatTy function is not called on local variables (only on global ones)
   */
  code = R"(
    @.str = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"

    define dso_local i32 @main() #0 {
      %1 = alloca i32, align 4
      %2 = bitcast i32* %1 to i8*
      call void @llvm.var.annotation(
        i8* %2,
        i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0),
        i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
        i32 3,
        i8* null
      )
      ret i32 0
    }

    declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #1
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  auto instruction = M->getFunction("main")->getBasicBlockList().begin()->getInstList().begin();
  instruction = instruction.operator++().operator++();

  MultiValueMap<llvm::Value *, ValueInfo> variables;
  auto *annoPtrInstr = cast<llvm::ConstantExpr>(instruction->getOperand(1));
  auto *instr = instruction->getOperand(0);
  bool startingPoint;
  bool res = initializer.parseAnnotation(variables, annoPtrInstr, instr, &startingPoint);
  ASSERT_TRUE(res);

  EXPECT_EQ(variables.size(), 1);
  // initializer.removeNoFloatTy(variables);
  // EXPECT_EQ(variables.size(), 0);
}

TEST_F(AnnotationsTest, RemoveNoFloat_GlobalFloat)
{
  code = R"(
    @var = dso_local global float 0.000000e+00, align 4
    @.str = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @llvm.global.annotations = appending global
    [1 x { i8*, i8*, i8*, i32, i8* }]
    [{ i8*, i8*, i8*, i32, i8* } {
      i8* bitcast (float* @var to i8*),
      i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 1,
      i8* null
    }], section "llvm.metadata"

    define dso_local i32 @main() #0 {
      ret i32 0
    }
  )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  GlobalVariable *globalVars = M->getGlobalVariable("llvm.global.annotations");
  ASSERT_NE(globalVars, nullptr);

  MultiValueMap<Value *, ValueInfo> variables;
  auto *anno = cast<ConstantStruct>(globalVars->getInitializer()->getOperand(0));
  auto *annoPtrInstr = cast<ConstantExpr>(anno->getOperand(1));
  auto *instr = cast<ConstantExpr>(anno->getOperand(0))->getOperand(0);
  bool startingPoint;
  bool res = initializer.parseAnnotation(variables, annoPtrInstr, instr, &startingPoint);
  ASSERT_TRUE(res);

  EXPECT_EQ(variables.size(), 1);
  initializer.removeNoFloatTy(variables);
  EXPECT_EQ(variables.size(), 1);
}

TEST_F(AnnotationsTest, RemoveNoFloat_GlobalNoFloat)
{
  code = R"(
    @var = dso_local global i32 0, align 4
    @.str = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @llvm.global.annotations = appending global
    [1 x { i8*, i8*, i8*, i32, i8* }]
    [{ i8*, i8*, i8*, i32, i8* } {
      i8* bitcast (i32* @var to i8*),
      i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0),
      i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0),
      i32 1,
      i8* null
    }], section "llvm.metadata"

    define dso_local i32 @main() #0 {
      ret i32 0
    }
  )";
  std::unique_ptr<Module> M1 = makeLLVMModule(Context, code);
  std::unique_ptr<Module> M2 = makeLLVMModule(Context, code);

  MultiValueMap<llvm::Value *, ValueInfo> variables;
  MultiValueMap<llvm::Value *, ValueInfo> variables2;


  // cannot test the function directly
  initializer.readGlobalAnnotations(*M1, variables, false);
  EXPECT_EQ(variables.size(), 1);
  initializer.readGlobalAnnotations(*M2, variables2, true);
  EXPECT_EQ(variables2.size(), 0);
}


TEST_F(AnnotationsTest, ReadGlobalAnnotations_None)
{
  code = R"(
    define dso_local i32 @main() #0 {
        ret i32 0
    }
    )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> variables;

  initializer.readGlobalAnnotations(*M, variables, false);
  ASSERT_EQ(variables.size(), 0);
}

TEST_F(AnnotationsTest, ReadGlobalAnnotations_Variables)
{
  code = R"(
    @.str = private unnamed_addr constant [28 x i8] c"target('floatfun') scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @.str.2 = private unnamed_addr constant [26 x i8] c"target('intfun') scalar()\00", section "llvm.metadata"
    @.str.3 = private unnamed_addr constant [29 x i8] c"target('unusedfun') scalar()\00", section "llvm.metadata"
    @floatvar = dso_local global float 0.000000e+00, align 4
    @.str.4 = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"
    @intvar = dso_local global i32 0, align 4
    @.str.5 = private unnamed_addr constant [17 x i8] c"struct[scalar()]\00", section "llvm.metadata"
    @llvm.global.annotations = appending global [5 x { i8*, i8*, i8*, i32, i8* }] [{ i8*, i8*, i8*, i32, i8* } { i8* bitcast (float ()* @floatfun to i8*), i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 14, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (i32 ()* @intfun to i8*), i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 18, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (float ()* @unusedfun to i8*), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 22, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (float* @floatvar to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 1, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (i32* @intvar to i8*), i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 2, i8* null }], section "llvm.metadata"

    ; Function Attrs: noinline nounwind optnone sspstrong uwtable
    define dso_local i32 @main() #0 {
      %1 = alloca float, align 4
      %2 = alloca i32, align 4
      %3 = call float @floatfun()
      store float %3, float* %1, align 4
      %4 = call i32 @intfun()
      store i32 %4, i32* %2, align 4
      ret i32 0
    }

    ; Function Attrs: noinline nounwind optnone sspstrong uwtable
    define dso_local float @floatfun() #0 {
      ret float 0.000000e+00
    }

    ; Function Attrs: noinline nounwind optnone sspstrong uwtable
    define dso_local i32 @intfun() #0 {
      ret i32 0
    }

    ; Function Attrs: noinline nounwind optnone sspstrong uwtable
    define dso_local float @unusedfun() #0 {
      ret float 0.000000e+00
    }
    )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> queue;

  initializer.readGlobalAnnotations(*M, queue, false);
  ASSERT_EQ(queue.size(), 2);
  EXPECT_EQ(queue.count(M->getGlobalVariable("intvar")), 1);
  EXPECT_EQ(queue.count(M->getGlobalVariable("floatvar")), 1);

  // the check for the consistency of metadata is performed in the ParseAnnotatedVariable test case,
  // here we just check that the annotation is correct
  auto globalVar = queue.begin();
  EXPECT_EQ(globalVar->first->getName(), "floatvar");
  EXPECT_EQ(globalVar->second.metadata->toString(), "scalar()");
  globalVar++;
  EXPECT_EQ(globalVar->first->getName(), "intvar");
  EXPECT_EQ(globalVar->second.metadata->toString(), "struct(scalar())");
}

TEST_F(AnnotationsTest, ReadGlobalAnnotations_Functions)
{
  code = R"(
    @.str = private unnamed_addr constant [28 x i8] c"target('floatfun') scalar()\00", section "llvm.metadata"
    @.str.1 = private unnamed_addr constant [10 x i8] c"testing.c\00", section "llvm.metadata"
    @.str.2 = private unnamed_addr constant [26 x i8] c"target('intfun') scalar()\00", section "llvm.metadata"
    @.str.3 = private unnamed_addr constant [29 x i8] c"target('unusedfun') scalar()\00", section "llvm.metadata"
    @floatvar = dso_local global float 0.000000e+00, align 4
    @.str.4 = private unnamed_addr constant [9 x i8] c"scalar()\00", section "llvm.metadata"
    @intvar = dso_local global i32 0, align 4
    @.str.5 = private unnamed_addr constant [17 x i8] c"struct[scalar()]\00", section "llvm.metadata"
    @llvm.global.annotations = appending global [5 x { i8*, i8*, i8*, i32, i8* }] [{ i8*, i8*, i8*, i32, i8* } { i8* bitcast (float ()* @floatfun to i8*), i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 14, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (i32 ()* @intfun to i8*), i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 18, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (float ()* @unusedfun to i8*), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 22, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (float* @floatvar to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 1, i8* null }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (i32* @intvar to i8*), i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i32 0, i32 0), i32 2, i8* null }], section "llvm.metadata"

    define dso_local i32 @main() #0 {
      %1 = alloca float, align 4
      %2 = alloca i32, align 4
      %3 = call float @floatfun()
      store float %3, float* %1, align 4
      %4 = call i32 @intfun()
      store i32 %4, i32* %2, align 4
      ret i32 0
    }

    define dso_local float @floatfun() #0 {
      ret float 0.000000e+00
    }

    define dso_local i32 @intfun() #0 {
      ret i32 0
    }

    define dso_local float @unusedfun() #0 {
      ret float 0.000000e+00
    }
    )";
  std::unique_ptr<llvm::Module> M = makeLLVMModule(Context, code);
  MultiValueMap<Value *, ValueInfo> queue;

  // TODO: implement this test
}
} // namespace
