#include "TestUtils.h"

using namespace taffo_test;

/// Creates a llvm::Module object starting from a LLVM-IR string.
std::unique_ptr<llvm::Module> taffo_test::makeLLVMModule(llvm::LLVMContext &Context, const std::string &code)
{
  llvm::StringRef ModuleStr(code);
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad LLVM IR?");
  return M;
}

/// Creates a FatalErrorHandler that throws an exception instead of exiting.
void taffo_test::FatalErrorHandler(void *user_data, const char *reason, bool gen_crash_diag)
{
  throw std::runtime_error(reason);
}

/**
 * Generates a InputInfo object
 * @param[in] min the Min value of the IRange
 * @param[in] max the Max value of the IRange
 * @param[in] isFinal
 * @return
 */
mdutils::InputInfo *taffo_test::genII(double min, double max, bool isFinal)
{
  return new mdutils::InputInfo(nullptr, std::make_shared<mdutils::Range>(min, max), nullptr, false, isFinal);
}

/**
 * Generates a Function
 * @param[in] M the Module to add the Function to
 * @param[in] retType
 * @param[in] params
 * @return
 */
llvm::Function *taffo_test::genFunction(llvm::Module &M, llvm::Type *retType, llvm::ArrayRef<llvm::Type *> params)
{
  llvm::FunctionType *FT = llvm::FunctionType::get(retType, params, false);
  llvm::Function *F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, "fun", &M);
  return F;
}

/**
 * Generates a Function
 * @param[in] M the Module to add the Function to
 * @param[in] name
 * @param[in] retType
 * @param[in] params
 * @return
 */
llvm::Function *taffo_test::genFunction(llvm::Module &M, const std::string &name, llvm::Type *retType, llvm::ArrayRef<llvm::Type *> params)
{
  llvm::FunctionType *FT = llvm::FunctionType::get(retType, params, false);
  llvm::Function *F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, name, &M);
  return F;
}


llvm::GlobalVariable *taffo_test::genGlobalVariable(llvm::Module &M, llvm::Type *T, llvm::Constant *init, bool isConstant)
{
  return new llvm::GlobalVariable(M, T, isConstant, llvm::GlobalValue::ExternalLinkage, init, "var");
}

llvm::GlobalVariable *taffo_test::genGlobalVariable(llvm::Module &M, llvm::Type *T, int init, bool isConstant)
{
  if (T->isIntegerTy())
    return genGlobalVariable(M, T, llvm::ConstantInt::get(T, init), isConstant);
  llvm::dbgs() << "Type and initial value not compatible\n";
  return nullptr;
}

llvm::GlobalVariable *taffo_test::genGlobalVariable(llvm::Module &M, llvm::Type *T, double init, bool isConstant)
{
  if (T->isFloatTy() || T->isDoubleTy())
    return genGlobalVariable(M, T, llvm::ConstantFP::get(T, init), isConstant);
  llvm::dbgs() << "Type and initial value not compatible\n";
  return nullptr;
}

llvm::LoadInst *taffo_test::genLoadInstr(llvm::LLVMContext &Context)
{
  std::string code = R"(
    define i32 @main() {
      %a = alloca float, align 4
      %b = load float, float* %a, align 4
      ret i32 0
    }
  )";

  auto M = makeLLVMModule(Context, code);
  auto F = M->getFunction("main");
  auto load = F->getEntryBlock().getFirstNonPHI()->getNextNode();
  if (llvm::isa<llvm::LoadInst>(load))
    return llvm::cast<llvm::LoadInst>(load);
  llvm_unreachable("genLoadInstr failed");
}
