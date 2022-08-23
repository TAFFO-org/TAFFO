#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

/// Creates a llvm::Module object starting from a LLVM-IR string.
static std::unique_ptr<llvm::Module> makeLLVMModule(llvm::LLVMContext &Context, const std::string &code)
{
  llvm::StringRef ModuleStr(code);
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad LLVM IR?");
  return M;
}

/// Creates a FatalErrorHandler that throws an exception instead of exiting.
static void FatalErrorHandler(void *user_data, const std::string &reason, bool gen_crash_diag)
{
  throw std::runtime_error(reason.c_str());
}

/**
 * Generates a InputInfo object
 * @param[in] min the Min value of the IRange
 * @param[in] max the Max value of the IRange
 * @param[in] isFinal
 * @return
 */
static mdutils::InputInfo*genII(double min, double max, bool isFinal=false) {
  return new mdutils::InputInfo(nullptr, std::make_shared<mdutils::Range>(min, max), nullptr, false, isFinal);
}

/**
 * Generates a Function
 * @param[in] M the Module to add the Function to
 * @param[in] retType
 * @param[in] params
 * @return
 */
static llvm::Function* genFunction(llvm::Module &M, llvm::Type* retType, llvm::ArrayRef<llvm::Type*> params={}) {
  llvm::FunctionType* FT = llvm::FunctionType::get(retType, params, false);
  llvm::Function* F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, "fun", &M);
  return F;
}

static llvm::GlobalVariable* genGlobalVariable(llvm::Module &M, llvm::Type *T, llvm::Constant *init =nullptr, bool isConstant=false) {
  return new llvm::GlobalVariable(M, T, isConstant, llvm::GlobalValue::ExternalLinkage, init, "var");
}

static llvm::GlobalVariable* genGlobalVariable(llvm::Module &M, llvm::Type *T, int init, bool isConstant=false) {
  if (T->isIntegerTy())
    return genGlobalVariable(M, T, llvm::ConstantInt::get(T, init), isConstant);
  llvm::dbgs() << "Type and initial value not compatible\n";
  return nullptr;
}

static llvm::GlobalVariable* genGlobalVariable(llvm::Module &M, llvm::Type *T, double init, bool isConstant=false) {
  if (T->isFloatTy() || T->isDoubleTy())
    return genGlobalVariable(M, T, llvm::ConstantFP::get(T, init), isConstant);
  llvm::dbgs() << "Type and initial value not compatible\n";
  return nullptr;
}
