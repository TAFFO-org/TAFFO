#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"


/// Creates a llvm::Module object starting from a LLVM-IR string.
static std::unique_ptr<llvm::Module> makeLLVMModule(llvm::LLVMContext &Context, const std::string &code) {
  llvm::StringRef ModuleStr(code);
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad LLVM IR?");
  return M;
}

/// Creates a FatalErrorHandler that throws an exception instead of exiting.
static void FatalErrorHandler(void *user_data, const std::string &reason, bool gen_crash_diag) {
  throw std::runtime_error(reason.c_str());
}

