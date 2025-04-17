#ifndef __TAFFO_TEST_UTILS_H__
#define __TAFFO_TEST_UTILS_H__

#include "TaffoUtils/InputInfo.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

namespace taffo_test {

class Test : public testing::Test {
protected:
  llvm::LLVMContext Context;
  std::shared_ptr<llvm::Module> M;

  Test() { M = std::make_unique<llvm::Module>("test", Context); }
};

/// Creates a llvm::Module object starting from a LLVM-IR string.
std::unique_ptr<llvm::Module> makeLLVMModule(llvm::LLVMContext& Context, const std::string& code);

/// Creates a FatalErrorHandler that throws an exception instead of exiting.
void FatalErrorHandler(void* user_data, const char* reason, bool gen_crash_diag);

/**
 * Generates a InputInfo object
 * @param[in] min the Min value of the IRange
 * @param[in] max the Max value of the IRange
 * @param[in] isFinal
 * @return
 */
mdutils::InputInfo* genII(double min, double max, bool isFinal = false);

/**
 * Generates a Function
 * @param[in] M the Module to add the Function to
 * @param[in] retType
 * @param[in] params
 * @return
 */
llvm::Function* genFunction(llvm::Module& M, llvm::Type* retType, llvm::ArrayRef<llvm::Type*> params = {});

/**
 * Generates a Function
 * @param[in] M the Module to add the Function to
 * @param[in] name
 * @param[in] retType
 * @param[in] params
 * @return
 */
llvm::Function*
genFunction(llvm::Module& M, const std::string& name, llvm::Type* retType, llvm::ArrayRef<llvm::Type*> params = {});

llvm::GlobalVariable*
genGlobalVariable(llvm::Module& M, llvm::Type* T, llvm::Constant* init = nullptr, bool isConstant = false);

llvm::GlobalVariable* genGlobalVariable(llvm::Module& M, llvm::Type* T, int init, bool isConstant = false);

llvm::GlobalVariable* genGlobalVariable(llvm::Module& M, llvm::Type* T, double init, bool isConstant = false);

llvm::LoadInst* genLoadInstr(llvm::LLVMContext& Context);
} // namespace taffo_test

#endif
