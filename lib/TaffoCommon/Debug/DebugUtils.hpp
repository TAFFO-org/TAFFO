#pragma once

#include <llvm/ADT/Twine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

template <typename... Args>
void genPrintf(llvm::IRBuilder<>& builder, llvm::Module& m, const llvm::Twine& str, Args... args) {
  auto& ctx = m.getContext();
  std::string functionName("printf");

  llvm::Function* print = m.getFunction(functionName);
  if (!print) {
    llvm::SmallVector<llvm::Type*, 4> funArgs;
    funArgs.push_back(llvm::PointerType::get(ctx, 0));
    llvm::FunctionType* fun_type = llvm::FunctionType::get(llvm::Type::getInt32Ty(ctx), funArgs, true);
    print = llvm::Function::Create(fun_type, llvm::GlobalValue::ExternalLinkage, functionName, m);
  }

  auto stringConstant = builder.CreateGlobalStringPtr(str.str());
  llvm::SmallVector<llvm::Value*, 4> callArgs;
  callArgs.insert(callArgs.begin(), stringConstant);
  callArgs.insert(callArgs.end(), {args...});
  builder.CreateCall(print, callArgs);
}

int write_module(const std::string& fileName, const llvm::Module& m);
