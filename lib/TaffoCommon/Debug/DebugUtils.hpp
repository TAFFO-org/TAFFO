#pragma once

#include <llvm/IR/IRBuilder.h>

#ifndef NDEBUG

#define IF_TAFFO_DEBUG if (::llvm::DebugFlag && ::llvm::isCurrentDebugType(DEBUG_TYPE))

#else

#define IF_TAFFO_DEBUG if (false)

#endif

template <typename... Args>
void genPrintf(llvm::IRBuilder<>& builder, llvm::Module& m, const llvm::Twine& str, Args... args) {
  // %call3.flt = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i64 0, i64 0),
  // double %6), !taffo.info !31, !taffo.
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
