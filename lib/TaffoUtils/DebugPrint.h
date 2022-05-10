#pragma once
#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#ifdef IPRINT


template <typename... Args>
static void wrapper_printf(llvm::IRBuilder<> &builder, llvm::Module &m, const llvm::Twine &str, Args... args)
{

  // %call3.flt = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i64 0, i64 0), double %6), !taffo.info !31, !taffo.

  auto &cntx = m.getContext();
  std::string function_name("printf");

  llvm::Function *print = nullptr;
  if ((print = m.getFunction(function_name)) == 0) {
    std::vector<llvm::Type *> fun_arguments;
    fun_arguments.push_back(
        llvm::Type::getInt8PtrTy(cntx)); // depends on your type
    llvm::FunctionType *fun_type = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(cntx), fun_arguments, true);
    print = llvm::Function::Create(fun_type, llvm::GlobalValue::ExternalLinkage,
                                   function_name, m);
  }

  auto constant_string = builder.CreateGlobalStringPtr(str.str());
  llvm::SmallVector<llvm::Value *, 4> small;

  small.insert(small.begin(), constant_string);
  small.insert(small.end(), {args...});

  builder.CreateCall(print, small);
}


static void addDebugBBPrint(llvm::Module &m)
{

  llvm::IRBuilder<> builder(m.getContext());
  for (auto &F : m) {
    if (F.isDeclaration()) {
      continue;
    }

    auto f_name = F.getName();
    //    llvm::dbgs() << f_name << "\n";
    builder.SetInsertPoint(F.begin()->getFirstNonPHI());
    wrapper_printf(builder, m, f_name + "\n");
    for (auto &BB : F) {
      builder.SetInsertPoint(BB.getFirstNonPHI());
      auto b_name = BB.getName();
      //      llvm::dbgs() << b_name << "\n";
      wrapper_printf(builder, m, "\t\t" + b_name + "\n");
    }
  }
}


#else


template <typename... Args>
static void wrapper_printf(llvm::IRBuilder<> &builder, llvm::Module &m, const llvm::Twine &str, Args... args)
{
}


static void addDebugBBPrint(llvm::Module &m)
{
}

#endif