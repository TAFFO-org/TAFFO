

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>
#include <utility>
#define DEBUG_TYPE "taffo-init"


static llvm::Function *recursive_clone_function_other_module(llvm::Module &M, llvm::Function *old_f, llvm::Twine prefix)
{

  if (auto func = M.getFunction((prefix + old_f->getName()).str())) {
    return func;
  }

  auto new_f = llvm::Function::Create(old_f->getFunctionType(), llvm::GlobalValue::ExternalWeakLinkage, prefix + old_f->getName(), &M);
  llvm::ValueToValueMapTy VMap{};
  llvm::SmallVector<llvm::ReturnInst *> Returns;
  auto old_arg = old_f->arg_begin();
  auto new_arg = new_f->arg_begin();
  for (; old_arg != old_f->arg_end(); old_arg++, new_arg++) {
    VMap[old_arg] = new_arg;
  }


  llvm::CloneFunctionInto(new_f, old_f, VMap, true, Returns);
  assert(new_f && "Clone Function info return nullptr");
  return new_f;
}


/**
 *Copy functions from hero dev module into hero host for info propagation
 **/
static void handleHero(llvm::Module &M, bool Hero)
{
  LLVM_DEBUG(llvm::dbgs() << "Handle Hero " << Hero << "\n");
  if (!Hero)
    return;


  llvm::SMDiagnostic diagnostic;
  auto dev_module = llvm::parseIRFile("3mm-dev.ll", diagnostic, M.getContext());

  auto target_mapper = M.getFunction("__tgt_target_mapper");
  auto target_teams_mapper = M.getFunction("__tgt_target_teams_mapper");

  // use given argument to find the name of the function to clone
  auto function_cloner_by_index = [&M, &dev_module](llvm::Function *function_to_clone, int arg_num) {
    if (function_to_clone != nullptr) {
      for (auto users : function_to_clone->users()) {
        if (auto call = llvm::dyn_cast<llvm::CallInst>(users)) {
          // in Target_mapper the 3 argument is typically name_of_loop_line.region_id
          LLVM_DEBUG(llvm::dbgs() << "TMP: " << *call << "\n");
          LLVM_DEBUG(llvm::dbgs() << "TMP: " << *call->getArgOperand(arg_num) << "\n");
          LLVM_DEBUG(llvm::dbgs() << "TMP: " << call->getArgOperand(arg_num)->getName() << "\n");
          auto name = call->getArgOperand(arg_num)->getName().split(".region_id").first.substr(1);
          LLVM_DEBUG(llvm::dbgs() << "Searching " << name << "\n");
          auto function_to_copy = dev_module->getFunction(name);
          assert(function_to_copy && "The function shuld exists ");
          auto new_f = recursive_clone_function_other_module(M, function_to_copy, "__dev-");
          LLVM_DEBUG(llvm::dbgs() << "Created " << *new_f << "\n");
        }
      }
    }
  };


  function_cloner_by_index(target_mapper, 2);
  function_cloner_by_index(target_teams_mapper, 2);
}
