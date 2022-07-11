#include "LLVMFloatToFixedPass.h"
#include "ModuleCloneUtils.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

void fix_call(llvm::Function *function)
{
  if (!function) {
    return;
  }

  llvm::IRBuilder<> builder(function->getContext());
  //Find all call place
  for (auto caller : function->users()) {
    if (auto target_mapper_caller = llvm::dyn_cast<llvm::CallSite>(caller)) {
      //search for a call site as previous instruction
      if (auto prev_inst = llvm::dyn_cast<llvm::CallSite>(target_mapper_caller->getPrevNode())) {
        if (prev_inst->getCalledFunction()->getName().startswith("hero-openmp")) {

          auto call_site = llvm::dyn_cast<llvm::CallSite>(target_mapper_caller->getPrevNode());

          auto taffo_called_function = call_site->getCalledFunction();
          auto base_arg = target_mapper_caller->getOperand(4);
          auto not_base_arg = target_mapper_caller->getOperand(5);
          SmallVector<Instruction *> args;
          builder.SetInsertPoint(call_site);
          size_t i = 0;
          auto end_type = base_arg->getType()->getPointerElementType();
          for (auto &arg : taffo_called_function->args()) {
            auto arg_type = arg.getType();
            if (arg_type->isPointerTy()) {
              auto op_code = llvm::CastInst::getCastOpcode(&arg, true, end_type, true);
              auto tmp = llvm::CastInst::Create(op_code, &arg, end_type, "", call_site);
              builder.CreateStore(tmp, builder.CreateInBoundsGEP(base_arg, {0, i}));
              builder.CreateStore(tmp, builder.CreateInBoundsGEP(not_base_arg, {0, i}));
            } else {
              auto tmp = builder.CreateInBoundsGEP(base_arg, {0, i});
              tmp = builder.CreateBitOrPointerCast(tmp, arg_type->getPointerTo());
              builder.CreateStore(&arg, tmp);

              tmp = builder.CreateInBoundsGEP(base_arg, {0, i});
              tmp = builder.CreateBitOrPointerCast(tmp, arg_type->getPointerTo());
              builder.CreateStore(&arg, tmp);
            }
            i++;
          }
        }
      }
    }
  }
}


void flttofix::FloatToFixed::handleHero(llvm::Module &host_module)
{
  auto end_position = host_module.getModuleIdentifier().find("-host.ll");
  auto dev_name = host_module.getModuleIdentifier().substr(0, end_position) + "-dev.ll";
  auto dev_module = cloneModuleInto(dev_name, host_module, "__dev-");


  auto target_mapper = host_module.getFunction("__tgt_target_mapper");
  auto target_teams_mapper = host_module.getFunction("__tgt_target_teams_mapper");

  fix_call(target_mapper);
  fix_call(target_teams_mapper);
}