#include "CallSiteVersions.h"
#include "LLVMFloatToFixedPass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <tuple>
#include <utility>


SmallVector<llvm::CallSite *> fix_call(llvm::Function *function)
{
  SmallVector<llvm::CallSite *> ret{};
  if (!function) {
    return ret;
  }


  //Find all call place
  for (auto caller : function->users()) {
    if (auto target_mapper_caller = llvm::dyn_cast<llvm::CallSite>(caller)) {
      //search for a call site as previous instruction
      if (auto prev_inst = llvm::dyn_cast<llvm::CallSite>(target_mapper_caller->getPrevNode())) {
        if (prev_inst->getCalledFunction()->getName().startswith("hero-openmp")) {

          LLVM_DEBUG(llvm::dbgs() << "In " << prev_inst->getParent()->getParent()->getName() << "\nfound " << *caller << "\nwith predecessor " << *prev_inst << "\n");

          llvm::IRBuilder<> builder(function->getContext());
          auto call_site = prev_inst;
          builder.SetInsertPoint(call_site);
          auto base_arg = llvm::cast<GetElementPtrInst>(target_mapper_caller->getOperand(4))->getOperand(0);
          auto not_base_arg = llvm::cast<GetElementPtrInst>(target_mapper_caller->getOperand(5))->getOperand(0);

          size_t i = 0;

          for (auto &op : call_site->operands()) {
            auto &arg = *llvm::cast<llvm::Value>(op.get());
            auto arg_type = arg.getType();
            auto base_gep = builder.CreateGEP(base_arg, {builder.getInt32(0), builder.getInt32(i)});
            auto not_base_gep = builder.CreateGEP(not_base_arg, {builder.getInt32(0), builder.getInt32(i)});
            auto end_type = base_gep->getType()->getPointerElementType();

            if (arg_type->isPointerTy()) {

              LLVM_DEBUG(llvm::dbgs() << "Create a casting pointer " << *arg_type << " to pointer " << *end_type << " and store it\n");

              auto op_code = llvm::CastInst::getCastOpcode(&arg, true, end_type, true);
              LLVM_DEBUG(llvm::dbgs() << "\tcode: \n");
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *base_gep << "\n");
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *not_base_gep << "\n");
              auto tmp = llvm::CastInst::Create(op_code, &arg, end_type, "", call_site);
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *tmp << "\n");
              auto dbg = builder.CreateStore(tmp, base_gep);
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *dbg << "\n");
              dbg = builder.CreateStore(tmp, not_base_gep);
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *dbg << "\n");
            } else {


              auto tmp = builder.CreateBitOrPointerCast(base_gep, arg_type->getPointerTo());
              LLVM_DEBUG(llvm::dbgs() << "store the value " << *arg_type << " in the pointer " << *tmp << "\n");
              LLVM_DEBUG(llvm::dbgs() << "\tcode: \n");
              auto dbg = builder.CreateStore(&arg, tmp);
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *dbg << "\n");


              tmp = builder.CreateBitOrPointerCast(not_base_gep, arg_type->getPointerTo());
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *tmp << "\n");
              dbg = builder.CreateStore(&arg, tmp);
              LLVM_DEBUG(llvm::dbgs() << "\t\t" << *dbg << "\n");
            }
            i++;
          }
          ret.emplace_back(call_site);
        }
      }
    }
  }
}


void export_functions(SmallVectorImpl<llvm::CallSite *> &functions_to_export, StringRef filename, llvm::LLVMContext &cntx)
{

  IRBuilder<> builder(cntx);

  if (filename.rfind('/') != llvm::StringRef::npos) {
    filename = filename.rsplit('/').second;
  }
  {
    auto dot_ll_position = filename.find(".ll");
    filename = filename.slice(0, dot_ll_position + 3);
  }
  llvm::SMDiagnostic diagnostic;


  auto dev_module = llvm::parseIRFile(filename, diagnostic, cntx);
  auto identifiers = dev_module->getIdentifiedStructTypes();
  auto type_tgt_offload_entry = llvm::find_if(identifiers, [](llvm::StructType *elem) {
    return elem->getName().startswith("struct.__tgt_offload_entry");
  });
  assert(type_tgt_offload_entry != identifiers.end() && "Not found type");

  (*type_tgt_offload_entry)->dump();


  assert(dev_module && "Cannot retrive module");
  for (auto function_to_export_call : functions_to_export) {
    auto function_to_export = function_to_export_call->getCalledFunction();
    auto function_type = function_to_export->getFunctionType();
    auto function_name = function_to_export->getName();
    auto new_dev_callee = dev_module->getOrInsertFunction(function_name, function_type);

    if (auto new_function = llvm::dyn_cast<llvm::Function>(new_dev_callee.getCallee())) {
      llvm::ValueToValueMapTy Vmap{};
      for (auto args : zip(function_to_export->args(), new_function->args())) {
        Vmap.insert({&std::get<0>(args), &std::get<1>(args)});
      }
      SmallVector<llvm::ReturnInst *> returns;
      CloneFunctionInto(new_function, function_to_export, Vmap, true, returns);
      new_function->setLinkage(llvm::GlobalValue::WeakAnyLinkage);
      auto name_in_dev = builder.CreateGlobalStringPtr(function_name, ".omp_offloading.entry_name", 0, &*dev_module);
      Constant *global_value_values[5];

      global_value_values[0] = ConstantExpr::getBitCast(new_function, Type::getInt8PtrTy(cntx));

      global_value_values[1] = ConstantExpr::getBitCast(name_in_dev, Type::getInt8PtrTy(cntx));
      global_value_values[2] = builder.getInt32(0);
      global_value_values[3] = builder.getInt32(0);
      global_value_values[4] = builder.getInt32(0);

      auto tmp = ConstantStruct::get(*type_tgt_offload_entry, global_value_values);
      new GlobalVariable(*dev_module, *type_tgt_offload_entry, true, llvm::GlobalValue::WeakAnyLinkage, tmp, ".omp_offloading.entry." + function_name);
      auto new_region_id = new GlobalVariable(*(function_to_export->getParent()), Type::getInt8Ty(cntx), true, llvm::GlobalValue::WeakAnyLinkage, builder.getInt8(0), "." + function_name + ".region_id");
      function_to_export_call->getNextNode()->setOperand(2, new_region_id);
    }
  }


  verifyModule(*dev_module);
  int file = 0;
  auto err = llvm::sys::fs::openFileForWrite(filename, file);
  assert(!err.value() && "Fail open module");
  raw_fd_ostream stream{file, false};
  dev_module->print(stream, nullptr);

  llvm::sys::fs::closeFile(file);
}

void flttofix::FloatToFixed::handleHero(llvm::Module &host_module)
{
  LLVM_DEBUG(llvm::dbgs() << "\n##### Handle Init Hero ######\n");
  auto end_position = host_module.getModuleIdentifier().find("-host.ll");
  auto dev_name = host_module.getModuleIdentifier().substr(0, end_position) + "-dev.ll";


  auto target_mapper = host_module.getFunction("__tgt_target_mapper");
  auto target_teams_mapper = host_module.getFunction("__tgt_target_teams_mapper");

  auto functions_to_export = fix_call(target_mapper);
  functions_to_export.append(fix_call(target_teams_mapper));

  export_functions(functions_to_export, dev_name, host_module.getContext());

  LLVM_DEBUG(llvm::dbgs() << "##### End Init Hero ######\n\n");
}