#include "ModuleCloneUtils.h"
#include "OpenMPAnalyzer.h"
#include "TaffoInitializerPass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cassert>
#include <utility>
#define DEBUG_TYPE "taffo-init"


//Try to remove as much cast as possible from openmp region handoff
llvm::Function *create_function_less_cast(llvm::Function *function_to_clone)
{


  //Find the "real" type of the openmp region
  llvm::SmallVector<llvm::Type *, 3> new_types;
  bool found;
  for (const auto &arg : function_to_clone->args()) {
    found = false;
    for (const auto &user : arg.users()) {
      if (!llvm::isa<llvm::CastInst>(user) && !llvm::isa<llvm::StoreInst>(user))
        continue;
      if (llvm::isa<llvm::StoreInst>(user)) {
        auto store_inst = llvm::cast<llvm::StoreInst>(user);
        for (auto store_users : store_inst->getPointerOperand()->users()) {
          if (auto cast = llvm::dyn_cast<llvm::CastInst>(store_users)) {
            new_types.push_back(cast->getDestTy());
            found = true;
            break;
          }
        }
      }

      if (llvm::isa<llvm::CastInst>(user)) {
        auto cast_inst = llvm::cast<llvm::CastInst>(user);
        new_types.push_back(cast_inst->getDestTy());
        found = true;
        break;
      }
    }
    if (found == false) {
      new_types.push_back(arg.getType());
    }
  }


  auto function_cloned = llvm::Function::Create(llvm::FunctionType::get(function_to_clone->getReturnType(), new_types, false),
                                                function_to_clone->getLinkage(), "hero-openmp-target-region", function_to_clone->getParent());
  llvm::ValueToValueMapTy VMap;
  for (size_t i = 0; i < function_to_clone->arg_size(); ++i) {
    VMap[function_to_clone->getArg(i)] = function_cloned->getArg(i);
  }


  llvm::SmallVector<llvm::ReturnInst *, 3> ret;
  llvm::CloneFunctionInto(function_cloned, function_to_clone, VMap, false, ret, "");


  //Remove immediate bit cast

  for (auto &arg : function_cloned->args()) {
    for (auto user : make_early_inc_range(arg.users())) {
      if (!llvm::isa<llvm::CastInst>(user))
        continue;

      auto cast_inst = llvm::cast<llvm::CastInst>(user);
      if (cast_inst->getSrcTy() == cast_inst->getDestTy()) {
        cast_inst->replaceAllUsesWith(cast_inst->getOperand(0));
        cast_inst->eraseFromParent();
      }
    }
  }


  //Remove store alloca pattern bit cast

  for (auto &arg : function_cloned->args()) {
    for (auto user : make_early_inc_range(arg.users())) {
      if (!llvm::isa<llvm::StoreInst>(user))
        continue;

      auto store_inst = llvm::cast<llvm::StoreInst>(user);
      for (auto store_users : make_early_inc_range(store_inst->getPointerOperand()->users()))
        if (auto cast_inst = llvm::dyn_cast<llvm::CastInst>(store_users)) {
          cast_inst->replaceAllUsesWith(&arg);
          cast_inst->eraseFromParent();
          store_inst->eraseFromParent();
        }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Inserted new function: "
                          << *function_cloned->getFunctionType() << "\n");

  return function_cloned;
}


llvm::Value *find_original_value(llvm::Value *StoredValues, llvm::StoreInst *lastAccess, llvm::MemorySSA &Mssa)
{

  auto walker = Mssa.getSkipSelfWalker();
  llvm::MemoryAccess *clobber;
  if (auto load = llvm::dyn_cast<LoadInst>(StoredValues)) {
    llvm::dbgs()
        << "\n\nClobber di " << *load << " \n";
    clobber = walker->getClobberingMemoryAccess(Mssa.getMemoryAccess(load));

  } else {
    llvm::dbgs()
        << "\n\nClobber di " << *lastAccess << " \n";
    clobber = walker->getClobberingMemoryAccess(lastAccess);
  }


  while (!Mssa.isLiveOnEntryDef(clobber)) {

    if (auto stampa = dyn_cast<llvm::MemoryUseOrDef>(clobber)) {
      auto inst = stampa->getMemoryInst();
      llvm::dbgs() << "\t\t-->  " << *inst << "\n";
      if (auto store_inst = llvm::dyn_cast<StoreInst>(inst)) {
        llvm::dbgs() << "Found: " << *store_inst << "\n";
        return store_inst->getValueOperand();
      }

      clobber = walker->getClobberingMemoryAccess(stampa->getDefiningAccess(), MemoryLocation::get(llvm::dyn_cast<LoadInst>(StoredValues)));
    } else {
      llvm::dbgs() << "\t\t-->" << *cast<MemoryPhi>(clobber) << "\n";
    }
  }
  llvm::dbgs() << "\t\t-->  " << *dyn_cast<llvm::MemoryUseOrDef>(clobber) << "\n\n\n";
  llvm_unreachable("Not found original value");
}

/// Create a call to \p function_to_call inserting it before the \p insert_point
/// Create appropriate cast from value stored in \p OAs and argument of the \p function_to_call
void insert_call_to_new_function(llvm::Function *function_to_call, OffloadArray *OAs, llvm::Instruction *insert_point, llvm::MemorySSA &Mssa)
{
  llvm::IRBuilder<> builder(function_to_call->getContext());


  llvm::SmallVector<llvm::Value *, 3> args;
  for (size_t i = 0; i < function_to_call->arg_size(); ++i) {
    auto value_to_pass = OAs[0].StoredValues[i];
    auto last_access = OAs[0].LastAccesses[i];

    if (llvm::isa<LoadInst>(value_to_pass)) {
      value_to_pass = find_original_value(value_to_pass, last_access, Mssa);
      auto &first_BB = *insert_point->getFunction()->begin();
      llvm::IRBuilder<> builder(insert_point);
      llvm::Instruction *generic = builder.CreateAlloca(value_to_pass->getType());
      LLVM_DEBUG(llvm::dbgs() << "Create new alloca " << *generic << "\n");
      generic = builder.CreateStore(value_to_pass, generic);
      LLVM_DEBUG(llvm::dbgs() << "Create new store " << *generic << "\n");
      value_to_pass = llvm::cast<llvm::StoreInst>(generic)->getPointerOperand();
    }

    auto argument = function_to_call->getArg(i);
    auto start_type = value_to_pass->getType();
    auto end_type = argument->getType();


    if (start_type != end_type) {
      if (start_type->isPointerTy() && !end_type->isPointerTy()) {
        builder.SetInsertPoint(insert_point);
        value_to_pass = builder.CreateLoad(value_to_pass);
        start_type = value_to_pass->getType();
      }
      LLVM_DEBUG(llvm::dbgs() << "Create cast from " << *start_type << " to " << *end_type << "\n");
      auto op_code = llvm::CastInst::getCastOpcode(value_to_pass, true, end_type, true);
      auto tmp = llvm::CastInst::Create(op_code, value_to_pass, end_type, "", insert_point);
      LLVM_DEBUG(llvm::dbgs() << "Cast created: \n\t" << *tmp << "\n");
      args.push_back(tmp);

    } else {
      args.push_back(value_to_pass);
    }
  }

  llvm::CallInst::Create(function_to_call, args, "", insert_point);
}


namespace taffo
{


/**
 *Copy functions from hero dev module into hero host for info propagation
 **/
void TaffoInitializer::handleHero(llvm::Module &host_module, bool Hero)
{

  if (!Hero)
    return;

  LLVM_DEBUG(llvm::dbgs() << "\n##### Handle Init Hero ######\n");
  auto end_position = host_module.getModuleIdentifier().find("-host.ll");
  auto dev_name = host_module.getModuleIdentifier().substr(0, end_position) + "-dev.ll";
  auto dev_module = cloneModuleInto(dev_name, host_module, "__dev-");


  auto target_mapper = host_module.getFunction("__tgt_target_mapper");
  auto target_teams_mapper = host_module.getFunction("__tgt_target_teams_mapper");

  // use given argument to find the name of the function to clone
  auto function_cloner_by_index = [&host_module, this](llvm::Function *function_to_inline, int arg_num) {
    if (function_to_inline != nullptr) {
      for (auto users : function_to_inline->users()) {
        if (auto call = llvm::dyn_cast<llvm::CallInst>(users)) {

          // in Target_mapper the 3 argument is typically name_of_loop_line.region_id
          auto name = "__dev-" + call->getArgOperand(arg_num)->getName().split(".region_id").first.substr(1);

          LLVM_DEBUG(llvm::dbgs() << "Searching " << name << "\n");
          auto function_to_call = host_module.getFunction(name.str());
          assert(function_to_call && "The function shuld exists ");


          OffloadArray OAs[3];
          {
            auto ret = getValuesInOffloadArrays(*call, OAs);
            if (!ret) {

              LLVM_DEBUG(llvm::dbgs() << "No information on types for " << *call << "\n");
              assert(ret && "we must have information on types");
            }
          }


          LLVM_DEBUG(llvm::dbgs() << "Founded!\n\tPassed values:\n");
          for (const auto &stored_value : OAs[0].StoredValues) {
            LLVM_DEBUG(llvm::dbgs() << "\t\t" << *stored_value << "\n");
          }

          //Create a new function with less cast
          function_to_call = create_function_less_cast(function_to_call);

          auto &Mssa = this->getAnalysis<llvm::MemorySSAWrapperPass>(*(call->getFunction())).getMSSA();
          //create a call to the new function
          insert_call_to_new_function(function_to_call, OAs, call, Mssa);
        }
      }
    }
  };


  function_cloner_by_index(target_mapper, 2);
  function_cloner_by_index(target_teams_mapper, 2);


  LLVM_DEBUG(llvm::dbgs() << "##### End Init Hero ######\n\n");
}


} // namespace taffo