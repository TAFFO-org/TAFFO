#include "CallSiteVersions.h"
#include "LLVMFloatToFixedPass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cstddef>
#include <deque>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>


class ident_struct_type_remapper : public ValueMapTypeRemapper
{

  std::unordered_map<Type *, Type *> structed_map;

public:
  ident_struct_type_remapper(const std::vector<StructType *> &host_list_struct, const std::vector<StructType *> &dev_struct_list)
  {
    auto dev_struct = std::find_if(dev_struct_list.begin(), dev_struct_list.end(), [](StructType *lrs) { return lrs->getName().startswith("struct.ident_t"); });
    if (dev_struct == dev_struct_list.end()) {
      llvm_unreachable("Non c'era :(");
      return;
    }


    for (auto &host_struct : host_list_struct) {
      if (host_struct->getName().startswith("struct.ident_t")) {
        LLVM_DEBUG(llvm::dbgs() << "Insert {" << host_struct->getName() << ", " << (*dev_struct)->getName() << " }\n");
        structed_map.insert({host_struct, *dev_struct});
      }
    }
  }
  Type *remapType(Type *SrcTy) override
  {
    auto find = structed_map.find(SrcTy);
    if (find != structed_map.end()) {
      return find->second;
    }
    return SrcTy;
  }
};


//Clone all global but not the one that start with hero as they are target dependant (contains assembly code)
void cloneGlobalVariable(llvm::Module &dev, llvm::Module &host, llvm::ValueToValueMapTy &GtoG)
{
  LLVM_DEBUG(llvm::dbgs() << "START " << __PRETTY_FUNCTION__ << "\n");

  using namespace llvm;
  //Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //


  std::unique_ptr<ValueMapTypeRemapper> remapper = std::make_unique<ident_struct_type_remapper>(host.getIdentifiedStructTypes(), dev.getIdentifiedStructTypes());
  std::unordered_set<Function *> alredy_present_function;

  for (const llvm::GlobalVariable &I : host.globals()) {
    if (!(I.getName().startswith("__dev") || I.getName().startswith("__tgt_offload_entry")))
      continue;


    StringRef old_name = I.getName();
    if (I.getName().startswith("__dev"))
      old_name = I.getName().substr(6, std::string::npos);
    LLVM_DEBUG(llvm::dbgs() << "Searching global: " << I.getName() << " as " << old_name << "\n");
    if (auto glob = dev.getNamedGlobal(old_name)) {
      LLVM_DEBUG(llvm::dbgs() << "Found global: " << *glob << "\n");
      GtoG[&I] = glob;
    } else {

      llvm::GlobalVariable *NewGV = new GlobalVariable(
          dev, remapper->remapType(I.getValueType()), I.isConstant(), I.getLinkage(),
          (llvm::Constant *)nullptr, old_name, (llvm::GlobalVariable *)nullptr,
          I.getThreadLocalMode(), I.getType()->getAddressSpace());
      NewGV->copyAttributesFrom(&I);
      LLVM_DEBUG(llvm::dbgs() << "Created global: " << *NewGV << "\n");

      GtoG[&I] = NewGV;
    }
  }


  // Loop over the functions in the module, making external functions as before
  for (const Function &I : host) {
    if (!(I.getName().startswith("__dev") || I.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99")))
      continue;
    auto old_name = I.getName();
    if (I.getName().startswith("__dev")) {
      old_name = old_name.substr(6, std::string::npos);
    }
    LLVM_DEBUG(llvm::dbgs() << "Search function: " << I.getName() << " as " << old_name << "\n");
    if (auto old = dev.getFunction(old_name)) {
      LLVM_DEBUG(llvm::dbgs() << "Founded\n");
      GtoG[&I] = old;
      alredy_present_function.insert(old);
    } else {
      Function *NF = Function::Create(cast<FunctionType>(I.getValueType()), I.getLinkage(),
                                      I.getAddressSpace(), old_name, &dev);
      NF->copyAttributesFrom(&I);
      LLVM_DEBUG(llvm::dbgs() << "Created function: " << NF->getName() << "\n");
      GtoG[&I] = NF;
    }
  }


  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (const GlobalVariable &G : host.globals()) {
    if (G.getName().startswith("llvm.used"))
      continue;
    if (!(G.getName().startswith("__dev") || G.getName().startswith("__tgt_offload_entry")))
      continue;

    GlobalVariable *GV = cast<GlobalVariable>(GtoG[&G]);
    if (GV->hasInitializer()) {
      continue;
    }


    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    G.getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(MD.first, *MapMetadata(MD.second, GtoG));

    if (G.isDeclaration())
      continue;
    if (G.hasInitializer())
      GV->setInitializer(MapValue(G.getInitializer(), GtoG, llvm::RF_None, remapper.get()));
  }

  // Similarly, copy over function bodies now...

  for (const Function &I : host) {
    if (!(I.getName().startswith("__dev") || I.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99")))
      continue;
    Function *F = cast<Function>(GtoG[&I]);
    if (!F->isDeclaration()) {
      continue;
    }


    if (alredy_present_function.find(F) != alredy_present_function.end())
      continue;

    if (I.isDeclaration()) {
      // Copy over metadata for declarations since we're not doing it below in
      // CloneFunctionInto().
      SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
      I.getAllMetadata(MDs);
      for (auto MD : MDs)
        F->addMetadata(MD.first, *MapMetadata(MD.second, GtoG));
      continue;
    }

    Function::arg_iterator DestI = F->arg_begin();
    for (const Argument &J : I.args()) {
      DestI->setName(J.getName());
      GtoG[&J] = &*DestI++;
    }

    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(F, &I, GtoG, true,
                      Returns, "", nullptr, remapper.get());

    if (I.hasPersonalityFn())
      F->setPersonalityFn(MapValue(I.getPersonalityFn(), GtoG));
  }


  // // And named metadata....
  // for (const NamedMDNode &NMD : dev_module.named_metadata()) {
  //   NamedMDNode *NewNMD = host_module.getOrInsertNamedMetadata(NMD.getName());
  //   for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
  //     NewNMD->addOperand(MapMetadata(NMD.getOperand(i), GtoG));
  // }

  llvm::dbgs() << "Pizza\n";
  ValueMapper VM(GtoG, llvm::RF_None | llvm::RF_None, remapper.get());
  if (auto kmpc = dev.getFunction("__kmpc_fork_call"))
    for (auto call_user : make_early_inc_range(kmpc->users())) {
      if (auto call_inst = llvm::dyn_cast<CallInst>(call_user)) {
        std::vector<Value *> types;
        for (auto &op : call_inst->operands()) {
          types.push_back(op.get());
        }

        CallInst::Create(call_inst->getCalledFunction(), types, None, "", call_inst);
        call_inst->eraseFromParent();
      }
    }
  LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
}


SmallVector<llvm::CallSite *> fix_call(llvm::Function *function)
{
  LLVM_DEBUG(llvm::dbgs() << "START " << __PRETTY_FUNCTION__ << "\n");
  SmallVector<llvm::CallSite *> ret{};
  if (!function) {
    LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
    return ret;
  }


  //Find all call place
  for (auto caller : function->users()) {
    if (auto target_mapper_caller = llvm::dyn_cast<llvm::CallSite>(caller)) {
      //search for a call site as previous instruction
      if (auto prev_inst = llvm::dyn_cast<llvm::CallSite>(target_mapper_caller->getPrevNode())) {
        if (prev_inst->getCalledFunction()->getName().startswith("__omp_offloading_10303_4dc05d6_func_l99")) {

          LLVM_DEBUG(llvm::dbgs() << "In " << prev_inst->getParent()->getParent()->getName() << "\nfound " << *caller << "\nwith predecessor " << *prev_inst << "\n");

          llvm::IRBuilder<> builder(function->getContext());
          auto call_site = prev_inst;
          builder.SetInsertPoint(call_site);
          auto base_arg = llvm::cast<GetElementPtrInst>(target_mapper_caller->getOperand(4))->getOperand(0);
          auto not_base_arg = llvm::cast<GetElementPtrInst>(target_mapper_caller->getOperand(5))->getOperand(0);

          size_t i = 0;

          for (auto &op : call_site->args()) {
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
  LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
  return ret;
}

void retrive_constexpr_user(llvm::Constant *c, std::vector<User *> &to_remove)
{
  for (auto user : c->users()) {
    if (llvm::isa<Constant>(user) && !llvm::isa<GlobalValue>(user)) {
      if (!std::any_of(to_remove.cbegin(), to_remove.cend(), [&user](auto elem) { return user == elem; })) {
        to_remove.push_back(user);
        retrive_constexpr_user(llvm::cast<llvm::Constant>(user), to_remove);
      }
    } else {
      if (!std::any_of(to_remove.cbegin(), to_remove.cend(), [&user](auto elem) { return user == elem; })) {
        to_remove.push_back(user);
      }
    }
  }
}

void clean_host(llvm::Module &M, llvm::LLVMContext &cntx)
{
  LLVM_DEBUG(llvm::dbgs() << "START " << __PRETTY_FUNCTION__ << "\n");
  std::vector<User *> to_remove;
  for (auto &glob : M.globals()) {
    if (glob.hasName() && (glob.getName().startswith("__dev") || glob.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99"))) {
      to_remove.push_back(&glob);
    }
  }

  for (auto &glob : M.functions()) {
    if (glob.hasName() && (glob.getName().startswith("__dev") || glob.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99"))) {
      to_remove.push_back(&glob);
    }
  }

  size_t i = 0;
  while (i < to_remove.size()) {

    auto front = to_remove[i];

    for (auto user : front->users()) {
      if (llvm::isa<Constant>(user) && !llvm::isa<GlobalValue>(user)) {
        retrive_constexpr_user(llvm::cast<llvm::Constant>(user), to_remove);
      } else {
        if (!std::any_of(to_remove.cbegin(), to_remove.cend(), [&user](auto elem) { return user == elem; })) {
          to_remove.push_back(user);
        }
      }
    }

    i++;
  }


  // Unlink all
  for (size_t i = 0; i < to_remove.size(); ++i) {
    auto *user = to_remove[i];
    user->dropAllReferences();
  }


  //remove all instuction

  for (size_t i = 0; i < to_remove.size();) {
    auto *user = to_remove[i];

    if (auto current = llvm::dyn_cast<llvm::Instruction>(user)) {
      current->dropAllReferences();
      if (current->getParent())
        current->removeFromParent();
      current->deleteValue();
      std::swap(*(to_remove.begin() + i), *(to_remove.end() - 1));
      to_remove.pop_back();
      continue;
    }
    i++;
  }

  //remove all Global

  for (size_t i = 0; i < to_remove.size();) {
    auto *user = to_remove[i];

    if (auto current = llvm::dyn_cast<llvm::GlobalValue>(user)) {
      current->dropAllReferences();
      if (current->getParent())
        current->eraseFromParent();
      std::swap(*(to_remove.begin() + i), *(to_remove.end() - 1));
      to_remove.pop_back();
      continue;
    }
    i++;
  }
  LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
}


void export_functions(llvm::Module &M, SmallVectorImpl<llvm::CallSite *> &functions_to_export, StringRef filename, llvm::LLVMContext &cntx)
{
  LLVM_DEBUG(llvm::dbgs() << "START " << __PRETTY_FUNCTION__ << "\n");
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

  assert(dev_module && "Cannot retrive module");

  ValueToValueMapTy GtoG;
  cloneGlobalVariable(*dev_module, M, GtoG);


  for (auto function_to_export_call : functions_to_export) {
    auto function_to_export = function_to_export_call->getCalledFunction();
    auto function_name = function_to_export->getName();

    auto new_dev_callee = GtoG[function_to_export];

    if (auto new_function = llvm::dyn_cast<llvm::Function>(new_dev_callee)) {


      new_function->setLinkage(llvm::GlobalValue::WeakAnyLinkage);
      auto name_in_dev = builder.CreateGlobalStringPtr(function_name, ".omp_offloading.entry_name", 0, &*dev_module);

      auto name_in_host = builder.CreateGlobalStringPtr(function_name, ".omp_offloading.entry_name", 0, &M);

      {
        auto identifiers = dev_module->getIdentifiedStructTypes();
        auto type_tgt_offload_entry = llvm::find_if(identifiers, [](llvm::StructType *elem) {
          return elem->getName().startswith("struct.__tgt_offload_entry");
        });
        assert(type_tgt_offload_entry != identifiers.end() && "Not found type");
        Constant *global_value_values[5];

        global_value_values[0] = ConstantExpr::getBitCast(new_function, Type::getInt8PtrTy(cntx));

        global_value_values[1] = ConstantExpr::getBitCast(name_in_dev, Type::getInt8PtrTy(cntx));
        global_value_values[2] = builder.getInt32(0);
        global_value_values[3] = builder.getInt32(0);
        global_value_values[4] = builder.getInt32(0);

        auto tmp = ConstantStruct::get(*type_tgt_offload_entry, global_value_values);
        auto offld_entry = new GlobalVariable(*dev_module, *type_tgt_offload_entry, true, llvm::GlobalValue::WeakAnyLinkage, tmp,
                                              ".omp_offloading.entry." + function_name);
        offld_entry->setSection("omp_offloading_entries");
      }
      {

        auto identifiers = M.getIdentifiedStructTypes();
        auto type_tgt_offload_entry = llvm::find_if(identifiers, [](llvm::StructType *elem) {
          return elem->getName().startswith("struct.__tgt_offload_entry");
        });
        assert(type_tgt_offload_entry != identifiers.end() && "Not found type");
        (*type_tgt_offload_entry)->dump();
        auto new_region_id = new GlobalVariable(M, Type::getInt8Ty(cntx), true, llvm::GlobalValue::WeakAnyLinkage, builder.getInt8(0), "." + function_name + ".region_id");
        function_to_export_call->getNextNode()->setOperand(2, new_region_id);
        Constant *global_value_values[5];

        global_value_values[0] = ConstantExpr::getBitCast(new_region_id, Type::getInt8PtrTy(cntx));

        global_value_values[1] = ConstantExpr::getBitCast(name_in_host, Type::getInt8PtrTy(cntx));

        global_value_values[2] = builder.getInt64(0);
        global_value_values[3] = builder.getInt32(0);
        global_value_values[4] = builder.getInt32(0);

        auto tmp = ConstantStruct::get(*type_tgt_offload_entry, global_value_values);
        auto offld_entry = new GlobalVariable(M, *type_tgt_offload_entry, true, llvm::GlobalValue::WeakAnyLinkage, tmp,
                                              ".omp_offloading.entry." + function_name);
        offld_entry->setSection("omp_offloading_entries");
      }

    } else {
      llvm_unreachable("Cloned not a function");
    }
  }


  verifyModule(*dev_module);
  int file = 0;
  auto err = llvm::sys::fs::openFileForWrite(filename, file);
  assert(!err.value() && "Fail open module");
  raw_fd_ostream stream{file, false};
  dev_module->print(stream, nullptr);
  LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
  llvm::sys::fs::closeFile(file);
}

void flttofix::FloatToFixed::handleHero(llvm::Module &host_module, bool Hero)
{
  if (!Hero)
    return;

  LLVM_DEBUG(llvm::dbgs() << "\n##### Handle Init Hero ######\n");
  auto end_position = host_module.getModuleIdentifier().find("-host.ll");
  auto dev_name = host_module.getModuleIdentifier().substr(0, end_position) + "-dev.ll";


  auto target_mapper = host_module.getFunction("__tgt_target_mapper");
  auto target_teams_mapper = host_module.getFunction("__tgt_target_teams_mapper");

  auto functions_to_export = fix_call(target_mapper);
  functions_to_export.append(fix_call(target_teams_mapper));

  export_functions(host_module, functions_to_export, dev_name, host_module.getContext());
  host_module.dump();
  clean_host(host_module, host_module.getContext());


  LLVM_DEBUG(llvm::dbgs() << "##### End Init Hero ######\n\n");
}