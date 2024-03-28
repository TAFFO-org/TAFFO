#include "CallSiteVersions.h"
#include "LLVMFloatToFixedPass.h"
#include "WriteModule.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
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
#include <llvm-12/llvm/ADT/Twine.h>
#include <llvm/Support/FormatVariadic.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

static constexpr int substring_size = 6;
class ident_struct_type_remapper : public ValueMapTypeRemapper
{

  std::unordered_map<Type *, Type *> structed_map;

public:
  ident_struct_type_remapper(const std::vector<StructType *> &host_list_struct, const std::vector<StructType *> &dev_struct_list)
  {
    auto dev_struct = std::find_if(dev_struct_list.begin(), dev_struct_list.end(), [](StructType *lrs) { return lrs->getName().startswith("struct.ident_t"); });
    if (dev_struct == dev_struct_list.end()) {
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
    // handle struct morph from i32 to i64 from host to target
    auto find = structed_map.find(SrcTy);
    if (find != structed_map.end()) {
      return find->second;
    }
    // addresspace also for argment of functions
    if (auto fnct = dyn_cast<FunctionType>(SrcTy)) {
      llvm::SmallVector<Type *, 8> args_type;
      for (const auto &param : fnct->params()) {
        if (auto ptr = dyn_cast<PointerType>(param)) {
          args_type.push_back(ptr->getElementType()->getPointerTo(1));
        } else {
          args_type.push_back(param);
        }
      }
      return FunctionType::get(fnct->getReturnType(), args_type, false);
    }
    return SrcTy;
  }
};


// Clone all global but not the one that start with hero as they are target dependant (contains assembly code)
void cloneGlobalVariable(llvm::Module &dev, llvm::Module &host, llvm::ValueToValueMapTy &GtoG)
{
  LLVM_DEBUG(llvm::dbgs() << "START " << __PRETTY_FUNCTION__ << "\n");

  using namespace llvm;
  // Loop over all of the global variables, making corresponding globals in the
  //  new module.  Here we add them to the VMap and to the new Module.  We
  //  don't worry about attributes or initializers, they will come later.
  //


  std::unique_ptr<ValueMapTypeRemapper> remapper = std::make_unique<ident_struct_type_remapper>(host.getIdentifiedStructTypes(), dev.getIdentifiedStructTypes());
  std::unordered_set<Function *> alredy_present_function;

  for (const llvm::GlobalVariable &I : host.globals()) {
    if (!(I.getName().startswith("__dev-") || I.getName().startswith("__tgt_offload_entry")))
      continue;


    StringRef old_name = I.getName();
    if (I.getName().startswith("__dev-"))
      old_name = I.getName().substr(substring_size, std::string::npos);
    LLVM_DEBUG(llvm::dbgs() << "Searching global: " << I.getName() << " as " << old_name << "\n");
    if (auto glob = dev.getNamedGlobal(old_name)) {
      LLVM_DEBUG(llvm::dbgs() << "Found global: " << *glob << "\n");
      GtoG[&I] = glob;
    } else {

      llvm::GlobalVariable *NewGV = new GlobalVariable(
          dev, remapper->remapType(I.getValueType()), I.isConstant(), I.getLinkage(),
          (llvm::Constant *)nullptr, old_name, (llvm::GlobalVariable *)nullptr,
          I.getThreadLocalMode(), remapper->remapType(I.getType())->getPointerAddressSpace());
      NewGV->copyAttributesFrom(&I);
      LLVM_DEBUG(llvm::dbgs() << "Created global: " << *NewGV << "\n");

      GtoG[&I] = NewGV;
    }
  }


  // Loop over the functions in the module, making external functions as before
  for (const Function &F : host) {
    if (!(F.getName().startswith("__dev-") || F.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99") || F.getName().startswith("llvm.")) || F.getName().contains("_trampoline"))
      continue;
    auto old_name = F.getName();
    if (F.getName().startswith("__dev-")) {
      old_name = old_name.substr(substring_size, std::string::npos);
    }
    LLVM_DEBUG(llvm::dbgs() << "Search function: " << F.getName() << " as " << old_name << "\n");
    if (auto old = dev.getFunction(old_name)) {
      LLVM_DEBUG(llvm::dbgs() << "Founded\n");
      GtoG[&F] = old;
      alredy_present_function.insert(old);
    } else {
      auto function_type = cast<FunctionType>(remapper->remapType(F.getValueType()));

      if (old_name.contains("omp_outlined")) {
        llvm::SmallVector<Type *, 8> types;
        auto param = function_type->params();
        types.push_back(PointerType::get(llvm::cast<PointerType>(param[0])->getElementType(), 0));
        types.push_back(PointerType::get(llvm::cast<PointerType>(param[1])->getElementType(), 0));


        for (size_t i = 2; i < param.size(); ++i) {
          types.push_back(param[i]);
        }
        function_type = FunctionType::get(function_type->getReturnType(), types, false);
      }


      Function *NF = Function::Create(function_type, F.getLinkage(),
                                      F.getAddressSpace(), old_name, &dev);

      NF->copyAttributesFrom(&F);
      LLVM_DEBUG(llvm::dbgs() << "Created function: " << NF->getName() << "\n");
      LLVM_DEBUG(llvm::dbgs() << *NF);
      GtoG[&F] = NF;
    }
  }


  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (const GlobalVariable &G : host.globals()) {
    if (G.getName().startswith("llvm.used"))
      continue;
    if (!(G.getName().startswith("__dev-") || G.getName().startswith("__tgt_offload_entry")))
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
    if (!(I.getName().startswith("__dev-") || I.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99") || I.getName().startswith("llvm.")) || I.getName().startswith("llvm.") || I.getName().contains("trampoline"))
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

  // Reset the function to corret call type
  {
    for (auto &fnct : dev) {
      for (auto &BB : fnct)
        for (auto &I : make_early_inc_range(BB)) {
          if (auto call_inst = dyn_cast<CallSite>(&I)) {
            PointerType *FPTy = cast<PointerType>(call_inst->getCalledOperand()->getType());
            if (FPTy->getElementType() != call_inst->getFunctionType()) {
              call_inst->setCalledFunction(call_inst->getCalledFunction());
            }
          }
        }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
}


// openmp .offload_sizes express the dimension of the underlyng data in bytes
//  sometimes can be zero if it is alredy specified by previous call
//  prev_call_site is the call to the fixpoint version of the function without indirection
//  target_mapper is the call to one of the possible mapper function
void fix_size(CallSite *prev_call_site, llvm::CallSite *target_mapper_caller, const llvm::DenseMap<llvm::Function *, llvm::Function *> &functionPool)
{
  // retrive old global size
  auto size_global = cast<ConstantDataArray>(cast<GlobalVariable>(target_mapper_caller->getOperand(6)->stripPointerCastsAndAliases())->getInitializer());
  auto &cntx = prev_call_site->getContext();
  auto &M = *prev_call_site->getModule();

  // retrive old type size
  Function *old_f = nullptr;
  for (const auto &fun : functionPool) {
    if (fun.getSecond() == prev_call_site->getCalledFunction()) {
      old_f = fun.getFirst();
    }
  }

  if (old_f == nullptr) {
    return;
  }

  SmallVector<Type *> old_arg_types;
  for (const auto &arg : old_f->args()) {
    old_arg_types.push_back(arg.getType());
  }

  // retrive new type size
  SmallVector<Type *> new_arg_types;
  for (const auto &arg : prev_call_site->getCalledFunction()->args()) {
    new_arg_types.push_back(arg.getType());
  }

  SmallVector<unsigned long, 3> calculated_sizes;

  for (const auto &arg : enumerate(zip(old_arg_types, new_arg_types))) {
    Type *old_arg = std::get<0>(arg.value());
    Type *new_arg = std::get<1>(arg.value());
    auto size = size_global->getElementAsInteger(arg.index());
    unsigned long old_size = 0;
    unsigned long new_size = 0;

    if (const auto *pointed = dyn_cast<PointerType>(old_arg)) {
      if (pointed->getElementType()->isArrayTy()) {
        // TODO handle recursion of array
        auto old_array_type = cast<ArrayType>(pointed->getElementType());
        auto new_array_type = cast<ArrayType>(cast<PointerType>(new_arg)->getElementType());
        old_size = old_array_type->getElementType()->getScalarSizeInBits();
        new_size = new_array_type->getElementType()->getScalarSizeInBits();

      } else if (pointed->getElementType()->isIntOrIntVectorTy() || pointed->getElementType()->isFPOrFPVectorTy()) {
        old_size = pointed->getElementType()->getScalarSizeInBits();
        new_size = cast<PointerType>(new_arg)->getElementType()->getScalarSizeInBits();

      } else {
        llvm_unreachable("Type not handled");
      }
    } else {
      if (old_arg->isArrayTy()) {
        // TODO handle recursion of array
        auto old_array_type = cast<ArrayType>(old_arg);
        auto new_array_type = cast<ArrayType>(new_arg);
        old_size = old_array_type->getElementType()->getScalarSizeInBits();
        new_size = new_array_type->getElementType()->getScalarSizeInBits();

      } else if (old_arg->isIntOrIntVectorTy() || old_arg->isFPOrFPVectorTy()) {
        old_size = old_arg->getScalarSizeInBits();
        new_size = new_arg->getScalarSizeInBits();

      } else {
        llvm_unreachable("Type not handled");
      }
    }
    if (old_size == 0 || new_size == 0) {
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Error on size handling\n\tOld Size {2}-> {0}\n\tNew Size {3}->{1}", old_size, new_size, *old_arg, *new_arg));

    } else if (old_size != new_size) {
      assert(size % old_size == 0 && "Not a multiple of size");
      size = size / old_size * new_size;
    }
    calculated_sizes.push_back(size);
  }
  auto data_to_insert = cast<ConstantDataArray>(ConstantDataArray::get(cntx, calculated_sizes));
  auto new_data = new GlobalVariable(M, data_to_insert->getType(), true, llvm::GlobalValue::InternalLinkage, data_to_insert, ".offload_sizes");
  Constant *C[2] = {cast<Constant>(ConstantInt::get(Type::getInt32Ty(cntx), 0)), cast<Constant>(ConstantInt::get(Type::getInt32Ty(cntx), 0))};
  auto const_expr = ConstantExpr::getGetElementPtr(data_to_insert->getType(), new_data, C, true);
  target_mapper_caller->setOperand(6, const_expr);
}


// prev_call_site is the call to the fixpoint version of the function without indirection
// target_mapper is the call to one of the possible mapper function
void fix_argument_cast(CallSite *prev_call_site, llvm::CallSite *target_mapper_caller)
{
  LLVM_DEBUG(llvm::dbgs() << "In " << prev_call_site->getParent()->getParent()->getName() << "\nfound " << *target_mapper_caller << "\nwith predecessor " << *prev_call_site << "\n");

  llvm::IRBuilder<> builder(prev_call_site->getContext());
  auto call_site = prev_call_site;
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
}

SmallVector<llvm::CallSite *> fix_call(llvm::Function *function, llvm::DenseMap<llvm::Function *, llvm::Function *> &functionPool)
{
  LLVM_DEBUG(llvm::dbgs() << "START " << __PRETTY_FUNCTION__ << "\n");
  SmallVector<llvm::CallSite *> ret{};
  if (!function) {
    LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
    return ret;
  }


  // Find all call place
  for (const auto &caller : function->users()) {
    if (const auto &target_mapper_caller = llvm::dyn_cast<llvm::CallSite>(caller)) {
      // search for a call site as previous instruction
      if (const auto &prev_call_site = llvm::dyn_cast<llvm::CallSite>(target_mapper_caller->getPrevNode())) {
        if (prev_call_site->getCalledFunction()->getName().startswith("__omp_offloading_10303_4dc05d6_func_l99")) {
          fix_size(prev_call_site, target_mapper_caller, functionPool);
          fix_argument_cast(prev_call_site, target_mapper_caller);
          ret.emplace_back(prev_call_site);
        }
      }
    }
  }


  LLVM_DEBUG(llvm::dbgs() << "END " << __PRETTY_FUNCTION__ << "\n");
  return ret;
}

void retrive_constexpr_user(llvm::Constant *c, std::vector<User *> &to_remove)
{
  for (const auto &user : c->users()) {
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

void clean_host(llvm::Module &M, const llvm::LLVMContext &cntx)
{
  LLVM_DEBUG(llvm::dbgs() << "START " << __PRETTY_FUNCTION__ << "\n");
  std::vector<User *> to_remove;
  for (auto &glob : M.globals()) {
    if (glob.hasName() && (glob.getName().startswith("__dev-") || glob.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99"))) {
      to_remove.push_back(&glob);
    }
  }

  for (auto &glob : M.functions()) {
    if (glob.hasName() && (glob.getName().startswith("__dev-") || glob.getName().startswith("__omp_offloading_10303_4dc05d6_func_l99"))) {
      to_remove.push_back(&glob);
    }
  }

  size_t i = 0;
  while (i < to_remove.size()) {

    auto front = to_remove[i];

    for (const auto &user : front->users()) {
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


  // remove all instuction

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

  // remove all Global

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

void normalize_addrspace(llvm::Module &dev)
{
  // go through all instructions until a fixed point is reached each time search for GEP with address space not aligned between source and dest and recreate a new GEP with the correct address space
  //  also we cannot use RAUW because address space is part of the type so we use remapinstruction that has the power to create invalid IR
  // we continue to iterate until all GEP are correctly handled
  //  the fixed point iteration is due to possible dependency on future gep
  // all this stuff is a hack because clonefunctioninto creates invalid GEP, and we cannot remap all pointers to pointer addrespace(1) as this will imply also local alloca will be transformed.
  {
    bool stop_looping = false;
    ValueToValueMapTy VM{};
    llvm::SmallVector<Instruction *, 6> to_rem;
    while (!stop_looping) {
      stop_looping = true;
      for (auto &fnct : dev) {
        for (auto &BB : fnct)
          for (auto &I : make_early_inc_range(BB)) {
            if (auto gep = dyn_cast<GetElementPtrInst>(&I)) {
              if (VM.find(gep) == VM.end() && gep->getOperand(0)->getType()->isPointerTy() && gep->getType()->isPointerTy() && gep->getOperand(0)->getType()->getPointerAddressSpace() != gep->getType()->getPointerAddressSpace()) {
                SmallVector<Value *, 8> elems;
                for (unsigned int i = 1; i < gep->getNumOperands(); ++i)
                  elems.push_back(gep->getOperand(i));
                to_rem.push_back(gep);
                VM[gep] = GetElementPtrInst::Create(gep->getSourceElementType(), gep->getOperand(0), elems, "", gep);
                for (auto user : gep->users()) {
                  RemapInstruction(cast<Instruction>(user), VM, llvm::RF_IgnoreMissingLocals);
                }
                stop_looping = false;
              }
            } else {
              RemapInstruction(cast<Instruction>(&I), VM, llvm::RF_IgnoreMissingLocals);
            }
          }
      }
    }

    for (auto *rem : to_rem) {

      rem->dropAllReferences();
      rem->eraseFromParent();
    }
  }
  // fix address space inside array of pointers
  {
    for (auto &fnct : dev) {
      for (auto &BB : fnct)
        for (auto &I : make_early_inc_range(BB)) {
          if (auto sto = dyn_cast<StoreInst>(&I)) {
            auto fir_type = sto->getOperand(0)->getType();
            auto sec_type = cast<PointerType>(sto->getOperand(1)->getType())->getPointerElementType();

            if (fir_type->isPointerTy() && sec_type->isPointerTy() && fir_type->getPointerAddressSpace() != sec_type->getPointerAddressSpace()) {
              auto fir_addr = fir_type->getPointerAddressSpace();
              auto new_stored_type = cast<PointerType>(sec_type)->getElementType()->getPointerTo(fir_addr)->getPointerTo(0);
              auto CasI = CastInst::CreatePointerBitCastOrAddrSpaceCast(sto->getOperand(1), new_stored_type, "", sto);
              sto->setOperand(1, CasI);
            }
          }
        }
    }
  }
  // fix bitcast with invalid address space
  {
    for (auto &fnct : dev) {
      for (auto &BB : fnct)
        for (auto &I : make_early_inc_range(BB)) {
          if (auto cast_inst = dyn_cast<BitCastInst>(&I)) {
            if (!CastInst::castIsValid(Instruction::BitCast, cast_inst->getOperand(0), cast_inst->getType())) {

              auto new_cast = CastInst::CreatePointerBitCastOrAddrSpaceCast(cast_inst->getOperand(0), cast_inst->getType(), "", cast_inst);
              cast_inst->replaceAllUsesWith(new_cast);
              cast_inst->eraseFromParent();
            }
          }
        }
    }
  }


  // AddrSpaceCast must be between different address spaces
  {
    for (auto &fnct : dev) {
      for (auto &BB : fnct)
        for (auto &I : make_early_inc_range(BB)) {
          if (auto cast_inst = dyn_cast<AddrSpaceCastInst>(&I)) {
            if (cast_inst->getSrcAddressSpace() == cast_inst->getDestAddressSpace()) {
              cast_inst->replaceAllUsesWith(cast_inst->getOperand(0));
              cast_inst->removeFromParent();
              cast_inst->deleteValue();
            }
          }
        }
    }
  }
}


auto set_new_region_id(Module &M, CallSite *function_to_export_call, Twine function_name)
{
  auto &cntx = M.getContext();
  auto new_region_id = new GlobalVariable(M, Type::getInt8Ty(cntx), true, llvm::GlobalValue::WeakAnyLinkage, ConstantInt::get(Type::getInt8Ty(cntx), 0), "." + function_name + ".region_id");
  function_to_export_call->getNextNode()->setOperand(2, new_region_id);
  return new_region_id;
}

void export_functions(llvm::Module &M, const SmallVectorImpl<llvm::CallSite *> &functions_to_export, StringRef filename, llvm::LLVMContext &cntx)
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
  normalize_addrspace(*dev_module);
  write_module("crash_hero.ll", *dev_module);
  assert(!verifyModule(*dev_module, &llvm::dbgs()) && "Broken after normal Dev module");


  for (const auto &function_to_export_call : functions_to_export) {
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

        auto new_region_id = set_new_region_id(M, function_to_export_call, function_name);

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


  write_module(filename, *dev_module);
  llvm::dbgs() << "\n\n\n";
  assert(!verifyModule(*dev_module, &llvm::dbgs()) && "Broken Dev module");
  LLVM_DEBUG(llvm::dbgs() << "\nEND " << __PRETTY_FUNCTION__ << "\n");
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


  auto functions_to_export = fix_call(target_mapper, functionPool);
  functions_to_export.append(fix_call(target_teams_mapper, functionPool));


  export_functions(host_module, functions_to_export, dev_name, host_module.getContext());

  clean_host(host_module, host_module.getContext());


  assert(!verifyModule(host_module, &llvm::dbgs()) && "Broken host module");

  LLVM_DEBUG(llvm::dbgs() << "##### End Init Hero ######\n\n");
}