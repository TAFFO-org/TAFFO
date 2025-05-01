#include "CudaKernelPatcher.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/IPO/OpenMPOpt.h>

#define DEBUG_TYPE "taffo-init"

using namespace llvm;
using namespace taffo;

void getAndDeleteAnnotationsOfArgumentCuda(Function& KernF, unsigned ArgId, std::optional<ConstantExpr*>& Res) {
  Argument* Arg = KernF.getArg(ArgId);

  /* clang produces code like this for each argument:
   *  %arg.addr = alloca
   *  store %arg, %arg.addr
   * First thing we do is to search for the single store associated to the argument */
  StoreInst* StoreUser = dyn_cast_or_null<StoreInst>(*(Arg->user_begin()));
  if (!StoreUser)
    return;
  AllocaInst* StorePtr = dyn_cast_or_null<AllocaInst>(StoreUser->getPointerOperand());
  if (!StorePtr)
    return;

  /* The annotation adds a function call that takes the .addr pointer we just obtained through a bitcast:
   *  %tmp = bitcast %arg.addr, i8*
   *  call void @llvm.var.annotation(i8* %tmp, "annotation", __FILE__, __LINE__, i8* null)
   * Now we attempt to detect the pattern above */
  CallInst* CallI = nullptr;
  for (User* U : StorePtr->users()) {
    BitCastInst* BCI = dyn_cast<BitCastInst>(U);
    if (!BCI)
      continue;

    for (User* UU : BCI->users()) {
      CallI = dyn_cast<CallInst>(UU);
      if (!CallI)
        continue;
      if (CallI->getCalledFunction()->getName() != "llvm.var.annotation")
        continue;
      ConstantExpr* AnnoStringCExp = dyn_cast<ConstantExpr>(CallI->getArgOperand(1));
      if (!AnnoStringCExp || AnnoStringCExp->getOpcode() != Instruction::GetElementPtr)
        continue;
      GlobalVariable* AnnoContent = dyn_cast<GlobalVariable>(AnnoStringCExp->getOperand(0));
      if (!AnnoContent)
        continue;
      ConstantDataSequential* AnnoStr = dyn_cast<ConstantDataSequential>(AnnoContent->getInitializer());
      if (!AnnoStr || !(AnnoStr->isString()))
        continue;

      LLVM_DEBUG(log() << "Found annotation \"" << AnnoStr->getAsString() << "\" on function arg " << ArgId << " ("
                       << Arg->getName() << ") of function " << KernF.getName() << "\n");
      Res = AnnoStringCExp;
      break;
    }
    if (Res.has_value())
      break;
  }
  if (!Res.has_value()) {
    LLVM_DEBUG(log() << "Found no annotation on function arg " << ArgId << " (" << Arg->getName() << ") of function "
                     << KernF.getName() << "\n");
    return;
  }

  TaffoInfo::getInstance().eraseValue(CallI);
}

void createCudaKernelTrampoline(Module& M, Function& KernF) {
  /* Collect the annotations */
  SmallVector<std::optional<ConstantExpr*>, 8> Annotations;
  SmallVector<Type*, 8> ArgTypes;
  unsigned NumAnnos = 0;
  for (unsigned ArgId = 0; ArgId < KernF.arg_size(); ArgId++) {
    std::optional<ConstantExpr*> OptAnn;
    getAndDeleteAnnotationsOfArgumentCuda(KernF, ArgId, OptAnn);
    NumAnnos += !!OptAnn.has_value();
    Annotations.append({OptAnn});
    ArgTypes.append({KernF.getArg(ArgId)->getType()});
  }
  if (NumAnnos == 0) {
    LLVM_DEBUG(log() << "No annotations, no trampoline. Skipping.\n");
    return;
  }

  /* Create the trampoline function */
  FunctionType* FunTy = FunctionType::get(Type::getVoidTy(KernF.getContext()), ArgTypes, false);
  auto FunName = KernF.getName() + ".taffo.ocl.tramp";
  Function* NewF = Function::Create(FunTy, KernF.getLinkage(), KernF.getAddressSpace(), FunName, &M);
  BasicBlock* TheBB = BasicBlock::Create(NewF->getContext(), "", NewF);
  /* Disable optimizations on the trampoline function. This is required later to ensure the VRA is properly fooled,
   * otherwise mem2reg will replace all our work with a function consisting of just a call.
   *   See also: in Annotations.cpp we disable removing this attribute just for trampolines. */
  NewF->addFnAttr(Attribute::OptimizeNone);
  NewF->addFnAttr(Attribute::NoInline);

  /* Do the dirty job and simulate clang... */
  SmallVector<AllocaInst*, 8> Allocas;
  IRBuilder<> Builder(TheBB);
  Function* AnnoFun = Intrinsic::getDeclaration(&M, Intrinsic::var_annotation);
  for (unsigned ArgId = 0; ArgId < NewF->arg_size(); ArgId++) {
    Argument* Arg = NewF->getArg(ArgId);
    AllocaInst* Alloca = Builder.CreateAlloca(Arg->getType());
    Allocas.append({Alloca});
  }
  for (unsigned ArgId = 0; ArgId < NewF->arg_size(); ArgId++) {
    AllocaInst* Alloca = Allocas[ArgId];
    Builder.CreateStore(NewF->getArg(ArgId), Alloca);
    auto& OptAnn = Annotations[ArgId];
    if (!OptAnn.has_value())
      continue;
    ConstantExpr* AnnoStr = OptAnn.value();
    Value* BCI = Builder.CreateBitCast(Alloca, PointerType::get(Type::getInt8Ty(KernF.getContext()), 0));
    Constant* Null = Constant::getNullValue(PointerType::get(Type::getInt8Ty(M.getContext()), 0));
    Builder.CreateCall(AnnoFun->getFunctionType(), AnnoFun, {BCI, AnnoStr, Null, Builder.getIntN(32, 0), Null});
  }
  SmallVector<Value*, 8> Loads;
  for (unsigned ArgId = 0; ArgId < NewF->arg_size(); ArgId++) {
    AllocaInst* Alloca = Allocas[ArgId];
    LoadInst* Load = Builder.CreateLoad(ArgTypes[ArgId], Alloca);
    Loads.append({Load});
  }
  Builder.CreateCall(KernF.getFunctionType(), &KernF, Loads);
  Builder.CreateRetVoid();

  TaffoInfo::getInstance().setOpenCLTrampoline(*NewF, KernF);
  LLVM_DEBUG(log() << "Created trampoline:\n"
                   << *NewF);
}

#define CUDA_KERNEL_METADATA "nvvm.annotations"
static void getCudaKernels(llvm::Module& M, llvm::SmallVectorImpl<llvm::Function*>& Fs) {
  NamedMDNode* MD = M.getNamedMetadata(CUDA_KERNEL_METADATA);

  if (!MD)
    return;

  for (auto* Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;
    MDString* KindID = dyn_cast<MDString>(Op->getOperand(1));
    if (!KindID || KindID->getString() != "kernel")
      continue;

    Function* KernelFn = mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
    if (!KernelFn)
      continue;

    Fs.append({KernelFn});
  }

  return;
}

void taffo::createCudaKernelTrampolines(Module& M) {
  LLVM_DEBUG(log() << "Creating Cuda trampolines...\n");
  SmallVector<Function*, 2> KernFs;
  getCudaKernels(M, KernFs);
  for (Function* F : KernFs) {
    LLVM_DEBUG(log() << "Found Cuda kernel function " << F->getName() << "\n");
    createCudaKernelTrampoline(M, *F);
  }
  LLVM_DEBUG(log() << "Finished creating Cuda trampolines.\n\n");
}
