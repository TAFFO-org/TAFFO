#include "OpenCLKernelPatcher.h"
#include "Metadata.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include <string>

#define DEBUG_TYPE "taffo-init"

using namespace llvm;
using namespace taffo;


void findOpenCLKernels(Module &M, SmallVectorImpl<Function *> &Fs)
{
  for (auto& F: M.functions()) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL && !F.isVarArg()) {
      LLVM_DEBUG(dbgs() << "Found OpenCL kernel function " << F.getName() << "\n");
      Fs.append({&F});
    }
  }
}

void getAndDeleteAnnotationsOfArgument(Function& KernF, unsigned ArgId, Optional<ConstantExpr *>& Res)
{
  Argument *Arg = KernF.getArg(ArgId);

  /* clang produces code like this for each argument:
   *  %arg.addr = alloca
   *  store %arg, %arg.addr
   * First thing we do is to search for the single store associated to the argument */
  StoreInst *StoreUser = dyn_cast_or_null<StoreInst>(*(Arg->user_begin()));
  if (!StoreUser)
    return;
  AllocaInst *StorePtr = dyn_cast_or_null<AllocaInst>(StoreUser->getPointerOperand());
  if (!StorePtr)
    return;

  /* The annotation adds a function call that takes the .addr pointer we just obtained through a bitcast:
   *  %tmp = bitcast %arg.addr, i8*
   *  call void @llvm.var.annotation(i8* %tmp, "annotation", __FILE__, __LINE__, i8* null)
   * Now we attempt to detect the pattern above */
  CallInst *CallI = nullptr;
  for (User *U: StorePtr->users()) {
    BitCastInst *BCI = dyn_cast<BitCastInst>(U);
    if (!BCI) 
      continue;
    
    for (User *UU: BCI->users()) {
      CallI = dyn_cast<CallInst>(UU);
      if (!CallI)
        continue;
      if (CallI->getCalledFunction()->getName() != "llvm.var.annotation")
        continue;
      ConstantExpr *AnnoStringCExp = dyn_cast<ConstantExpr>(CallI->getArgOperand(1));
      if (!AnnoStringCExp || AnnoStringCExp->getOpcode() != Instruction::GetElementPtr)
        continue;
      GlobalVariable *AnnoContent = dyn_cast<GlobalVariable>(AnnoStringCExp->getOperand(0));
      if (!AnnoContent)
        continue;
      ConstantDataSequential *AnnoStr = dyn_cast<ConstantDataSequential>(AnnoContent->getInitializer());
      if (!AnnoStr || !(AnnoStr->isString()))
        continue;
      
      LLVM_DEBUG(dbgs() << "Found annotation \"" << AnnoStr->getAsString() << "\" on function arg " << ArgId << " (" << Arg->getName() << ") of function " << KernF.getName() << "\n");
      Res = AnnoStringCExp;
      break;
    }
    if (Res.hasValue())
      break;
  }
  if (!Res.hasValue()) {
    LLVM_DEBUG(dbgs() << "Found no annotation on function arg " << ArgId << " (" << Arg->getName() << ") of function " << KernF.getName() << "\n");
    return;
  }
  
  
  CallI->eraseFromParent();
}

void createOpenCLKernelTrampoline(Module &M, Function& KernF)
{
  /* Collect the annotations */
  SmallVector<Optional<ConstantExpr *>, 8> Annotations;
  SmallVector<Type *, 8> ArgTypes;
  for (unsigned ArgId = 0; ArgId < KernF.arg_size(); ArgId++) {
    Optional<ConstantExpr *> OptAnn;
    getAndDeleteAnnotationsOfArgument(KernF, ArgId, OptAnn);
    Annotations.append({OptAnn});
    ArgTypes.append({KernF.getArg(ArgId)->getType()});
  }

  /* Create the trampoline function */
  FunctionType *FunTy = FunctionType::get(Type::getVoidTy(KernF.getContext()), ArgTypes, false);
  auto FunName = KernF.getName() + ".taffo.ocl.tramp";
  Function *NewF = Function::Create(FunTy, KernF.getLinkage(), KernF.getAddressSpace(), FunName, &M);
  BasicBlock *TheBB = BasicBlock::Create(NewF->getContext(), "", NewF);

  /* Do the dirty job and simulate clang... */
  SmallVector<AllocaInst *, 8> Allocas;
  IRBuilder<> Builder(TheBB);
  Function *AnnoFun = Intrinsic::getDeclaration(&M, Intrinsic::var_annotation);
  for (unsigned ArgId = 0; ArgId < NewF->arg_size(); ArgId++) {
    Argument *Arg = NewF->getArg(ArgId);
    AllocaInst *Alloca = Builder.CreateAlloca(Arg->getType());
    Allocas.append({Alloca});
  }
  for (unsigned ArgId = 0; ArgId < NewF->arg_size(); ArgId++) {
    AllocaInst *Alloca = Allocas[ArgId];
    Builder.CreateStore(NewF->getArg(ArgId), Alloca);
    auto& OptAnn = Annotations[ArgId];
    if (!OptAnn.hasValue())
      continue;
    ConstantExpr *AnnoStr = OptAnn.getValue();
    Value *BCI = Builder.CreateBitCast(Alloca, Type::getInt8PtrTy(KernF.getContext()));
    Constant *Null = Constant::getNullValue(Type::getInt8PtrTy(M.getContext()));
    Builder.CreateCall(AnnoFun->getFunctionType(), AnnoFun, {BCI, AnnoStr, Null, Builder.getIntN(32, 0), Null});
  }
  SmallVector<Value *, 8> Loads;
  for (unsigned ArgId = 0; ArgId < NewF->arg_size(); ArgId++) {
    AllocaInst *Alloca = Allocas[ArgId];
    LoadInst *Load = Builder.CreateLoad(ArgTypes[ArgId], Alloca);
    Loads.append({Load});
  }
  Builder.CreateCall(KernF.getFunctionType(), &KernF, Loads);
  Builder.CreateRetVoid();

  /* Add metadata for identification */
  mdutils::MetadataManager::setOpenCLCloneTrampolineMetadata(NewF, &KernF);

  LLVM_DEBUG(dbgs() << "Created trampoline:\n" << *NewF);
}

void taffo::createOpenCLKernelTrampolines(Module &M)
{
  LLVM_DEBUG(dbgs() << "Creating OpenCL trampolines...\n");
  SmallVector<Function *, 2> KernFs;
  findOpenCLKernels(M, KernFs);
  for (Function *F: KernFs) {
    createOpenCLKernelTrampoline(M, *F);
  }
  LLVM_DEBUG(dbgs() << "Finished creating OpenCL trampolines.\n\n");
}
