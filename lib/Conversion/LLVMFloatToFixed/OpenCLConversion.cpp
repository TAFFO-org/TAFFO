#include "LLVMFloatToFixedPass.h"
#include "Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;
using namespace flttofix;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"


bool FloatToFixed::isSupportedOpenCLFunction(Function *F)
{
  if (F->getName() == "clCreateBuffer")
    return true;
  if (F->getName() == "clEnqueueReadBuffer")
    return true;
  return false;
}


Value *FloatToFixed::convertOpenCLCall(CallBase *C)
{
  Function *F = C->getCalledFunction();

  if (F->getName() == "clCreateBuffer" || F->getName() == "clEnqueueReadBuffer") {
    unsigned BufferArgId;
    if (F->getName() == "clCreateBuffer") {
      LLVM_DEBUG(dbgs() << "clCreateBuffer detected, attempting to convert\n");
      BufferArgId = 3;
    } else {
      LLVM_DEBUG(dbgs() << "clEnqueueReadBuffer detected, attempting to convert\n");
      BufferArgId = 5;
    }
    
    Value *TheBuffer = C->getArgOperand(BufferArgId);
    if (auto *BC = dyn_cast<BitCastOperator>(TheBuffer)) {
      TheBuffer = BC->getOperand(0);
    }
    Value *NewBuffer = matchOp(TheBuffer);
    if (!NewBuffer || !hasInfo(NewBuffer)) {
      LLVM_DEBUG(dbgs() << "Buffer argument not converted; trying fallback.");
      return Unsupported;
    }
    LLVM_DEBUG(dbgs() << "Found converted buffer: " << *NewBuffer << "\n");
    LLVM_DEBUG(dbgs() << "Buffer fixp type is: " << valueInfo(NewBuffer)->fixpType.toString() << "\n");
    Type *VoidPtrTy = Type::getInt8Ty(C->getContext())->getPointerTo();
    if (NewBuffer->getType() != VoidPtrTy) {
      NewBuffer = new BitCastInst(NewBuffer, VoidPtrTy, "", C);
    } 
    C->setArgOperand(BufferArgId, NewBuffer);
    return C;
  }

  llvm_unreachable("Wait why are we handling an OpenCL call that we don't know about?");
  return Unsupported;
}


void FloatToFixed::cleanUpOpenCLKernelTrampolines(Module *M)
{
  LLVM_DEBUG(dbgs() << "Cleaning up OpenCL trampolines inserted by Initializer...\n");
  SmallVector<Function *, 4> FuncsToDelete;

  for (Function& F: M->functions()) {
    Function *KernF = nullptr;
    mdutils::MetadataManager::retrieveOpenCLCloneTrampolineMetadata(&F, &KernF);
    if (!KernF)
      continue;
    
    MDNode *MDN = KernF->getMetadata(CLONED_FUN_METADATA);
    assert(MDN && "OpenCL kernel function with trampoline but no cloned function??");
    ValueAsMetadata *MDNewKernF = cast<ValueAsMetadata>(MDN->getOperand(0U));
    Function *NewKernF = cast<Function>(MDNewKernF->getValue());
    Function *NewFixpKernF = functionPool[NewKernF];
    assert(NewFixpKernF && "OpenCL kernel function cloned but not converted????");

    LLVM_DEBUG(dbgs() << "Processing trampoline " << F.getName() 
        << ", KernF=" << KernF->getName() << ", NewKernF=" << NewKernF->getName() 
        << ", NewFixpKernF=" << NewFixpKernF->getName() << "\n");

    FuncsToDelete.append({&F, KernF, NewKernF});
    std::string KernFunName = std::string(KernF->getName());
    KernF->setName("");
    NewFixpKernF->setName(KernFunName);

    NamedMDNode *NVVMM = M->getNamedMetadata("nvvm.annotations");
    unsigned I = 0U;
    for (MDNode *NVVMNode: NVVMM->operands()) {
      ValueAsMetadata *MDKF = dyn_cast<ValueAsMetadata>(NVVMNode->getOperand(0U));
      if (MDKF->getValue() == KernF) {
        LLVM_DEBUG(dbgs() << "Found NVVM annotation " << *NVVMNode << "\n");
        MDNode *NewNVVMNode = MDNode::get(M->getContext(), {
          ValueAsMetadata::get(NewFixpKernF),
          NVVMNode->getOperand(1U),
          NVVMNode->getOperand(2U)
        });
        NVVMM->setOperand(I, NewNVVMNode);
      }
      I++;
    }
  }

  for (Function *F: FuncsToDelete) {
    F->eraseFromParent();
  }
  LLVM_DEBUG(dbgs() << "Finished!\n");
}

