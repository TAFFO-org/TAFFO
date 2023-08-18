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
  if (F->getName() == "clEnqueueWriteBuffer")
    return true;
  if (F->getName() == "clSetKernelArg")
    return true;
  return false;
}


Value *FloatToFixed::convertOpenCLCall(CallBase *C)
{
  Function *F = C->getCalledFunction();

  unsigned BufferArgId;
  unsigned BufferSizeArgId;
  LLVM_DEBUG(dbgs() << F->getName() << " detected, attempting to convert\n");
  if (F->getName() == "clCreateBuffer") {
    BufferArgId = 3;
    BufferSizeArgId = 2;
  } else if (F->getName() == "clEnqueueReadBuffer" || F->getName() == "clEnqueueWriteBuffer") {
    BufferArgId = 5;
    BufferSizeArgId = 4;
  } else if (F->getName() == "clSetKernelArg") {
    BufferArgId = 3;
    BufferSizeArgId = 2;
  } else {
    llvm_unreachable("Wait why are we handling an OpenCL call that we don't know about?");
    return Unsupported;
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
  Value *NewBufferArg;
  if (NewBuffer->getType() != VoidPtrTy) {
    NewBufferArg = new BitCastInst(NewBuffer, VoidPtrTy, "", C);
  } else {
    NewBufferArg = NewBuffer;
  }
  C->setArgOperand(BufferArgId, NewBufferArg);

  LLVM_DEBUG(dbgs() << "Attempting to adjust buffer size\n");
  Type *OldTy = TheBuffer->getType();
  Type *NewTy = NewBuffer->getType();
  Value *OldBufSz = C->getArgOperand(BufferSizeArgId);
  Value *NewBufSz = adjustBufferSize(OldBufSz, OldTy, NewTy, C, true);
  if (OldBufSz != NewBufSz) {
    C->setArgOperand(BufferSizeArgId, NewBufSz);
    LLVM_DEBUG(dbgs() << "Buffer size was adjusted\n");
  } else {
    LLVM_DEBUG(dbgs() << "Buffer size did not need any adjustment\n");
  }

  return C;
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

