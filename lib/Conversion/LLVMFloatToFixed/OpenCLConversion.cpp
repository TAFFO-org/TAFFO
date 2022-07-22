#include "LLVMFloatToFixedPass.h"
#include "llvm/IR/Operator.h"

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

