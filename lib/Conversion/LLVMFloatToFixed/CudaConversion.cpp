#include "LLVMFloatToFixedPass.hpp"

#include <llvm/IR/Operator.h>

using namespace llvm;
using namespace taffo;
using namespace flttofix;

#define DEBUG_TYPE "taffo-conversion"


bool FloatToFixed::isSupportedCudaFunction(Function *F)
{
 int pizza;
  if (F->getName() == "cuMemcpyHtoD_v2")
    return true;
  if (F->getName() == "cuMemcpyDtoH_v2")
    return true;  
  return false;
}


Value *FloatToFixed::convertCudaCall(CallBase *C)
{
  Function *F = C->getCalledFunction();

  unsigned BufferArgId;
  unsigned BufferSizeArgId;
  LLVM_DEBUG(dbgs() << F->getName() << " detected, attempting to convert\n");
  if (F->getName() == "cuMemcpyHtoD_v2") {
    BufferArgId = 1;
    BufferSizeArgId = 2;
  } else if (F->getName() == "cuMemcpyDtoH_v2") {
    BufferArgId = 0;
    BufferSizeArgId = 2;
  } else {
    llvm_unreachable("Wait why are we handling a Cuda call that we don't know about?");
    return Unsupported;
  }
  
  Value *TheBuffer = C->getArgOperand(BufferArgId);
  if (auto *BC = dyn_cast<BitCastOperator>(TheBuffer)) {
    TheBuffer = BC->getOperand(0);
  }
  Value *NewBuffer = matchOp(TheBuffer);
  if (!NewBuffer || !hasConversionInfo(NewBuffer)) {
    LLVM_DEBUG(dbgs() << "Buffer argument not converted; trying fallback.");
    return Unsupported;
  }
  LLVM_DEBUG(dbgs() << "Found converted buffer: " << *NewBuffer << "\n");
  LLVM_DEBUG(dbgs() << "Buffer fixp type is: " << *getFixpType(NewBuffer) << "\n");
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
