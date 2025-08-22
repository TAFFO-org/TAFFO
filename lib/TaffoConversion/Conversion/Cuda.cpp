#include "../ConversionPass.hpp"

#include <llvm/IR/Operator.h>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conv"

bool ConversionPass::isSupportedCudaFunction(Function* F) {
  if (F->getName() == "cuMemcpyHtoD_v2")
    return true;
  if (F->getName() == "cuMemcpyDtoH_v2")
    return true;
  return false;
}

Value* ConversionPass::convertCudaCall(CallBase* C) {
  Function* F = C->getCalledFunction();

  unsigned BufferArgId;
  unsigned BufferSizeArgId;
  LLVM_DEBUG(log() << F->getName() << " detected, attempting to convert\n");
  if (F->getName() == "cuMemcpyHtoD_v2") {
    BufferArgId = 1;
    BufferSizeArgId = 2;
  }
  else if (F->getName() == "cuMemcpyDtoH_v2") {
    BufferArgId = 0;
    BufferSizeArgId = 2;
  }
  else {
    llvm_unreachable("Wait why are we handling a Cuda call that we don't know about?");
    return unsupported;
  }

  Value* TheBuffer = C->getArgOperand(BufferArgId);
  if (auto* BC = dyn_cast<BitCastOperator>(TheBuffer))
    TheBuffer = BC->getOperand(0);
  Value* NewBuffer = convertedValues.at(TheBuffer);
  if (!NewBuffer || !taffoConvInfo.hasValueConvInfo(NewBuffer)) {
    LLVM_DEBUG(log() << "Buffer argument not converted; trying fallback.");
    return unsupported;
  }
  LLVM_DEBUG(log() << "Found converted buffer: " << *NewBuffer << "\n");
  LLVM_DEBUG(log() << "Buffer convType is: " << *taffoConvInfo.getNewType(NewBuffer) << "\n");
  Type* VoidPtrTy = Type::getInt8Ty(C->getContext())->getPointerTo();
  Value* NewBufferArg;
  if (NewBuffer->getType() != VoidPtrTy)
    NewBufferArg = new BitCastInst(NewBuffer, VoidPtrTy, "", C);
  else
    NewBufferArg = NewBuffer;
  C->setArgOperand(BufferArgId, NewBufferArg);

  // TODO fix soon
  /*LLVM_DEBUG(log() << "Attempting to adjust buffer size\n");
  Type* OldTy = TheBuffer->getType();
  Type* NewTy = NewBuffer->getType();
  Value* OldBufSz = C->getArgOperand(BufferSizeArgId);
  Value* NewBufSz = adjustMemoryAllocationSize(OldBufSz, OldTy, NewTy, C, true);
  if (OldBufSz != NewBufSz) {
    C->setArgOperand(BufferSizeArgId, NewBufSz);
    LLVM_DEBUG(log() << "Buffer size was adjusted\n");
  }
  else {
    LLVM_DEBUG(log() << "Buffer size did not need any adjustment\n");
  }*/

  return C;
}
