//===-- IPO/OpenMPOpt.cpp - Collection of OpenMP specific optimizations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "OpenMPAnalyzer.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "taffo-utils"
#endif

using namespace llvm;


bool getValuesInOffloadArrays(llvm::CallInst &RuntimeCall,
                              MutableArrayRef<OffloadArray> OAs)
{
  using namespace llvm;
  assert(OAs.size() == 3 && "Need space for three offload arrays!");

  // A runtime call that involves memory offloading looks something like:
  // call void @__tgt_target_data_begin_mapper(arg0, arg1,
  //   i8** %offload_baseptrs, i8** %offload_ptrs, i64* %offload_sizes,
  // ...)
  // So, the idea is to access the allocas that allocate space for these
  // offload arrays, offload_baseptrs, offload_ptrs, offload_sizes.
  // Therefore:
  // i8** %offload_baseptrs.
  Value *BasePtrsArg =
      RuntimeCall.getArgOperand(OffloadArray::BasePtrsArgNum);
  // i8** %offload_ptrs.
  Value *PtrsArg = RuntimeCall.getArgOperand(OffloadArray::PtrsArgNum);
  // i8** %offload_sizes.
  Value *SizesArg = RuntimeCall.getArgOperand(OffloadArray::SizesArgNum);

  // Get values stored in **offload_baseptrs.
  auto *V = getUnderlyingObject(BasePtrsArg);
  if (!isa<AllocaInst>(V)) {
    LLVM_DEBUG(llvm::dbgs() << "Should be an alloca for BasePtrsArray but found " << *V << "\n");
    return false;
  }
  auto *BasePtrsArray = cast<AllocaInst>(V);
  if (!OAs[0].initialize(*BasePtrsArray, RuntimeCall)) {
    LLVM_DEBUG(llvm::dbgs() << "Initialization for BasePtrsArray not found in BB\n");
    return false;
  }

  // Get values stored in **offload_baseptrs.
  V = getUnderlyingObject(PtrsArg);
  if (!isa<AllocaInst>(V)) {
    LLVM_DEBUG(llvm::dbgs() << "Should be an alloca for PtrsArg but found " << *V << "\n");
    return false;
  }
  auto *PtrsArray = cast<AllocaInst>(V);
  if (!OAs[1].initialize(*PtrsArray, RuntimeCall)) {
    LLVM_DEBUG(llvm::dbgs() << "Initialization  for PtrsArg not found in BB\n");
    return false;
  }

  // Get values stored in **offload_sizes.
  V = getUnderlyingObject(SizesArg);
  // If it's a [constant] global array don't analyze it.
  if (isa<GlobalValue>(V))
    return isa<Constant>(V);
  if (!isa<AllocaInst>(V))
    return false;

  auto *SizesArray = cast<AllocaInst>(V);
  if (!OAs[2].initialize(*SizesArray, RuntimeCall))
    return false;

  return true;
}


bool OffloadArray::initialize(AllocaInst &Array, Instruction &Before)
{

  if (!Array.getAllocatedType()->isArrayTy()) {
    LLVM_DEBUG(llvm::dbgs() << Array << "is not  an array type\n");
    return false;
  }

  if (!getValues(Array, Before))
    return false;

  this->Array = &Array;
  return true;
}


bool OffloadArray::getValues(AllocaInst &Array, Instruction &Before)
{
  // Initialize container.
  const uint64_t NumValues = Array.getAllocatedType()->getArrayNumElements();
  StoredValues.assign(NumValues, nullptr);
  LastAccesses.assign(NumValues, nullptr);

  // TODO: This assumes the instruction \p Before is in the same
  //  BasicBlock as Array. Make it general, for any control flow graph.
  BasicBlock *BB = Array.getParent();
  if (BB != Before.getParent()) {
    LLVM_DEBUG(
        llvm::dbgs() << "[Array]: " << Array << "\n";
        llvm::dbgs() << "[Before]: " << Before << "\n";
        llvm::dbgs() << "[Problem]: They are not in the same BB\n";
        llvm::dbgs() << "[TEMP Hack SOULUTION]: Seatch initialization in the same BB of the call\n";);
    BB = Before.getParent();
  }

  const DataLayout &DL = Array.getModule()->getDataLayout();
  const unsigned int PointerSize = DL.getPointerSize();

  for (Instruction &I : *BB) {
    if (&I == &Before)
      break;

    if (!isa<StoreInst>(&I))
      continue;

    auto *S = cast<StoreInst>(&I);
    int64_t Offset = -1;
    auto *Dst =
        GetPointerBaseWithConstantOffset(S->getPointerOperand(), Offset, DL);
    if (Dst == &Array) {
      int64_t Idx = Offset / PointerSize;
      StoredValues[Idx] = getUnderlyingObject(S->getValueOperand());
      LastAccesses[Idx] = S;
    }
  }

  return isFilled();
}


bool OffloadArray::isFilled()
{
  const unsigned NumValues = StoredValues.size();
  for (unsigned I = 0; I < NumValues; ++I) {
    if (!StoredValues[I] || !LastAccesses[I]) {
      LLVM_DEBUG(llvm::dbgs() << "Not all value founded\n");
      return false;
    }
  }


  return true;
}
