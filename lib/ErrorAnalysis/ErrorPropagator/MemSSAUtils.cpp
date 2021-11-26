#include "MemSSAUtils.h"

namespace ErrorProp {

using namespace llvm;

void MemSSAUtils::findLOEError(Instruction *I) {
  Value *Pointer;
  switch(I->getOpcode()) {
    case Instruction::Load:
      Pointer = (cast<LoadInst>(I))->getPointerOperand();
      break;
    case Instruction::GetElementPtr:
      Pointer = (cast<GetElementPtrInst>(I))->getPointerOperand();
      break;
    case Instruction::BitCast:
      Pointer = (cast<BitCastInst>(I))->getOperand(0U);
    default:
      return;
  }
  const RangeErrorMap::RangeError *RE = RMap.getRangeError(Pointer);
  if (RE != nullptr && RE->second.hasValue()) {
    Res.push_back(RE);
  }
  else {
    Instruction *PI = dyn_cast<Instruction>(Pointer);
    if (PI != nullptr)
      findLOEError(PI);
  }
}

void MemSSAUtils::findMemDefError(Instruction *I, const MemoryDef *MD) {
  assert(MD != nullptr && "MD is null.");

  Instruction *MI = MD->getMemoryInst();
  if (isa<CallInst>(MI) || isa<InvokeInst>(MI))
    // The computed error should have been attached to the actual parameter.
    findLOEError(I);
  else
    Res.push_back(RMap.getRangeError(MD->getMemoryInst()));
}

void MemSSAUtils::
findMemPhiError(Instruction *I, MemoryPhi *MPhi) {
  assert(MPhi != nullptr && "MPhi is null.");

  for (Use &MU : MPhi->incoming_values()) {
    MemoryAccess *MA = cast<MemoryAccess>(&MU);
    findMemSSAError(I, MA);
  }
}

void MemSSAUtils::
findMemSSAError(Instruction *I, MemoryAccess *MA) {
  if (MA == nullptr) {
    LLVM_DEBUG(dbgs() << "WARNING: nullptr MemoryAccess passed to findMemSSAError!\n");
    return;
  }

  if (!Visited.insert(MA).second)
    return;

  if (MemSSA.isLiveOnEntryDef(MA))
    findLOEError(I);
  else if (isa<MemoryUse>(MA)) {
    MemorySSAWalker *MSSAWalker = MemSSA.getWalker();
    assert(MSSAWalker != nullptr && "Null MemorySSAWalker.");
    findMemSSAError(I, MSSAWalker->getClobberingMemoryAccess(MA));
  }
  else if (isa<MemoryDef>(MA))
    findMemDefError(I, cast<MemoryDef>(MA));
  else {
    assert(isa<MemoryPhi>(MA));
    findMemPhiError(I, cast<MemoryPhi>(MA));
  }
}

Value *MemSSAUtils::getOriginPointer(MemorySSA &MemSSA, Value *Pointer) {
  assert(Pointer != nullptr);

  if (isa<Argument>(Pointer) || isa<AllocaInst>(Pointer)) {
    return Pointer;
  }
  else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Pointer)) {
    return getOriginPointer(MemSSA, GEPI->getPointerOperand());
  }
  else if (BitCastInst *BCI = dyn_cast<BitCastInst>(Pointer)) {
    return getOriginPointer(MemSSA, BCI->getOperand(0U));
  }
  else if (LoadInst *LI = dyn_cast<LoadInst>(Pointer)) {
    MemorySSAWalker *MSSAWalker = MemSSA.getWalker();
    assert(MSSAWalker != nullptr && "Null MemorySSAWalker.");
    if (MemoryDef *MD = dyn_cast<MemoryDef>(MSSAWalker->getClobberingMemoryAccess(LI))) {
      if (MemSSA.isLiveOnEntryDef(MD)) {
	return getOriginPointer(MemSSA, LI->getPointerOperand());
      }
      else if (StoreInst *SI = dyn_cast<StoreInst>(MD->getMemoryInst())) {
	return getOriginPointer(MemSSA, SI->getValueOperand());
      }
    }
    // TODO: Handle MemoryPHI
    return getOriginPointer(MemSSA, LI->getPointerOperand());
  }
  return nullptr;
}

} // end of namespace ErrorProp
