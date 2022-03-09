#include "MemSSAUtils.hpp"
#include "llvm/IR/Instructions.h"

namespace taffo
{

using namespace llvm;

SmallVectorImpl<Value *> &
MemSSAUtils::getDefiningValues(Instruction *i)
{
  Visited.clear();
  Res.clear();

  findClobberingValues(i, MemSSA.getMemoryAccess(i));

  return Res;
}

void MemSSAUtils::findLOEValue(Instruction *i)
{
  Value *pointer = (cast<LoadInst>(i))->getPointerOperand();
  Res.push_back(getOriginPointer(MemSSA, pointer));
}

void MemSSAUtils::findMemDefValue(Instruction *i, const MemoryDef *md)
{
  assert(md != nullptr && "md is null.");

  Instruction *mi = md->getMemoryInst();
  if (isa<CallInst>(mi) || isa<InvokeInst>(mi))
    // The computed error should have been attached to the actual parameter.
    findLOEValue(i);
  else
    Res.push_back(mi);
}

void MemSSAUtils::
    findMemPhiValue(Instruction *i, MemoryPhi *mphi)
{
  assert(mphi != nullptr && "mphi is null.");

  for (Use &mu : mphi->incoming_values()) {
    MemoryAccess *ma = cast<MemoryAccess>(&mu);
    findClobberingValues(i, ma);
  }
}

void MemSSAUtils::
    findClobberingValues(Instruction *i, MemoryAccess *ma)
{
  if (ma == nullptr) {
    return;
  }

  if (!Visited.insert(ma).second)
    return;

  if (MemSSA.isLiveOnEntryDef(ma))
    findLOEValue(i);
  else if (isa<MemoryUse>(ma)) {
    MemorySSAWalker *MSSAWalker = MemSSA.getWalker();
    assert(MSSAWalker != nullptr && "Null MemorySSAWalker.");
    findClobberingValues(i, MSSAWalker->getClobberingMemoryAccess(ma));
  } else if (isa<MemoryDef>(ma))
    findMemDefValue(i, cast<MemoryDef>(ma));
  else {
    assert(isa<MemoryPhi>(ma));
    findMemPhiValue(i, cast<MemoryPhi>(ma));
  }
}

Value *MemSSAUtils::getOriginPointer(MemorySSA &MemSSA, Value *Pointer)
{
  assert(Pointer != nullptr);

  if (isa<Argument>(Pointer) || isa<AllocaInst>(Pointer) || isa<GlobalVariable>(Pointer)) {
    return Pointer;
  } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Pointer)) {
    return getOriginPointer(MemSSA, GEPI->getPointerOperand());
  } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(Pointer)) {
    return getOriginPointer(MemSSA, BCI->getOperand(0U));
  } else if (LoadInst *LI = dyn_cast<LoadInst>(Pointer)) {
    MemorySSAWalker *MSSAWalker = MemSSA.getWalker();
    assert(MSSAWalker != nullptr && "Null MemorySSAWalker.");
    if (MemoryDef *MD = dyn_cast<MemoryDef>(MSSAWalker->getClobberingMemoryAccess(LI))) {
      if (MemSSA.isLiveOnEntryDef(MD)) {
        return getOriginPointer(MemSSA, LI->getPointerOperand());
      } else if (StoreInst *SI = dyn_cast<StoreInst>(MD->getMemoryInst())) {
        return getOriginPointer(MemSSA, SI->getValueOperand());
      }
    }
    // TODO: Handle MemoryPHI
    return getOriginPointer(MemSSA, LI->getPointerOperand());
  }
  return nullptr;
}

} // end of namespace taffo
