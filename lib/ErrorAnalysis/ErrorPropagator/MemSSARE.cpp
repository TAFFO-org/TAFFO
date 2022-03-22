#include "MemSSARE.h"

namespace ErrorProp
{

using namespace llvm;
using namespace taffo;

MemSSARE::REVector &MemSSARE::getRangeErrors(llvm::Instruction *I, bool Sloppy)
{
  SmallVectorImpl<Value *> &DefVals = this->getDefiningValues(I);

  Res.clear();
  Res.reserve(DefVals.size());
  for (Value *V : DefVals) {
    const RangeErrorMap::RangeError *RE = RMap.getRangeError(V);
    if (RE != nullptr && RE->second.hasValue()) {
      Res.push_back(RE);
    }
  }

  // Kludje for when AliasAnalysis fails (i.e. almost always).
  if (Sloppy)
    findLOEError(I);

  return Res;
}

void MemSSARE::findLOEError(Instruction *I)
{
  Value *Pointer;
  switch (I->getOpcode()) {
  case Instruction::Load:
    Pointer = (cast<LoadInst>(I))->getPointerOperand();
    break;
  case Instruction::GetElementPtr:
    Pointer = (cast<GetElementPtrInst>(I))->getPointerOperand();
    break;
  case Instruction::BitCast:
    Pointer = (cast<BitCastInst>(I))->getOperand(0U);
    break;
  default:
    return;
  }
  const RangeErrorMap::RangeError *RE = RMap.getRangeError(Pointer);
  if (RE != nullptr && RE->second.hasValue()) {
    Res.push_back(RE);
  } else {
    Instruction *PI = dyn_cast<Instruction>(Pointer);
    if (PI != nullptr)
      findLOEError(PI);
  }
}

} // end of namespace ErrorProp
