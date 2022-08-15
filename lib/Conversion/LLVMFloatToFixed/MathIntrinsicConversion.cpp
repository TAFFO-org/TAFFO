#include "LLVMFloatToFixedPass.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NoFolder.h"

using namespace llvm;
using namespace flttofix;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

enum MathIntrinsicFamily: unsigned {
  Unrecognized = 0,
  FMA,
  FMulAdd
};

static MathIntrinsicFamily getMathIntrinsicFamily(Function *F)
{
  if (F->getIntrinsicID() == Intrinsic::fma)
    return MathIntrinsicFamily::FMA;
  if (F->getName() == "fma")
    return MathIntrinsicFamily::FMA;
  if (F->getName() == "fmal")
    return MathIntrinsicFamily::FMA;
  if (F->getName() == "fmaf")
    return MathIntrinsicFamily::FMA;
  if (F->getIntrinsicID() == Intrinsic::fmuladd)
    return MathIntrinsicFamily::FMulAdd;
  return MathIntrinsicFamily::Unrecognized;
}

bool FloatToFixed::isSupportedMathIntrinsicFunction(Function *F)
{
  return getMathIntrinsicFamily(F) != MathIntrinsicFamily::Unrecognized;
}

Value *FloatToFixed::convertMathIntrinsicFunction(CallBase *C, FixedPointType &fixpt)
{
  /* Use the normal fallback path to handle non-converted values */
  if (valueInfo(C)->noTypeConversion)
    return Unsupported;

  Function *F = C->getCalledFunction();
  MathIntrinsicFamily Fam = getMathIntrinsicFamily(F);

  if (Fam == MathIntrinsicFamily::FMA || Fam == MathIntrinsicFamily::FMulAdd) {
    Value *Op1 = C->getArgOperand(0);
    Value *Op2 = C->getArgOperand(1);
    Value *Op3 = C->getArgOperand(2);

    if (fixpt.isFixedPoint()) {
      FixedPointType intype1 = fixpt;
      FixedPointType intype2 = fixpt;
      Value *val1 = translateOrMatchOperand(Op1, intype1, C, TypeMatchPolicy::RangeOverHintMaxInt);
      Value *val2 = translateOrMatchOperand(Op2, intype2, C, TypeMatchPolicy::RangeOverHintMaxInt);
      Value *val3 = translateOrMatchOperandAndType(Op3, fixpt, C);
      if (!val1 || !val2 || !val3)
        return nullptr;
      
      FixedPointType intermtype(
          fixpt.scalarIsSigned(),
          intype1.scalarFracBitsAmt() + intype2.scalarFracBitsAmt(),
          intype1.scalarBitsAmt() + intype2.scalarBitsAmt());
      Type *dbfxt = intermtype.scalarToLLVMType(C->getContext());

      IRBuilder<NoFolder> builder(C);
      Value *ext1 = intype1.scalarIsSigned() ? builder.CreateSExt(val1, dbfxt)
                                             : builder.CreateZExt(val1, dbfxt);
      Value *ext2 = intype2.scalarIsSigned() ? builder.CreateSExt(val2, dbfxt)
                                             : builder.CreateZExt(val2, dbfxt);
      Value *fixop = builder.CreateMul(ext1, ext2);
      Value *fixopcvt = genConvertFixedToFixed(fixop, intermtype, fixpt, C);
      Value *res = builder.CreateAdd(fixopcvt, val3);

      cpMetaData(ext1, val1);
      cpMetaData(ext2, val2);
      cpMetaData(fixop, C);
      cpMetaData(res, C);
      updateFPTypeMetadata(fixop, intermtype.scalarIsSigned(),
                           intermtype.scalarFracBitsAmt(),
                           intermtype.scalarBitsAmt());
      updateFPTypeMetadata(res, fixpt.scalarIsSigned(),
                           fixpt.scalarFracBitsAmt(),
                           fixpt.scalarBitsAmt());
      updateConstTypeMetadata(fixop, 0U, intype1);
      updateConstTypeMetadata(fixop, 1U, intype2);
      updateConstTypeMetadata(res, 0U, fixpt);
      updateConstTypeMetadata(res, 1U, fixpt);

      return res;
    } else if (fixpt.isFloatingPoint()) {
      Value *val1 = translateOrMatchOperandAndType(Op1, fixpt, C);
      Value *val2 = translateOrMatchOperandAndType(Op2, fixpt, C);
      Value *val3 = translateOrMatchOperandAndType(Op3, fixpt, C);
      if (!val1 || !val2 || !val3)
        return nullptr;
      Type *Ty = fixpt.scalarToLLVMType(C->getContext());
      Function *NewIntrins;
      if (Fam == MathIntrinsicFamily::FMA) {
        NewIntrins = Intrinsic::getDeclaration(C->getModule(), Intrinsic::fma, {Ty});
      } else {
        NewIntrins = Intrinsic::getDeclaration(C->getModule(), Intrinsic::fmuladd, {Ty});
      }
      IRBuilder<NoFolder> Builder(C);
      Value *Res = Builder.CreateCall(NewIntrins->getFunctionType(), NewIntrins, {val1, val2, val3});
      return Res;
    } else {
      llvm_unreachable("Unknown variable type. Are you trying to implement a new datatype?");
    }
  }
  
  llvm_unreachable("math intrinsic recognized but not handled!");
}

