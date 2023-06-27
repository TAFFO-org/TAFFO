#include "LLVMFloatToFixedPass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/NoFolder.h"
#include "PositBuilder.h"

using namespace llvm;
using namespace flttofix;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

enum MathIntrinsicFamily : unsigned {
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
      Value *ext1 = nullptr;
      Value *ext2 = nullptr;
      Value *fixop = nullptr;

      if (dbfxt->getScalarSizeInBits() > MaxTotalBitsConv) {
        dbfxt = fixpt.scalarToLLVMType(C->getContext());

        ext1 = val1;
        ext2 = val2;

        auto make_to_same_size = [this, C, &builder](FixedPointType &from, FixedPointType &to, Value *&ext, Value *val) {
          auto llvmfrom = from.scalarToLLVMType(C->getContext());
          auto llvmto = to.scalarToLLVMType(C->getContext());
          ext = from.scalarIsSigned() ? builder.CreateSExt(val, llvmto)
                                      : builder.CreateZExt(val, llvmto);

          auto diff = llvmto->getScalarSizeInBits() - llvmfrom->getScalarSizeInBits();
          // create metadata same as val2 but more bits
          cpMetaData(ext, val);
          updateFPTypeMetadata(ext, from.scalarIsSigned(), from.scalarFracBitsAmt(), from.scalarBitsAmt() + diff);

          ext = builder.CreateShl(ext, diff);
          // create metadata same as val2 but more bits and appropriate scalar frac
          cpMetaData(ext, val);
          updateFPTypeMetadata(ext, from.scalarIsSigned(), from.scalarFracBitsAmt() + diff, from.scalarBitsAmt() + diff);

          // update inttype2 to correct type
          from.scalarBitsAmt() = from.scalarBitsAmt() + diff;
          from.scalarFracBitsAmt() = from.scalarFracBitsAmt() + diff;
        };


        // Adjust to same size
        if (intype1.scalarBitsAmt() > intype2.scalarBitsAmt()) {
          make_to_same_size(intype2, intype1, ext2, val2);
        } else if (intype1.scalarBitsAmt() < intype2.scalarBitsAmt()) {
          make_to_same_size(intype1, intype2, ext1, val1);
        }


        const auto frac1_s = intype1.scalarFracBitsAmt();
        const auto frac2_s = intype2.scalarFracBitsAmt();

        auto target_frac = fixpt.scalarFracBitsAmt();
        int shift_right1 = 0;
        int shift_right2 = 0;
        auto new_frac1 = frac1_s;
        auto new_frac2 = frac2_s;

        if (target_frac % 2 == 0) {
          auto required_fract = target_frac / 2;
          shift_right1 = frac1_s - required_fract;
          shift_right2 = frac2_s - required_fract;
        } else {
          auto required_fract = target_frac / 2;
          if (frac1_s > frac2_s) {
            shift_right1 = frac1_s - (required_fract + 1);
            shift_right2 = frac2_s - required_fract;
          } else {

            shift_right2 = frac2_s - (required_fract + 1);
            shift_right1 = frac1_s - required_fract;
          }
        }

        // create shift to make space for all possible value
        if (shift_right1 > 0) {

          ext1 = intype1.scalarIsSigned() ? builder.CreateAShr(ext1, shift_right1)
                                          : builder.CreateLShr(ext1, shift_right1);
          new_frac1 = new_frac1 - shift_right1;
        }
        if (shift_right2 > 0) {
          ext2 = intype2.scalarIsSigned() ? builder.CreateAShr(ext2, shift_right2)
                                          : builder.CreateLShr(ext2, shift_right2);
          new_frac2 = new_frac2 - shift_right2;
        }

        auto new_frac = new_frac1 + new_frac2;


        intype1.scalarFracBitsAmt() = new_frac1;
        intype2.scalarFracBitsAmt() = new_frac2;

        cpMetaData(ext1, val1);
        updateFPTypeMetadata(ext1, intype1.scalarIsSigned(), intype1.scalarFracBitsAmt(), intype1.scalarBitsAmt());
        cpMetaData(ext2, val2);
        updateFPTypeMetadata(ext2, intype2.scalarIsSigned(), intype2.scalarFracBitsAmt(), intype2.scalarBitsAmt());
        intermtype.scalarBitsAmt() = intype1.scalarBitsAmt();
        intermtype.scalarFracBitsAmt() = new_frac;
        fixop = builder.CreateMul(ext1, ext2);


      } else {
        ext1 = intype1.scalarIsSigned() ? builder.CreateSExt(val1, dbfxt)
                                        : builder.CreateZExt(val1, dbfxt);
        ext2 = intype2.scalarIsSigned() ? builder.CreateSExt(val2, dbfxt)
                                        : builder.CreateZExt(val2, dbfxt);
        fixop = builder.CreateMul(ext1, ext2);
        cpMetaData(ext1, val1);
        cpMetaData(ext2, val2);
      }
      Value *fixopcvt = genConvertFixedToFixed(fixop, intermtype, fixpt, C);
      Value *res = builder.CreateAdd(fixopcvt, val3);

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
    } else if (fixpt.isPosit()) {
      Value *val1 = translateOrMatchOperandAndType(Op1, fixpt, C);
      Value *val2 = translateOrMatchOperandAndType(Op2, fixpt, C);
      Value *val3 = translateOrMatchOperandAndType(Op3, fixpt, C);
      if (!val1 || !val2 || !val3)
        return nullptr;
      IRBuilder<NoFolder> builder(C);
      return PositBuilder(this, builder, fixpt).CreateFMA(val1, val2, val3);
    } else {
      llvm_unreachable("Unknown variable type. Are you trying to implement a new datatype?");
    }
  }
  llvm_unreachable("math intrinsic recognized but not handled!");
}
