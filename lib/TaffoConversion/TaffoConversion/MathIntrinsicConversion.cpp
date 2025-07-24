#include "ConversionPass.hpp"
#include "TransparentType.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/NoFolder.h>

using namespace llvm;
using namespace taffo;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

enum MathIntrinsicFamily : unsigned {
  Unrecognized = 0,
  FMA,
  FMulAdd
};

static MathIntrinsicFamily getMathIntrinsicFamily(Function* F) {
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

bool ConversionPass::isSupportedMathIntrinsicFunction(Function* F) {
  return getMathIntrinsicFamily(F) != MathIntrinsicFamily::Unrecognized;
}

Value* ConversionPass::convertMathIntrinsicFunction(CallBase* C, const std::shared_ptr<FixedPointScalarType>& fixpt) {
  /* Use the normal fallback path to handle non-converted values */
  auto& taffoInfo = TaffoInfo::getInstance();
  if (getConversionInfo(C)->noTypeConversion)
    return Unsupported;

  Function* F = C->getCalledFunction();
  MathIntrinsicFamily Fam = getMathIntrinsicFamily(F);

  if (Fam == MathIntrinsicFamily::FMA || Fam == MathIntrinsicFamily::FMulAdd) {
    Value* Op1 = C->getArgOperand(0);
    Value* Op2 = C->getArgOperand(1);
    Value* Op3 = C->getArgOperand(2);

    if (fixpt->isFixedPoint()) {
      std::shared_ptr<FixedPointType> intype1 = fixpt->clone();
      std::shared_ptr<FixedPointType> intype2 = fixpt->clone();
      Value* val1 = translateOrMatchOperand(Op1, intype1, C, TypeMatchPolicy::RangeOverHintMaxInt);
      Value* val2 = translateOrMatchOperand(Op2, intype2, C, TypeMatchPolicy::RangeOverHintMaxInt);
      Value* val3 = translateOrMatchOperandAndType(Op3, fixpt, C);
      std::shared_ptr<FixedPointScalarType> scalarIntype1 = std::static_ptr_cast<FixedPointScalarType>(intype1);
      std::shared_ptr<FixedPointScalarType> scalarIntype2 = std::static_ptr_cast<FixedPointScalarType>(intype2);
      if (!val1 || !val2 || !val3)
        return nullptr;
      std::shared_ptr<FixedPointScalarType> intermtype =
        std::make_shared<FixedPointScalarType>(fixpt->isSigned(),
                                               scalarIntype1->getBits() + scalarIntype2->getBits(),
                                               scalarIntype1->getFractionalBits() + scalarIntype2->getFractionalBits());
      Type* dbfxt = intermtype->scalarToLLVMType(C->getContext());

      IRBuilder<NoFolder> builder(C);
      Value* ext1 = nullptr;
      Value* ext2 = nullptr;
      Value* fixop = nullptr;

      if (dbfxt->getScalarSizeInBits() > MaxTotalBitsConv) {
        dbfxt = fixpt->scalarToLLVMType(C->getContext());

        ext1 = val1;
        ext2 = val2;

        auto make_to_same_size = [this, C, &builder](std::shared_ptr<FixedPointScalarType>& from,
                                                     std::shared_ptr<FixedPointScalarType>& to,
                                                     Value*& ext,
                                                     Value* val) {
          auto llvmfrom = from->scalarToLLVMType(C->getContext());
          auto llvmto = to->scalarToLLVMType(C->getContext());
          ext = from->isSigned() ? builder.CreateSExt(val, llvmto) : builder.CreateZExt(val, llvmto);

          auto diff = llvmto->getScalarSizeInBits() - llvmfrom->getScalarSizeInBits();
          // create metadata same as val2 but more bits
          copyValueInfo(ext, val);
          updateFPTypeMetadata(ext, from->isSigned(), from->getFractionalBits(), from->getBits() + diff);

          ext = builder.CreateShl(ext, diff);
          // create metadata same as val2 but more bits and appropriate scalar frac
          copyValueInfo(ext, val);
          updateFPTypeMetadata(ext, from->isSigned(), from->getFractionalBits() + diff, from->getBits() + diff);

          // update inttype2 to correct type
          from->setBits(from->getBits() + diff);
          from->setFractionalBits(from->getFractionalBits() + diff);
        };

        // Adjust to same size
        if (scalarIntype1->getBits() > scalarIntype2->getBits())
          make_to_same_size(scalarIntype2, scalarIntype1, ext2, val2);
        else if (scalarIntype1->getBits() < scalarIntype2->getBits())
          make_to_same_size(scalarIntype1, scalarIntype2, ext1, val1);

        const auto frac1_s = scalarIntype1->getFractionalBits();
        const auto frac2_s = scalarIntype2->getFractionalBits();

        auto target_frac = fixpt->getFractionalBits();
        int shift_right1 = 0;
        int shift_right2 = 0;
        auto new_frac1 = frac1_s;
        auto new_frac2 = frac2_s;

        if (target_frac % 2 == 0) {
          auto required_fract = target_frac / 2;
          shift_right1 = frac1_s - required_fract;
          shift_right2 = frac2_s - required_fract;
        }
        else {
          auto required_fract = target_frac / 2;
          if (frac1_s > frac2_s) {
            shift_right1 = frac1_s - (required_fract + 1);
            shift_right2 = frac2_s - required_fract;
          }
          else {

            shift_right2 = frac2_s - (required_fract + 1);
            shift_right1 = frac1_s - required_fract;
          }
        }

        // create shift to make space for all possible value
        if (shift_right1 > 0) {

          ext1 =
            scalarIntype1->isSigned() ? builder.CreateAShr(ext1, shift_right1) : builder.CreateLShr(ext1, shift_right1);
          new_frac1 = new_frac1 - shift_right1;
        }
        if (shift_right2 > 0) {
          ext2 =
            scalarIntype2->isSigned() ? builder.CreateAShr(ext2, shift_right2) : builder.CreateLShr(ext2, shift_right2);
          new_frac2 = new_frac2 - shift_right2;
        }

        auto new_frac = new_frac1 + new_frac2;

        scalarIntype1->setFractionalBits(new_frac1);
        scalarIntype2->setFractionalBits(new_frac2);

        copyValueInfo(ext1, val1);
        updateFPTypeMetadata(
          ext1, scalarIntype1->isSigned(), scalarIntype1->getFractionalBits(), scalarIntype1->getBits());
        copyValueInfo(ext2, val2);
        updateFPTypeMetadata(
          ext2, scalarIntype2->isSigned(), scalarIntype2->getFractionalBits(), scalarIntype2->getBits());
        intermtype->setBits(scalarIntype1->getBits());
        intermtype->setFractionalBits(new_frac);
        fixop = builder.CreateMul(ext1, ext2);
      }
      else {
        ext1 = scalarIntype1->isSigned() ? builder.CreateSExt(val1, dbfxt) : builder.CreateZExt(val1, dbfxt);
        ext2 = scalarIntype2->isSigned() ? builder.CreateSExt(val2, dbfxt) : builder.CreateZExt(val2, dbfxt);
        fixop = builder.CreateMul(ext1, ext2);
        copyValueInfo(ext1, val1);
        copyValueInfo(ext2, val2);
      }

      copyValueInfo(fixop, C);

      Value* fixopcvt = genConvertFixedToFixed(fixop, intermtype, fixpt, C);
      Value* res = builder.CreateAdd(fixopcvt, val3);

      copyValueInfo(res, C);
      updateFPTypeMetadata(fixop, intermtype->isSigned(), intermtype->getFractionalBits(), intermtype->getBits());
      updateFPTypeMetadata(res, fixpt->isSigned(), fixpt->getFractionalBits(), fixpt->getBits());
      updateConstTypeMetadata(fixop, 0U, scalarIntype1);
      updateConstTypeMetadata(fixop, 1U, scalarIntype2);
      updateConstTypeMetadata(res, 0U, fixpt);
      updateConstTypeMetadata(res, 1U, fixpt);

      return res;
    }
    else if (fixpt->isFloatingPoint()) {
      Value* val1 = translateOrMatchOperandAndType(Op1, fixpt, C);
      Value* val2 = translateOrMatchOperandAndType(Op2, fixpt, C);
      Value* val3 = translateOrMatchOperandAndType(Op3, fixpt, C);
      if (!val1 || !val2 || !val3)
        return nullptr;
      Type* Ty = fixpt->scalarToLLVMType(C->getContext());
      Function* NewIntrins;
      if (Fam == MathIntrinsicFamily::FMA)
        NewIntrins = Intrinsic::getDeclaration(C->getModule(), Intrinsic::fma, {Ty});
      else
        NewIntrins = Intrinsic::getDeclaration(C->getModule(), Intrinsic::fmuladd, {Ty});
      IRBuilder<NoFolder> Builder(C);
      Value* Res = Builder.CreateCall(NewIntrins->getFunctionType(), NewIntrins, {val1, val2, val3});
      return Res;
    }
    else {
      llvm_unreachable("Unknown variable type. Are you trying to implement a new datatype?");
    }
  }
  llvm_unreachable("math intrinsic recognized but not handled!");
}
