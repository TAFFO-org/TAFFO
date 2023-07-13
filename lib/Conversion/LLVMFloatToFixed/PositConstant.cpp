#include "PositConstant.h"
#include "llvm/Analysis/ConstantFolding.h"
#include <posit.h>

#define DEBUG_TYPE "taffo-conversion"

using namespace llvm;
using namespace flttofix;

template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
static Constant *get(LLVMContext &C, const FixedPointType &fixpt, Posit<T,totalbits,esbits,FT,positspec> posit) {
  assert(totalbits == fixpt.scalarBitsAmt() && "Mismatching arguments");

  Constant *innerRepr;
  switch (fixpt.scalarBitsAmt()) {
  case 32:
    innerRepr = ConstantInt::getSigned(Type::getInt32Ty(C), posit.v);
    break;
  case 16:
    innerRepr = ConstantInt::getSigned(Type::getInt16Ty(C), posit.v);
    break;
  case 8:
    innerRepr = ConstantInt::getSigned(Type::getInt8Ty(C), posit.v);
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  StructType *type = cast<llvm::StructType>(fixpt.scalarToLLVMType(C));
  return ConstantStruct::get(type, { innerRepr });
}

template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
static Constant *FoldBinOp(LLVMContext &C, const FixedPointType &fixpt, int opcode,
                           Posit<T,totalbits,esbits,FT,positspec> x,
                           Posit<T,totalbits,esbits,FT,positspec> y) {
  Posit<T,totalbits,esbits,FT,positspec> res;
  switch (opcode) {
  case Instruction::FAdd:
    res = x + y;
    break;
  case Instruction::FSub:
    res = x - y;
    break;
  case Instruction::FMul:
    res = x * y;
    break;
  case Instruction::FDiv:
    res = x / y;
    break;
  default:
    LLVM_DEBUG(dbgs() << "Unimplemented constant Posit binary operation\n");
    return nullptr;
  }

  return get(C, fixpt, res);
}

template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
static Constant *FoldUnaryOp(LLVMContext &C, const FixedPointType &fixpt, int opcode,
                           Posit<T,totalbits,esbits,FT,positspec> x) {
  Posit<T,totalbits,esbits,FT,positspec> res;
  switch (opcode) {
  case Instruction::FNeg:
    res = -x;
    break;
  default:
    LLVM_DEBUG(dbgs() << "Unimplemented constant Posit unary operation");
    return nullptr;
  }

  return get(C, fixpt, res);
}

template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
static Constant *FoldConv(LLVMContext &C, const DataLayout *dl, const FixedPointType &fixpt, Posit<T,totalbits,esbits,FT,positspec> src, Type *dstType) {
  if (dstType->isDoubleTy() || dstType->isFloatTy()) {
    return ConstantFP::get(dstType, (double)src);
  } else if (dstType->isFloatingPointTy()) {
    // Convert to double then fold to dest type
    Constant *DblRes = ConstantFP::get(Type::getDoubleTy(C), (double)src);
    return ConstantFoldCastOperand(Instruction::FPTrunc, DblRes, dstType, *dl);
  } else if (dstType->isIntegerTy()) {
    return ConstantInt::get(dstType, (int64_t)src, true /* IsSigned */);
  } else {
    LLVM_DEBUG(dbgs() << "Unimplemented constant Posit conversion\n");
    return nullptr;
  }
}

Constant *PositConstant::get(LLVMContext &C, const FixedPointType &fixpt, double floatVal) {
  switch (fixpt.scalarBitsAmt()) {
  case 32:
    {
      Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf> posit(floatVal);
      return get(C, fixpt, posit);
    }
  case 16:
    {
      Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf> posit(floatVal);
      return get(C, fixpt, posit);
    }
  case 8:
    {
      Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf> posit(floatVal);
      return get(C, fixpt, posit);
    }
  default:
    llvm_unreachable("Unimplemented Posit size");
  }
}

Constant *PositConstant::FoldBinOp(LLVMContext &C, const FixedPointType &fixpt, int opcode, Constant *c1, Constant *c2) {
  ConstantInt *v1 = dyn_cast<ConstantInt>(c1->getAggregateElement(0U));
  ConstantInt *v2 = dyn_cast<ConstantInt>(c2->getAggregateElement(0U));
  assert((v1 && v2) && "Expected two Posit structs");

  switch (fixpt.scalarBitsAmt()) {
  case 32:
    {
      Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf> x(
          Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf>::DeepInit(),
          (int32_t)v1->getSExtValue());
      Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf> y(
          Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf>::DeepInit(),
          (int32_t)v2->getSExtValue());
      return FoldBinOp(C, fixpt, opcode, x, y);
    }
  case 16:
    {
      Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf> x(
          Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf>::DeepInit(),
          (int16_t)v1->getSExtValue());
      Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf> y(
          Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf>::DeepInit(),
          (int16_t)v2->getSExtValue());
      return FoldBinOp(C, fixpt, opcode, x, y);
    }
  case 8:
    {
      Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf> x(
          Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf>::DeepInit(),
          (int8_t)v1->getSExtValue());
      Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf> y(
          Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf>::DeepInit(),
          (int8_t)v2->getSExtValue());
      return FoldBinOp(C, fixpt, opcode, x, y);
    }
  default:
    llvm_unreachable("Unimplemented Posit size");
  }
}

Constant *PositConstant::FoldUnaryOp(LLVMContext &C, const FixedPointType &fixpt, int opcode, Constant *c) {
  ConstantInt *v = dyn_cast<ConstantInt>(c->getAggregateElement(0U));
  assert(v && "Expected a Posit struct");

  switch (fixpt.scalarBitsAmt()) {
  case 32:
    {
      Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf> x(
          Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf>::DeepInit(),
          (int32_t)v->getSExtValue());
      return FoldUnaryOp(C, fixpt, opcode, x);
    }
  case 16:
    {
      Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf> x(
          Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf>::DeepInit(),
          (int16_t)v->getSExtValue());
      return FoldUnaryOp(C, fixpt, opcode, x);
    }
  case 8:
    {
      Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf> x(
          Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf>::DeepInit(),
          (int8_t)v->getSExtValue());
      return FoldUnaryOp(C, fixpt, opcode, x);
    }
  default:
    llvm_unreachable("Unimplemented Posit size");
  }
}

Constant *PositConstant::FoldConv(LLVMContext &C, const DataLayout *dl, const FixedPointType &fixpt, Constant *src, Type *dstType) {
  ConstantInt *v = dyn_cast<ConstantInt>(src->getAggregateElement(0U));
  assert(v && "Expected a Posit struct");

  switch (fixpt.scalarBitsAmt()) {
  case 32:
    {
      Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf> x(
          Posit<int32_t, 32, 2, uint32_t, PositSpec::WithInf>::DeepInit(),
          (int32_t)v->getSExtValue());
      return FoldConv(C, dl, fixpt, x, dstType);
    }
  case 16:
    {
      Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf> x(
          Posit<int16_t, 16, 2, uint16_t, PositSpec::WithInf>::DeepInit(),
          (int16_t)v->getSExtValue());
      return FoldConv(C, dl, fixpt, x, dstType);
    }
  case 8:
    {
      Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf> x(
          Posit<int8_t, 8, 2, uint8_t, PositSpec::WithInf>::DeepInit(),
          (int8_t)v->getSExtValue());
      return FoldConv(C, dl, fixpt, x, dstType);
    }
  default:
    llvm_unreachable("Unimplemented Posit size");
  }
}
