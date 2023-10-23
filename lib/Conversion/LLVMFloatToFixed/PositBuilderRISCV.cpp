#include "PositBuilderRISCV.h"
#include "PositConstant.h"
#include "llvm/IR/InlineAsm.h"

using namespace llvm;
using namespace flttofix;

#define DEBUG_TYPE "taffo-conversion"

/**
 * Emits assembly instructions for the "PPU" RISCV extension by Università degli Studi di Pisa
 * See: https://github.com/federicorossifr/ppu_public
 * See: https://github.com/federicorossifr/ibex_ppu_pv1
 * 
 * At the time of writing the hardware implements:
 * - Converting from/to floats (configurable size, here we assume float32)
 * - Addition, subtraction, multiplication, division
 * - Comparison -- because the ordering of two posits is exactly the integer ordering between their bit representation
 * 
 * BUG: We seem need a nop between PPU operations
 */

Value *PositBuilderRISCV::CreateConstructor(Value *arg1, const FixedPointType *srcMetadata) {
  if (srcMetadata && srcMetadata->isPosit())
    llvm_unreachable("The RISCV PPU does not support multiple posit types");
  if (srcMetadata && srcMetadata->isFixedPoint() && srcMetadata->scalarFracBitsAmt() > 0)
    llvm_unreachable("The RISCV PPU does not support creating a posit from a fixed-point.\n"
                     "TODO: Consider an intermediate step through floats");

  Type *srcType = arg1->getType();
  if (srcType->isIntegerTy()) {
    if (!srcMetadata || srcMetadata->scalarIsSigned())
      arg1 = builder.CreateSIToFP(arg1, Type::getFloatTy(C));
    else
      arg1 = builder.CreateUIToFP(arg1, Type::getFloatTy(C));
  } else if (srcType->isFloatingPointTy() && !srcType->isFloatTy()) {
    arg1 = builder.CreateFPCast(arg1, Type::getFloatTy(C));
  }

  FunctionType *opType = FunctionType::get(
      IntegerType::get(C, metadata.scalarBitsAmt()), /* Return type */
      { Type::getFloatTy(C) }, /* Arguments... */
      false /* isVarArg */
  );

  Value *ret = builder.CreateCall(InlineAsm::get(
      opType,
      ".insn r 0xb, 0, 0x68, $0, $1, x0; nop",
      "=r,r",
      false /* hasSideEffects */
  ), { arg1 });

  return builder.CreateInsertValue(UndefValue::get(llvmType), ret, {0});
}


Value *PositBuilderRISCV::CreateConv(Value *from, Type *dstType, const FixedPointType *dstMetadata) {
  if (dstMetadata && dstMetadata->isPosit())
    llvm_unreachable("The RISCV PPU does not support multiple posit types");
  if (dstMetadata && dstMetadata->isFixedPoint() && dstMetadata->scalarFracBitsAmt() > 0)
    llvm_unreachable("The RISCV PPU does not support converting a posit to a fixed-point.\n"
                     "TODO: Consider an intermediate step through floats");

  FunctionType *opType = FunctionType::get(
      Type::getFloatTy(C), /* Return type */
      { IntegerType::get(C, metadata.scalarBitsAmt()) }, /* Arguments... */
      false /* isVarArg */
  );

  from = builder.CreateExtractValue(from, {0});

  Value *ret = builder.CreateCall(InlineAsm::get(
      opType,
      ".insn r 0xb, 0, 0x69, $0, $1, x0; nop",
      "=r,r",
      false /* hasSideEffects */
  ), { from });

  if (dstType->isIntegerTy()) {
    if (!dstMetadata || dstMetadata->scalarIsSigned())
      ret = builder.CreateFPToSI(ret, dstType);
    else
      ret = builder.CreateFPToUI(ret, dstType);
  } else if (dstType->isFloatingPointTy() && !dstType->isFloatTy()) {
    ret = builder.CreateFPCast(ret, dstType);
  }

  return ret;
}


Value *PositBuilderRISCV::CreateBinOp(int opcode, Value *arg1, Value *arg2) {
  if (Constant *c1 = dyn_cast<Constant>(arg1)) {
    if (Constant *c2 = dyn_cast<Constant>(arg2)) {
      LLVM_DEBUG(dbgs() << "Attempting to fold constant Posit operation\n");
      Constant *res = PositConstant::FoldBinOp(C, metadata, opcode, c1, c2);
      if (res) {
        LLVM_DEBUG(dbgs() << "Folded in " << *res << "\n");
        return res;
      } else {
        LLVM_DEBUG(dbgs() << "Constant folding failed; falling back to runtime computation\n");
      }
    }
  }

  const char *asmInstr;
  switch (opcode) {
  case Instruction::FAdd:
    asmInstr = ".insn r 0xb, 0, 0x6A, $0, $1, $2; nop";
    break;
  case Instruction::FSub:
    asmInstr = ".insn r 0xb, 1, 0x6A, $0, $1, $2; nop";
    break;
  case Instruction::FMul:
    asmInstr = ".insn r 0xb, 2, 0x6A, $0, $1, $2; nop";
    break;
  case Instruction::FDiv:
    asmInstr = ".insn r 0xb, 4, 0x6A, $0, $1, $2; nop";
    break;
  default:
    llvm_unreachable("Unimplemented Posit binary operation");
  }

  Type *positRawType = IntegerType::get(C, metadata.scalarBitsAmt());
  FunctionType *opType = FunctionType::get(
      positRawType, /* Return type */
      { positRawType, positRawType }, /* Arguments... */
      false /* isVarArg */
  );

  arg1 = builder.CreateExtractValue(arg1, {0});
  arg2 = builder.CreateExtractValue(arg2, {0});

  Value *ret = builder.CreateCall(InlineAsm::get(
      opType,
      asmInstr,
      "=r,r,r",
      false /* hasSideEffects */
  ), { arg1, arg2 });

  return builder.CreateInsertValue(UndefValue::get(llvmType), ret, {0});
}

Value *PositBuilderRISCV::CreateUnaryOp(int opcode, Value *arg1) {
  if (Constant *c = dyn_cast<Constant>(arg1)) {
    LLVM_DEBUG(dbgs() << "Attempting to fold constant Posit operation\n");
    Constant *res = PositConstant::FoldUnaryOp(C, metadata, opcode, c);
    if (res) {
      LLVM_DEBUG(dbgs() << "Folded in " << *res << "\n");
      return res;
    } else {
      LLVM_DEBUG(dbgs() << "Constant folding failed; falling back to runtime computation\n");
    }
  }

  const char *asmInstr;
  switch (opcode) {
  case Instruction::FNeg:
    asmInstr = ".insn r 0xb, 1, 0x6A, $0, x0, $1; nop";
    break;
  }

  Type *positRawType = IntegerType::get(C, metadata.scalarBitsAmt());
  FunctionType *opType = FunctionType::get(
      positRawType, /* Return type */
      { positRawType }, /* Arguments... */
      false /* isVarArg */
  );

  arg1 = builder.CreateExtractValue(arg1, {0});

  Value *ret = builder.CreateCall(InlineAsm::get(
      opType,
      asmInstr,
      "=r,r",
      false /* hasSideEffects */
  ), { arg1 });

  return builder.CreateInsertValue(UndefValue::get(llvmType), ret, {0});
}

Value *PositBuilderRISCV::CreateCmp(CmpInst::Predicate pred, Value *arg1, Value *arg2) {
  assert((pred >= CmpInst::FIRST_ICMP_PREDICATE && pred <= CmpInst::LAST_ICMP_PREDICATE) &&
      "Please provide an integer comparison predicate");

  arg1 = builder.CreateExtractValue(arg1, {0});
  arg2 = builder.CreateExtractValue(arg2, {0});
  return builder.CreateCmp(pred, arg1, arg2);
}

Value *PositBuilderRISCV::CreateFMA(Value *arg1, Value *arg2, Value *arg3) {
  Type *positRawType = IntegerType::get(C, metadata.scalarBitsAmt());
  FunctionType *opType = FunctionType::get(
      positRawType, /* Return type */
      { positRawType, positRawType }, /* Arguments... */
      false /* isVarArg */
  );

  arg1 = builder.CreateExtractValue(arg1, {0});
  arg2 = builder.CreateExtractValue(arg2, {0});
  arg3 = builder.CreateExtractValue(arg3, {0});

  Value *mul = builder.CreateCall(InlineAsm::get(opType,
      ".insn r 0xb, 2, 0x6A, $0, $1, $2; nop",
      "=r,r,r",
      false /* hasSideEffects */
  ), { arg1, arg2 });

  Value *add = builder.CreateCall(InlineAsm::get(opType,
      ".insn r 0xb, 0, 0x6A, $0, $1, $2; nop",
      "=r,r,r",
      false /* hasSideEffects */
  ), { mul, arg3 });

  return builder.CreateInsertValue(UndefValue::get(llvmType), add, {0});
}
