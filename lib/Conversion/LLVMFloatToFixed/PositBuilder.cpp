#include "PositBuilder.h"
#include "PositConstant.h"

using namespace llvm;
using namespace flttofix;

#define DEBUG_TYPE "taffo-conversion"

Value *PositBuilder::getAlloc(unsigned idx, const FixedPointType &target) {
  Function *func = builder.GetInsertBlock()->getParent();
  auto &allocas = pass->positAllocaPool[{ func, target.scalarBitsAmt() }];
  Instruction *first = &(*func->getEntryBlock().getFirstInsertionPt());

  if (idx < allocas.size()) {
    // We have already created this alloca,
    // but here we should make sure that it still the first instruction,
    // because somebody might be trying to insert at the basicblock entry.
    allocas[idx]->moveBefore(first);
  }

  while (allocas.size() <= idx) {
    AllocaInst *alloc = new AllocaInst(target.scalarToLLVMType(C), func->getParent()->getDataLayout().getAllocaAddrSpace(),
        "posit" + std::to_string(target.scalarBitsAmt()) + "Arg" + std::to_string(allocas.size()), first);
    allocas.push_back(alloc);
  }

  return allocas[idx];
}

Value *PositBuilder::CreateConstructor(Value *arg1, const FixedPointType *srcMetadata) {
  if (srcMetadata && srcMetadata->isPosit()) {
    return PositBuilder(pass, builder, *srcMetadata).CreateConv(arg1, llvmType, &metadata);
  }

  const char* mangledName;
  char nameBuf[256];
  Type* srcType = arg1->getType();
  const bool isSrcFixpt = srcMetadata && srcMetadata->isFixedPoint() && srcMetadata->scalarFracBitsAmt() > 0;

  switch (metadata.scalarBitsAmt()) {
  case 32:
    if (srcType->isFloatTy()) {
      mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEC1Ef";
    } else if (srcType->isDoubleTy()) {
      mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEC1Ed";
    } else if (srcType->isFloatingPointTy()) {
      arg1 = builder.CreateFPCast(arg1, Type::getDoubleTy(C));
      mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEC1Ed";
    } else if (srcType->isIntegerTy(64)) {
      if (isSrcFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEElLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEC1El";
      }
    } else if (srcType->isIntegerTy(32)) {
      if (isSrcFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEEiLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEC1Ei";
      }
    } else if (srcType->isIntegerTy(16) && isSrcFixpt) {
      snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEEsLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
      mangledName = nameBuf;
    } else if (srcType->isIntegerTy(8) && isSrcFixpt) {
      snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEEaLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
      mangledName = nameBuf;
    } else if (srcType->isIntegerTy()) {
      assert(!isSrcFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit constructor from this integer size, passing through int64...");
      if (!srcMetadata || srcMetadata->scalarIsSigned())
        arg1 = builder.CreateSExtOrTrunc(arg1, Type::getInt64Ty(C));
      else
        arg1 = builder.CreateZExtOrTrunc(arg1, Type::getInt64Ty(C));
      mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEC1El";
    } else {
      llvm_unreachable("Unimplemented constructor from source type");
    }
    break;
  case 16:
    if (srcType->isFloatTy()) {
      mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEC1Ef";
    } else if (srcType->isDoubleTy()) {
      mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEC1Ed";
    } else if (srcType->isFloatingPointTy()) {
      arg1 = builder.CreateFPCast(arg1, Type::getDoubleTy(C));
      mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEC1Ed";
    } else if (srcType->isIntegerTy(64)) {
      if (isSrcFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEElLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEC1El";
      }
    } else if (srcType->isIntegerTy(32)) {
      if (isSrcFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEEiLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEC1Ei";
      }
    } else if (srcType->isIntegerTy(16) && isSrcFixpt) {
      snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEEsLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
      mangledName = nameBuf;
    } else if (srcType->isIntegerTy(8) && isSrcFixpt) {
      snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEEaLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
      mangledName = nameBuf;
    } else if (srcType->isIntegerTy()) {
      assert(!isSrcFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit constructor from this integer size, passing through int64...");
      if (!srcMetadata || srcMetadata->scalarIsSigned())
        arg1 = builder.CreateSExtOrTrunc(arg1, Type::getInt64Ty(C));
      else
        arg1 = builder.CreateZExtOrTrunc(arg1, Type::getInt64Ty(C));
      mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEC1El";
    } else {
      llvm_unreachable("Unimplemented constructor from source type");
    }
    break;
  case 8:
    if (srcType->isFloatTy()) {
      mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEC1Ef";
    } else if (srcType->isDoubleTy()) {
      mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEC1Ed";
    } else if (srcType->isFloatingPointTy()) {
      arg1 = builder.CreateFPCast(arg1, Type::getDoubleTy(C));
      mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEC1Ed";
    } else if (srcType->isIntegerTy(64)) {
      if (isSrcFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEElLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEC1El";
      }
    } else if (srcType->isIntegerTy(32)) {
      if (isSrcFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEEiLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEC1Ei";
      }
    } else if (srcType->isIntegerTy(16) && isSrcFixpt) {
      snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEEsLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
      mangledName = nameBuf;
    } else if (srcType->isIntegerTy(8) && isSrcFixpt) {
      snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit10from_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEEaLi%dEEEvPT_T0_", srcMetadata->scalarFracBitsAmt());
      mangledName = nameBuf;
    } else if (srcType->isIntegerTy()) {
      assert(!isSrcFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit constructor from this integer size, passing through int64...");
      if (!srcMetadata || srcMetadata->scalarIsSigned())
        arg1 = builder.CreateSExtOrTrunc(arg1, Type::getInt64Ty(C));
      else
        arg1 = builder.CreateZExtOrTrunc(arg1, Type::getInt64Ty(C));
      mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEC1El";
    } else {
      llvm_unreachable("Unimplemented constructor from source type");
    }
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  FunctionType *fnType = FunctionType::get(
      Type::getVoidTy(C), /* Return type */
      { llvmType->getPointerTo(), arg1->getType() }, /* Arguments... */
      false /* isVarArg */
  );
  FunctionCallee ctorFun = M->getOrInsertFunction(mangledName, fnType);
  Value *dst = getAlloc(0);

  builder.CreateCall(ctorFun, {dst, arg1});
  return builder.CreateLoad(llvmType, dst);
}

Value *PositBuilder::CreateBinOp(int opcode, Value *arg1, Value *arg2) {
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

  const char* mangledName;
  switch (metadata.scalarBitsAmt()) {
  case 32:
    switch (opcode) {
    case Instruction::FAdd:
      mangledName = "_ZN5positplIiLi32ELi2EjLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FSub:
      mangledName = "_ZN5positmiIiLi32ELi2EjLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FMul:
      mangledName = "_ZN5positmlIiLi32ELi2EjLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FDiv:
      mangledName = "_ZN5positdvIiLi32ELi2EjLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit binary operation");
    }
    break;
  case 16:
    switch (opcode) {
    case Instruction::FAdd:
      mangledName = "_ZN5positplIsLi16ELi2EtLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FSub:
      mangledName = "_ZN5positmiIsLi16ELi2EtLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FMul:
      mangledName = "_ZN5positmlIsLi16ELi2EtLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FDiv:
      mangledName = "_ZN5positdvIsLi16ELi2EtLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit binary operation");
    }
    break;
  case 8:
    switch (opcode) {
    case Instruction::FAdd:
      mangledName = "_ZN5positplIaLi8ELi2EhLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FSub:
      mangledName = "_ZN5positmiIaLi8ELi2EhLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FMul:
      mangledName = "_ZN5positmlIaLi8ELi2EhLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    case Instruction::FDiv:
      mangledName = "_ZN5positdvIaLi8ELi2EhLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit binary operation");
    }
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  FunctionType *fnType = FunctionType::get(
    llvmType, /* Return type */
    { llvmType->getPointerTo(), llvmType->getPointerTo() }, /* Arguments... */
    false /* isVarArg */
  );
  FunctionCallee fun = M->getOrInsertFunction(mangledName, fnType);
  Value *src1 = getAlloc(0);
  Value *src2 = getAlloc(1);
  builder.CreateStore(arg1, src1);
  builder.CreateStore(arg2, src2);
  return builder.CreateCall(fun, {src1, src2});
}

Value *PositBuilder::CreateUnaryOp(int opcode, Value *arg1) {
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

  const char* mangledName;
  switch (metadata.scalarBitsAmt()) {
  case 32:
    switch (opcode) {
    case Instruction::FNeg:
      mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEngEv";
      break;
    default:
      llvm_unreachable("Unimplemented Posit unary operation");
    }
    break;
  case 16:
    switch (opcode) {
    case Instruction::FNeg:
      mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEngEv";
      break;
    default:
      llvm_unreachable("Unimplemented Posit unary operation");
    }
    break;
  case 8:
    switch (opcode) {
    case Instruction::FNeg:
      mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEngEv";
      break;
    default:
      llvm_unreachable("Unimplemented Posit unary operation");
    }
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  FunctionType *fnType = FunctionType::get(
    llvmType, /* Return type */
    { llvmType->getPointerTo() }, /* Arguments... */
    false /* isVarArg */
  );
  FunctionCallee fun = M->getOrInsertFunction(mangledName, fnType);
  Value *src1 = getAlloc(0);
  builder.CreateStore(arg1, src1);
  return builder.CreateCall(fun, {src1});
}

Value *PositBuilder::CreateCmp(CmpInst::Predicate pred, Value *arg1, Value *arg2) {
  assert((pred >= CmpInst::FIRST_ICMP_PREDICATE && pred <= CmpInst::LAST_ICMP_PREDICATE) &&
      "Please provide an integer comparison predicate");

  const char* mangledName;
  switch (metadata.scalarBitsAmt()) {
  case 32:
    switch (pred) {
    case CmpInst::ICMP_EQ:
      mangledName = "_ZN5positeqIiLi32ELi2EjLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_NE:
      mangledName = "_ZN5positneIiLi32ELi2EjLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SGT:
      mangledName = "_ZN5positgtIiLi32ELi2EjLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SGE:
      mangledName = "_ZN5positgeIiLi32ELi2EjLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SLT:
      mangledName = "_ZN5positltIiLi32ELi2EjLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SLE:
      mangledName = "_ZN5positleIiLi32ELi2EjLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit comparison operation");
    }
    break;
  case 16:
    switch (pred) {
    case CmpInst::ICMP_EQ:
      mangledName = "_ZN5positeqIsLi16ELi2EtLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_NE:
      mangledName = "_ZN5positneIsLi16ELi2EtLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SGT:
      mangledName = "_ZN5positgtIsLi16ELi2EtLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SGE:
      mangledName = "_ZN5positgeIsLi16ELi2EtLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SLT:
      mangledName = "_ZN5positltIsLi16ELi2EtLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SLE:
      mangledName = "_ZN5positleIsLi16ELi2EtLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit comparison operation");
    }
    break;
  case 8:
    switch (pred) {
    case CmpInst::ICMP_EQ:
      mangledName = "_ZN5positeqIaLi8ELi2EhLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_NE:
      mangledName = "_ZN5positneIaLi8ELi2EhLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SGT:
      mangledName = "_ZN5positgtIaLi8ELi2EhLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SGE:
      mangledName = "_ZN5positgeIaLi8ELi2EhLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SLT:
      mangledName = "_ZN5positltIaLi8ELi2EhLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    case CmpInst::ICMP_SLE:
      mangledName = "_ZN5positleIaLi8ELi2EhLNS_9PositSpecE1EEEbRKNS_5PositIT_XT0_EXT1_ET2_XT3_EEES7_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit comparison operation");
    }
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  FunctionType *fnType = FunctionType::get(
    Type::getInt1Ty(C), /* Return type */
    { llvmType->getPointerTo(), llvmType->getPointerTo() }, /* Arguments... */
    false /* isVarArg */
  );

  FunctionCallee fun = M->getOrInsertFunction(mangledName, fnType);
  Value *src1 = getAlloc(0);
  Value *src2 = getAlloc(1);
  builder.CreateStore(arg1, src1);
  builder.CreateStore(arg2, src2);
  return builder.CreateCall(fun, {src1, src2});
}

Value *PositBuilder::CreateConv(Value *from, Type *dstType, const FixedPointType *dstMetadata) {
  if (Constant *c = dyn_cast<Constant>(from)) {
    LLVM_DEBUG(dbgs() << "Attempting to fold constant Posit conversion\n");
    Constant *res = PositConstant::FoldConv(C, &M->getDataLayout(), metadata, c, dstType);
    if (res) {
      LLVM_DEBUG(dbgs() << "Folded in " << *res << "\n");
      return res;
    } else {
      LLVM_DEBUG(dbgs() << "Constant folding failed; falling back to runtime computation\n");
    }
  }

  const char* mangledName;
  char nameBuf[256];
  Type* callDstType = dstType;
  const bool isDstFixpt = dstMetadata && dstMetadata->isFixedPoint() && dstMetadata->scalarFracBitsAmt() > 0;
  const bool isDstPosit = dstMetadata && dstMetadata->isPosit();

  switch (metadata.scalarBitsAmt()) {
  case 32:
    if (isDstPosit) {
      switch (dstMetadata->scalarBitsAmt()) {
      case 16:
        mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EE8to_positINS0_IsLi16ELi2EtLS1_1EEEEET_v";
        break;
      case 8:
        mangledName = "_ZN5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EE8to_positINS0_IaLi8ELi2EhLS1_1EEEEET_v";
        break;
      default:
        llvm_unreachable("Unsupported Posit to Posit size");
      }
    } else if (dstType->isFloatTy()) {
      mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcvfEv";
    } else if (dstType->isDoubleTy()) {
      mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcvdEv";
    } else if (dstType->isFloatingPointTy()) {
      mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcvdEv";
      callDstType = Type::getDoubleTy(C);
    } else if (dstType->isIntegerTy(64)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEElLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcvlEv";
      }
    } else if (dstType->isIntegerTy(32)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEEiLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcviEv";
      }
    } else if (dstType->isIntegerTy(16)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEEsLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcvsEv";
      }
    } else if (dstType->isIntegerTy(8)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIiLi32ELi2EjLNS_9PositSpecE1EEEaLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcvaEv";
      }
    } else if (dstType->isIntegerTy()) {
      assert(!isDstFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit conversion to this integer size, passing through int64...");
      mangledName = "_ZNK5posit5PositIiLi32ELi2EjLNS_9PositSpecE1EEcvlEv";
      callDstType = Type::getInt64Ty(C);
    } else {
      llvm_unreachable("Unimplemented conversion from Posit to other numeric type");
    }
    break;
  case 16:
    if (isDstPosit) {
      switch (dstMetadata->scalarBitsAmt()) {
      case 32:
        mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EE8to_positINS0_IiLi32ELi2EjLS1_1EEEEET_v";
        break;
      case 8:
        mangledName = "_ZN5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EE8to_positINS0_IaLi8ELi2EhLS1_1EEEEET_v";
        break;
      default:
        llvm_unreachable("Unsupported Posit to Posit size");
      }
    } else if (dstType->isFloatTy()) {
      mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcvfEv";
    } else if (dstType->isDoubleTy()) {
      mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcvdEv";
    } else if (dstType->isFloatingPointTy()) {
      mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcvdEv";
      callDstType = Type::getDoubleTy(C);
    } else if (dstType->isIntegerTy(64)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEElLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcvlEv";
      }
    } else if (dstType->isIntegerTy(32)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEEiLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcviEv";
      }
    } else if (dstType->isIntegerTy(16)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEEsLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcvsEv";
      }
    } else if (dstType->isIntegerTy(8)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIsLi16ELi2EtLNS_9PositSpecE1EEEaLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcvaEv";
      }
    } else if (dstType->isIntegerTy()) {
      assert(!isDstFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit conversion to this integer size, passing through int64...");
      mangledName = "_ZNK5posit5PositIsLi16ELi2EtLNS_9PositSpecE1EEcvlEv";
      callDstType = Type::getInt64Ty(C);
    } else {
      llvm_unreachable("Unimplemented conversion from Posit to other numeric type");
    }
    break;
  case 8:
    if (isDstPosit) {
      switch (dstMetadata->scalarBitsAmt()) {
      case 32:
        mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EE8to_positINS0_IiLi32ELi2EjLS1_1EEEEET_v";
        break;
      case 16:
        mangledName = "_ZN5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EE8to_positINS0_IsLi16ELi2EtLS1_1EEEEET_v";
        break;
      default:
        llvm_unreachable("Unsupported Posit to Posit size");
      }
    } else if (dstType->isFloatTy()) {
      mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEcvfEv";
    } else if (dstType->isDoubleTy()) {
      mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEcvdEv";
    } else if (dstType->isFloatingPointTy()) {
      mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEcvdEv";
      callDstType = Type::getDoubleTy(C);
    } else if (dstType->isIntegerTy(64)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEElLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEcvlEv";
      }
    } else if (dstType->isIntegerTy(32)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEEiLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEcviEv";
      }
    } else if (dstType->isIntegerTy(16)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEEsLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEcvsEv";
      }
    } else if (dstType->isIntegerTy(8)) {
      if (isDstFixpt) {
        snprintf(nameBuf, sizeof(nameBuf), "_ZN5posit8to_fixedINS_5PositIaLi8ELi2EhLNS_9PositSpecE1EEEaLi%dEEET0_PT_", dstMetadata->scalarFracBitsAmt());
        mangledName = nameBuf;
      } else {
        mangledName = "_ZNK5posit5PositIaLi8ELi2EhLNS_9PositSpecE1EEcvaEv";
      }
    } else if (dstType->isIntegerTy()) {
      assert(!isDstFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit conversion to this integer size, passing through int64...");
      mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcvlEv";
      callDstType = Type::getInt64Ty(C);
    } else {
      llvm_unreachable("Unimplemented conversion from Posit to other numeric type");
    }
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  FunctionType *fnType = FunctionType::get(
    callDstType, /* Return type */
    { llvmType->getPointerTo() }, /* Arguments... */
    false /* isVarArg */
  );
  FunctionCallee convFun = M->getOrInsertFunction(mangledName, fnType);
  Value *src1 = getAlloc(0);
  builder.CreateStore(from, src1);

  Value *ret = builder.CreateCall(convFun, { src1 });

  if (dstType->isFloatingPointTy() && dstType != callDstType) {
    ret = builder.CreateFPCast(ret, dstType);
  }

  if (dstType->isIntegerTy() && dstType != callDstType) {
    assert(!isDstFixpt);
    ret = builder.CreateSExtOrTrunc(ret, dstType);
  }

  return ret;
}

Value *PositBuilder::CreateFMA(Value *arg1, Value *arg2, Value *arg3) {
  const char* mangledName;
  switch (metadata.scalarBitsAmt()) {
  case 32:
    mangledName = "_ZN5posit3fmaIiLi32ELi2EjLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_S7_";
    break;
  case 16:
    mangledName = "_ZN5posit3fmaIsLi16ELi2EtLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_S7_";
    break;
  case 8:
    mangledName = "_ZN5posit3fmaIaLi8ELi2EhLNS_9PositSpecE1EEENS_5PositIT_XT0_EXT1_ET2_XT3_EEERKS5_S7_S7_";
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  FunctionType *fnType = FunctionType::get(
    llvmType, /* Return type */
    { llvmType->getPointerTo(), llvmType->getPointerTo(), llvmType->getPointerTo() }, /* Arguments... */
    false /* isVarArg */
  );

  FunctionCallee fun = M->getOrInsertFunction(mangledName, fnType);
  Value *src1 = getAlloc(0);
  Value *src2 = getAlloc(1);
  Value *src3 = getAlloc(2);
  builder.CreateStore(arg1, src1);
  builder.CreateStore(arg2, src2);
  builder.CreateStore(arg3, src3);
  return builder.CreateCall(fun, {src1, src2, src3});
}
