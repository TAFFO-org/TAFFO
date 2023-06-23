#include "PositBuilder.h"
#include "PositConstant.h"

using namespace llvm;
using namespace flttofix;

#define DEBUG_TYPE "taffo-conversion"

Value *PositBuilder::CreateConstructor(Value *arg1, bool isSigned) {
  const char* mangledName;
  Type* srcType = arg1->getType();

  switch (metadata.scalarBitsAmt()) {
  case 32:
    if (srcType->isFloatTy()) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ef";
    } else if (srcType->isDoubleTy()) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ed";
    } else if (srcType->isFloatingPointTy()) {
      arg1 = builder.CreateFPCast(arg1, Type::getDoubleTy(C));
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ed";
    } else if (srcType->isIntegerTy(64)) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1El";
    } else if (srcType->isIntegerTy(32)) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ei";
    } else if (srcType->isIntegerTy(16)) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Es";
    } else if (srcType->isIntegerTy(8)) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ea";
    } else if (srcType->isIntegerTy()) {
      LLVM_DEBUG(dbgs() << "Unimplemented Posit constructor from this integer size, passing through int64...");
      if (isSigned)
        arg1 = builder.CreateSExtOrTrunc(arg1, Type::getInt64Ty(C));
      else
        arg1 = builder.CreateZExtOrTrunc(arg1, Type::getInt64Ty(C));
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1El";
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
  Value *dst = builder.CreateAlloca(llvmType);
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
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEplERKS1_";
      break;
    case Instruction::FSub:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEmiERKS1_";
      break;
    case Instruction::FMul:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEmlERKS1_";
      break;
    case Instruction::FDiv:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEdvERKS1_";
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
  Value *src1 = builder.CreateAlloca(llvmType);
  Value *src2 = builder.CreateAlloca(llvmType);
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
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEngEv";
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
  Value *src1 = builder.CreateAlloca(llvmType);
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
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEeqERKS1_";
      break;
    case CmpInst::ICMP_NE:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEneERKS1_";
      break;
    case CmpInst::ICMP_SGT:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEgtERKS1_";
      break;
    case CmpInst::ICMP_SGE:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEgeERKS1_";
      break;
    case CmpInst::ICMP_SLT:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEltERKS1_";
      break;
    case CmpInst::ICMP_SLE:
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEleERKS1_";
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
  Value *src1 = builder.CreateAlloca(llvmType);
  Value *src2 = builder.CreateAlloca(llvmType);
  builder.CreateStore(arg1, src1);
  builder.CreateStore(arg2, src2);
  return builder.CreateCall(fun, {src1, src2});
}

Value *PositBuilder::CreateConv(Value *from, Type *dstType) {
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
  Type* callDstType = dstType;

  // TODO implement casting to bigger or smaller Posit
  switch (metadata.scalarBitsAmt()) {
  case 32:
    if (dstType->isFloatTy()) {
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvfEv";
    } else if (dstType->isDoubleTy()) {
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvdEv";
    } else if (dstType->isFloatingPointTy()) {
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvdEv";
      callDstType = Type::getDoubleTy(C);
    } else if (dstType->isIntegerTy(64)) {
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvlEv";
    } else if (dstType->isIntegerTy(32)) {
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcviEv";
    } else if (dstType->isIntegerTy(16)) {
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvsEv";
    } else if (dstType->isIntegerTy(8)) {
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvaEv";
    } else if (dstType->isIntegerTy()) {
      LLVM_DEBUG(dbgs() << "Unimplemented Posit conversion to this integer size, passing through int64...");
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvlEv";
      callDstType = Type::getInt64Ty(C);
    } else {
      llvm_unreachable("Unimplemented conversion from Posit32 to other numeric type");
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
  Value *src1 = builder.CreateAlloca(llvmType);
  builder.CreateStore(from, src1);
  Value *ret = builder.CreateCall(convFun, { src1 });

  if (dstType->isFloatingPointTy() && dstType != callDstType) {
    ret = builder.CreateFPTrunc(ret, dstType);
  }

  if (dstType->isIntegerTy() && dstType != callDstType) {
    ret = builder.CreateSExtOrTrunc(ret, dstType);
  }

  return ret;
}

Value *PositBuilder::CreateFMA(Value *arg1, Value *arg2, Value *arg3) {
  const char* mangledName;
  switch (metadata.scalarBitsAmt()) {
  case 32:
    mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EE3fmaERKS1_S3_";
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
  Value *src1 = builder.CreateAlloca(llvmType);
  Value *src2 = builder.CreateAlloca(llvmType);
  Value *src3 = builder.CreateAlloca(llvmType);
  builder.CreateStore(arg1, src1);
  builder.CreateStore(arg2, src2);
  builder.CreateStore(arg3, src3);
  return builder.CreateCall(fun, {src1, src2, src3});
}
