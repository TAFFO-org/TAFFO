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
  const char* mangledName;
  Type* srcType = arg1->getType();
  const bool isSrcFixpt = srcMetadata && srcMetadata->isFixedPoint() && srcMetadata->scalarFracBitsAmt() > 0;
  const bool isSrcPosit = srcMetadata && srcMetadata->isPosit();

  switch (metadata.scalarBitsAmt()) {
  case 32:
    if (isSrcPosit) {
      switch (srcMetadata->scalarBitsAmt()) {
      case 16:
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1IsLi16ELi2EtLS0_1EEERKS_IT_XT0_EXT1_ET2_XT3_EE";
        break;
      case 8:
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1IaLi8ELi2EhLS0_1EEERKS_IT_XT0_EXT1_ET2_XT3_EE";
        break;
      default:
        llvm_unreachable("Unsupported Posit to Posit size");
      }
    } else if (srcType->isFloatTy()) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ef";
    } else if (srcType->isDoubleTy()) {
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ed";
    } else if (srcType->isFloatingPointTy()) {
      arg1 = builder.CreateFPCast(arg1, Type::getDoubleTy(C));
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ed";
    } else if (srcType->isIntegerTy(64)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Eli";
      else
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1El";
    } else if (srcType->isIntegerTy(32)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Eii";
      else
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ei";
    } else if (srcType->isIntegerTy(16)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Esi";
      else
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Es";
    } else if (srcType->isIntegerTy(8)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Eai";
      else
        mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1Ea";
    } else if (srcType->isIntegerTy()) {
      assert(!isSrcFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit constructor from this integer size, passing through int64...");
      if (!srcMetadata || srcMetadata->scalarIsSigned())
        arg1 = builder.CreateSExtOrTrunc(arg1, Type::getInt64Ty(C));
      else
        arg1 = builder.CreateZExtOrTrunc(arg1, Type::getInt64Ty(C));
      mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EEC1El";
    } else {
      llvm_unreachable("Unimplemented constructor from source type");
    }
    break;
  case 16:
    if (isSrcPosit) {
      switch (srcMetadata->scalarBitsAmt()) {
      case 32:
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1IiLi32ELi2EjLS0_1EEERKS_IT_XT0_EXT1_ET2_XT3_EE";
        break;
      case 8:
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1IaLi8ELi2EhLS0_1EEERKS_IT_XT0_EXT1_ET2_XT3_EE";
        break;
      default:
        llvm_unreachable("Unsupported Posit to Posit size");
      }
    } else if (srcType->isFloatTy()) {
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Ef";
    } else if (srcType->isDoubleTy()) {
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Ed";
    } else if (srcType->isFloatingPointTy()) {
      arg1 = builder.CreateFPCast(arg1, Type::getDoubleTy(C));
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Ed";
    } else if (srcType->isIntegerTy(64)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Eli";
      else
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1El";
    } else if (srcType->isIntegerTy(32)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Eii";
      else
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Ei";
    } else if (srcType->isIntegerTy(16)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Esi";
      else
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Es";
    } else if (srcType->isIntegerTy(8)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Eai";
      else
        mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1Ea";
    } else if (srcType->isIntegerTy()) {
      assert(!isSrcFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit constructor from this integer size, passing through int64...");
      if (!srcMetadata || srcMetadata->scalarIsSigned())
        arg1 = builder.CreateSExtOrTrunc(arg1, Type::getInt64Ty(C));
      else
        arg1 = builder.CreateZExtOrTrunc(arg1, Type::getInt64Ty(C));
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEC1El";
    } else {
      llvm_unreachable("Unimplemented constructor from source type");
    }
    break;
  case 8:
    if (isSrcPosit) {
      switch (srcMetadata->scalarBitsAmt()) {
      case 32:
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1IiLi32ELi2EjLS0_1EEERKS_IT_XT0_EXT1_ET2_XT3_EE";
        break;
      case 16:
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1IsLi16ELi2EtLS0_1EEERKS_IT_XT0_EXT1_ET2_XT3_EE";
        break;
      default:
        llvm_unreachable("Unsupported Posit to Posit size");
      }
    } else if (srcType->isFloatTy()) {
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Ef";
    } else if (srcType->isDoubleTy()) {
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Ed";
    } else if (srcType->isFloatingPointTy()) {
      arg1 = builder.CreateFPCast(arg1, Type::getDoubleTy(C));
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Ed";
    } else if (srcType->isIntegerTy(64)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Eli";
      else
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1El";
    } else if (srcType->isIntegerTy(32)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Eii";
      else
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Ei";
    } else if (srcType->isIntegerTy(16)) {
      if (isSrcFixpt)
        mangledName = "__ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Esi";
      else
        mangledName = "__ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Es";
    } else if (srcType->isIntegerTy(8)) {
      if (isSrcFixpt)
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Eai";
      else
        mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1Ea";
    } else if (srcType->isIntegerTy()) {
      assert(!isSrcFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit constructor from this integer size, passing through int64...");
      if (!srcMetadata || srcMetadata->scalarIsSigned())
        arg1 = builder.CreateSExtOrTrunc(arg1, Type::getInt64Ty(C));
      else
        arg1 = builder.CreateZExtOrTrunc(arg1, Type::getInt64Ty(C));
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEC1El";
    } else {
      llvm_unreachable("Unimplemented constructor from source type");
    }
    break;
  default:
    llvm_unreachable("Unimplemented Posit size");
  }

  if (isSrcPosit) {
    Value *otherPosit = getAlloc(0, *srcMetadata);
    builder.CreateStore(arg1, otherPosit);
    arg1 = otherPosit;
  }

  std::vector<Type*> argTypes = { llvmType->getPointerTo(), arg1->getType() };
  if (isSrcFixpt)
    argTypes.push_back(Type::getInt32Ty(C));

  FunctionType *fnType = FunctionType::get(
      Type::getVoidTy(C), /* Return type */
      argTypes, /* Arguments... */
      false /* isVarArg */
  );
  FunctionCallee ctorFun = M->getOrInsertFunction(mangledName, fnType);
  Value *dst = getAlloc(0);

  std::vector<Value*> args = {dst, arg1};
  if (isSrcFixpt)
    args.push_back(ConstantInt::get(Type::getInt32Ty(C), srcMetadata->scalarFracBitsAmt()));

  builder.CreateCall(ctorFun, args);
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
  case 16:
    switch (opcode) {
    case Instruction::FAdd:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEplERKS1_";
      break;
    case Instruction::FSub:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEmiERKS1_";
      break;
    case Instruction::FMul:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEmlERKS1_";
      break;
    case Instruction::FDiv:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEdvERKS1_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit binary operation");
    }
    break;
  case 8:
    switch (opcode) {
    case Instruction::FAdd:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEplERKS1_";
      break;
    case Instruction::FSub:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEmiERKS1_";
      break;
    case Instruction::FMul:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEmlERKS1_";
      break;
    case Instruction::FDiv:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEdvERKS1_";
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
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEngEv";
      break;
    default:
      llvm_unreachable("Unimplemented Posit unary operation");
    }
    break;
  case 16:
    switch (opcode) {
    case Instruction::FNeg:
      mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEngEv";
      break;
    default:
      llvm_unreachable("Unimplemented Posit unary operation");
    }
    break;
  case 8:
    switch (opcode) {
    case Instruction::FNeg:
      mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEngEv";
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
  case 16:
    switch (pred) {
    case CmpInst::ICMP_EQ:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEeqERKS1_";
      break;
    case CmpInst::ICMP_NE:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEneERKS1_";
      break;
    case CmpInst::ICMP_SGT:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEgtERKS1_";
      break;
    case CmpInst::ICMP_SGE:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEgeERKS1_";
      break;
    case CmpInst::ICMP_SLT:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEltERKS1_";
      break;
    case CmpInst::ICMP_SLE:
      mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EEleERKS1_";
      break;
    default:
      llvm_unreachable("Unimplemented Posit comparison operation");
    }
    break;
  case 8:
    switch (pred) {
    case CmpInst::ICMP_EQ:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEeqERKS1_";
      break;
    case CmpInst::ICMP_NE:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEneERKS1_";
      break;
    case CmpInst::ICMP_SGT:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEgtERKS1_";
      break;
    case CmpInst::ICMP_SGE:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEgeERKS1_";
      break;
    case CmpInst::ICMP_SLT:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEltERKS1_";
      break;
    case CmpInst::ICMP_SLE:
      mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EEleERKS1_";
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
  if (dstMetadata && dstMetadata->isPosit())
    return PositBuilder(pass, builder, *dstMetadata).CreateConstructor(from, &metadata);

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
  const bool isDstFixpt = dstMetadata && dstMetadata->isFixedPoint() && dstMetadata->scalarFracBitsAmt() > 0;

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
      if (isDstFixpt)
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EE9toFixed64Ei";
      else
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvlEv";
    } else if (dstType->isIntegerTy(32)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EE9toFixed32Ei";
      else
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcviEv";
    } else if (dstType->isIntegerTy(16)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EE9toFixed16Ei";
      else
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvsEv";
    } else if (dstType->isIntegerTy(8)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EE8toFixed8Ei";
      else
        mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvaEv";
    } else if (dstType->isIntegerTy()) {
      assert(!isDstFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit conversion to this integer size, passing through int64...");
      mangledName = "_ZNK5PositIiLi32ELi2EjL9PositSpec1EEcvlEv";
      callDstType = Type::getInt64Ty(C);
    } else {
      llvm_unreachable("Unimplemented conversion from Posit to other numeric type");
    }
    break;
  case 16:
    if (dstType->isFloatTy()) {
      mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcvfEv";
    } else if (dstType->isDoubleTy()) {
      mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcvdEv";
    } else if (dstType->isFloatingPointTy()) {
      mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcvdEv";
      callDstType = Type::getDoubleTy(C);
    } else if (dstType->isIntegerTy(64)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EE9toFixed64Ei";
      else
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcvlEv";
    } else if (dstType->isIntegerTy(32)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EE9toFixed32Ei";
      else
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcviEv";
    } else if (dstType->isIntegerTy(16)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EE9toFixed16Ei";
      else
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcvsEv";
    } else if (dstType->isIntegerTy(8)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EE8toFixed8Ei";
      else
        mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcvaEv";
    } else if (dstType->isIntegerTy()) {
      assert(!isDstFixpt && "Unsupported fixed point size");
      LLVM_DEBUG(dbgs() << "Unimplemented Posit conversion to this integer size, passing through int64...");
      mangledName = "_ZNK5PositIsLi16ELi2EtL9PositSpec1EEcvlEv";
      callDstType = Type::getInt64Ty(C);
    } else {
      llvm_unreachable("Unimplemented conversion from Posit to other numeric type");
    }
    break;
  case 8:
    if (dstType->isFloatTy()) {
      mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcvfEv";
    } else if (dstType->isDoubleTy()) {
      mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcvdEv";
    } else if (dstType->isFloatingPointTy()) {
      mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcvdEv";
      callDstType = Type::getDoubleTy(C);
    } else if (dstType->isIntegerTy(64)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EE9toFixed64Ei";
      else
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcvlEv";
    } else if (dstType->isIntegerTy(32)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EE9toFixed32Ei";
      else
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcviEv";
    } else if (dstType->isIntegerTy(16)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EE9toFixed16Ei";
      else
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcvsEv";
    } else if (dstType->isIntegerTy(8)) {
      if (isDstFixpt)
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EE8toFixed8Ei";
      else
        mangledName = "_ZNK5PositIaLi8ELi2EhL9PositSpec1EEcvaEv";
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

  std::vector<Type*> argTypes = { llvmType->getPointerTo() };
  if (isDstFixpt)
    argTypes.push_back(Type::getInt32Ty(C));

  FunctionType *fnType = FunctionType::get(
    callDstType, /* Return type */
    argTypes, /* Arguments... */
    false /* isVarArg */
  );
  FunctionCallee convFun = M->getOrInsertFunction(mangledName, fnType);
  Value *src1 = getAlloc(0);
  builder.CreateStore(from, src1);

  std::vector<Value*> args = { src1 };
  if (isDstFixpt)
    args.push_back(ConstantInt::get(Type::getInt32Ty(C), dstMetadata->scalarFracBitsAmt()));

  Value *ret = builder.CreateCall(convFun, args);

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
    mangledName = "_ZN5PositIiLi32ELi2EjL9PositSpec1EE3fmaERKS1_S3_";
    break;
  case 16:
    mangledName = "_ZN5PositIsLi16ELi2EtL9PositSpec1EE3fmaERKS1_S3_";
    break;
  case 8:
    mangledName = "_ZN5PositIaLi8ELi2EhL9PositSpec1EE3fmaERKS1_S3_";
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
