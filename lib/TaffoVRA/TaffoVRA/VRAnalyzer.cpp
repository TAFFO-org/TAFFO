#include "MemSSAUtils.hpp"
#include "RangeOperations.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"
#include "VRAnalyzer.hpp"

#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Debug.h>
#include <llvm/IR/Operator.h>

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/IVDescriptors.h>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-vra"

void VRAnalyzer::convexMerge(const AnalysisStore& other) {
  // Since dyn_cast<T>() does not do cross-casting, we must do this:
  if (isa<VRAnalyzer>(other))
    VRAStore::convexMerge(cast<VRAStore>(cast<VRAnalyzer>(other)));
  else if (isa<VRAGlobalStore>(other))
    VRAStore::convexMerge(cast<VRAStore>(cast<VRAGlobalStore>(other)));
  else
    VRAStore::convexMerge(cast<VRAStore>(cast<VRAFunctionStore>(other)));
}

std::shared_ptr<CodeAnalyzer> VRAnalyzer::newCodeAnalyzer(CodeInterpreter& CI) {
  return std::make_shared<VRAnalyzer>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()), &CI);
}

std::shared_ptr<AnalysisStore> VRAnalyzer::newFunctionStore(CodeInterpreter& CI) {
  return std::make_shared<VRAFunctionStore>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()));
}

std::shared_ptr<CodeAnalyzer> VRAnalyzer::newInstructionAnalyzer(ModuleInterpreter& MI) {
  return std::make_shared<VRAnalyzer>(std::static_ptr_cast<VRALogger>(MI.getGlobalStore()->getLogger()), &MI);
}

std::shared_ptr<AnalysisStore> VRAnalyzer::newFnStore(ModuleInterpreter& MI) {
  return std::make_shared<VRAFunctionStore>(std::static_ptr_cast<VRALogger>(MI.getGlobalStore()->getLogger()));
}

std::shared_ptr<CodeAnalyzer> VRAnalyzer::clone() { return std::make_shared<VRAnalyzer>(*this); }

static const Value* stripCasts(const Value *V) {
    const Value *Cur = V;
    while (Cur) {
        if (auto *Cast = dyn_cast<CastInst>(Cur)) {
            Cur = Cast->getOperand(0);
            continue;
        }
        if (auto *CE = dyn_cast<ConstantExpr>(Cur)) {
            if (CE->isCast()) {
                Cur = CE->getOperand(0);
                continue;
            }
        }
        if (auto *Op = dyn_cast<Operator>(Cur)) {
            unsigned opc = Op->getOpcode();
            if (opc == Instruction::BitCast || opc == Instruction::AddrSpaceCast) {
                Cur = Op->getOperand(0);
                continue;
            }
        }
        break;
    }
    return Cur;
};

// return for load and stores its array dimension
static int getArrayAccessDimFromPtr(const Value *Ptr) {
  Ptr = stripCasts(Ptr);

  int dims = 0;
  const Value *Cur = Ptr;

  while (Cur) {
    Cur = stripCasts(Cur);

    const GEPOperator *GEP = dyn_cast<GEPOperator>(Cur);
    if (!GEP) break;

    Type *Ty = GEP->getSourceElementType();
    unsigned pos = 0;

    for (auto It = GEP->idx_begin(); It != GEP->idx_end(); ++It, ++pos) {
      if (pos == 0)
        continue;

      if (auto *Arr = dyn_cast<ArrayType>(Ty)) {
        ++dims;
        Ty = Arr->getElementType();
        continue;
      }

      if (auto *ST = dyn_cast<StructType>(Ty)) {
        if (auto *CI = dyn_cast<ConstantInt>(It->get())) {
          unsigned field = CI->getZExtValue();
          if (field < ST->getNumElements()) {
            Ty = ST->getElementType(field);
            continue;
          }
        }
        return -1;
      }
    }

    Cur = GEP->getPointerOperand();
  }

  return dims;
}

static bool isSameMemoryLoad(const Value* V, const Value* LoadJunction, const Value* StorePtr) {
  V = stripCasts(V);
  LoadJunction = stripCasts(LoadJunction);

  auto *LI = dyn_cast<LoadInst>(V);
  auto *LJ = dyn_cast<LoadInst>(LoadJunction);
  if (!LI || !LJ) return false;

  // base object check (quello che già fai tu)
  return getBaseMemoryObject(LI->getPointerOperand()) ==
         getBaseMemoryObject(StorePtr);
}

static bool isSameValueOrCasted(const llvm::Value* V, const llvm::Value* Ref) {
  return stripCasts(V) == stripCasts(Ref);
}

std::shared_ptr<taffo::Range> VRAnalyzer::extractDeltaFromPhiValue(const llvm::PHINode* Phi, const llvm::BasicBlock* Preheader, const llvm::BasicBlock* Latch) {
  if (!Phi || !Preheader || !Latch)
    return nullptr;

  const llvm::Value* Init = Phi->getIncomingValueForBlock(Preheader); (void)Init;
  const llvm::Value* Next = Phi->getIncomingValueForBlock(Latch);
  if (!Next)
    return nullptr;

  Next = stripCasts(Next);

  // ---- Case 1: next = phi + inc  (float/int)
  if (auto *BO = llvm::dyn_cast<llvm::BinaryOperator>(Next)) {
    auto Opc = BO->getOpcode();

    // ADD / FADD
    if (Opc == llvm::Instruction::Add || Opc == llvm::Instruction::FAdd) {
      const llvm::Value* A = BO->getOperand(0);
      const llvm::Value* B = BO->getOperand(1);

      if (isSameValueOrCasted(A, Phi))
        return getRange(getNode(B));  // delta = inc
      if (isSameValueOrCasted(B, Phi))
        return getRange(getNode(A));  // delta = inc
    }

    // SUB / FSUB: next = phi - inc  => delta = -inc
    if (Opc == llvm::Instruction::Sub || Opc == llvm::Instruction::FSub) {
      const llvm::Value* A = BO->getOperand(0);
      const llvm::Value* B = BO->getOperand(1);

      if (isSameValueOrCasted(A, Phi)) {
        auto inc = getRange(getNode(B));
        return handleUnaryInstruction(inc, Instruction::FNeg);
      }

      // next = inc - phi  NON è "phi + delta" (è coeff -1).
      // Se vuoi trattarlo come affine con coeff=-1 ti serve un'altra classe (non solo delta).
      return nullptr;
    }
  }

  // ---- Case 2: next = llvm.fmuladd(a, b, phi)  => phi + a*b
  if (auto *CB = llvm::dyn_cast<llvm::CallBase>(Next)) {
    if (auto *F = CB->getCalledFunction()) {
      llvm::StringRef Name = F->getName();
#if LLVM_VERSION_MAJOR <= 15
      if (Name.startswith("llvm.fmuladd")) {
#else
      if (Name.starts_with("llvm.fmuladd")) {
#endif
        if (CB->arg_size() == 3) {
          const llvm::Value* a = CB->getArgOperand(0);
          const llvm::Value* b = CB->getArgOperand(1);
          const llvm::Value* c = CB->getArgOperand(2);

          if (isSameValueOrCasted(c, Phi)) {
            auto ra = getRange(getNode(a));
            auto rb = getRange(getNode(b));
            return handleMul(ra, rb); // delta = a*b
          }
        }
      }
    }
  }

  // Non riconosciuto
  return nullptr;
}

std::shared_ptr<Range> VRAnalyzer::extractDeltaFromStoreValue(const Value* StoreVal, const Value* LoadJunction, const StoreInst* Store) {

  StoreVal = stripCasts(StoreVal);
  LoadJunction = stripCasts(LoadJunction);

  // case: new = old + inc
  if (auto *BO = dyn_cast<BinaryOperator>(StoreVal)) {
    if (BO->getOpcode() == Instruction::FAdd || BO->getOpcode() == Instruction::Add) {
      const Value* A = BO->getOperand(0);
      const Value* B = BO->getOperand(1);
      if (isSameMemoryLoad(A, LoadJunction, Store->getPointerOperand()))
        return getRange(getNode(B));     // delta = range(inc)
      if (isSameMemoryLoad(B, LoadJunction, Store->getPointerOperand()))
        return getRange(getNode(A));     // delta = range(inc)
    }

    // case: new = old - inc  => delta = -inc
    if (BO->getOpcode() == Instruction::FSub || BO->getOpcode() == Instruction::Sub) {
      const Value* A = BO->getOperand(0);
      const Value* B = BO->getOperand(1);
      if (isSameMemoryLoad(A, LoadJunction, Store->getPointerOperand())) {
        auto inc = getRange(getNode(B));
        return handleUnaryInstruction(inc, Instruction::FNeg);
      }
    }
  }

  // case: llvm.fmuladd(a,b,c)  and c == old
  if (auto *CB = dyn_cast<CallBase>(StoreVal)) {
    if (auto *F = CB->getCalledFunction()) {
#if LLVM_VERSION_MAJOR <= 15
      if (F->getName() == "llvm.fmuladd.f32" || F->getName().startswith("llvm.fmuladd")) {
#else
      if (F->getName() == "llvm.fmuladd.f32" || F->getName().starts_with("llvm.fmuladd")) {
#endif
        const Value* a = CB->getArgOperand(0);
        const Value* b = CB->getArgOperand(1);
        const Value* c = CB->getArgOperand(2);
        if (isSameMemoryLoad(c, LoadJunction, Store->getPointerOperand())) {
          auto ra = getRange(getNode(a));
          auto rb = getRange(getNode(b));
          return handleMul(ra, rb);  // delta = a*b
        }
      }
    }
  }

  return nullptr;
}

void VRAnalyzer::analyzeInstruction(Instruction* I) {
  assert(I);
  Instruction& i = *I;
  const unsigned OpCode = i.getOpcode();
  if (OpCode == Instruction::Call || OpCode == Instruction::Invoke) {
    handleSpecialCall(&i);
  }
  else if (Instruction::isCast(OpCode) && OpCode != Instruction::BitCast) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const Value* op = i.getOperand(0);
    const std::shared_ptr<Range> oldInfo = fetchRange(&i);
    const std::shared_ptr<Range> info = fetchRange(op);
    const std::shared_ptr<Range> res = handleCastInstruction(info, OpCode, i.getType());
    saveValueRange(&i, res);

    LLVM_DEBUG(
      if (!info)
        Logger->logInfo("operand range is null"));
    LLVM_DEBUG(logRangeln(&i));
  }
  else if (Instruction::isBinaryOp(OpCode)) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const Value* op1 = i.getOperand(0);
    const Value* op2 = i.getOperand(1);
    const std::shared_ptr<Range> info1 = fetchRange(op1);
    const std::shared_ptr<Range> info2 = fetchRange(op2);
    LLVM_DEBUG(tda::log() << " op1 range: " << (info1 ? info1->toString() : "null") << " op2: " << (info2 ? info2->toString() : "null"));
    const std::shared_ptr<Range> res = handleBinaryInstruction(info1, info2, OpCode);
    saveValueRange(&i, res);

    LLVM_DEBUG(
      if (!info1)
        Logger->logInfo("first range is null"));
    LLVM_DEBUG(
      if (!info2)
        Logger->logInfo("second range is null"));
    LLVM_DEBUG(logRangeln(&i));
  }
  else if (OpCode == Instruction::FNeg) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const Value* op1 = i.getOperand(0);
    const std::shared_ptr<Range> info1 = fetchRange(op1);
    const auto res = handleUnaryInstruction(info1, OpCode);
    saveValueRange(&i, res);

    LLVM_DEBUG(
      if (!info1)
        Logger->logInfo("operand range is null"));
    LLVM_DEBUG(logRangeln(&i));
  }
  else {
    switch (OpCode) {
    // memory operations
    case Instruction::Alloca:        handleAllocaInstr(I); break;
    case Instruction::Load:          handleLoadInstr(&i); break;
    case Instruction::Store:         handleStoreInstr(&i); break;
    case Instruction::GetElementPtr: handleGEPInstr(&i); break;
    case Instruction::BitCast:       handleBitCastInstr(I); break;
    case Instruction::Fence:
      LLVM_DEBUG(Logger->logErrorln("Handling of Fence not supported yet")); break; // TODO implement
    case Instruction::AtomicCmpXchg:
      LLVM_DEBUG(Logger->logErrorln("Handling of AtomicCmpXchg not supported yet"));
      break;                                                                                     // TODO implement
    case Instruction::AtomicRMW:
      LLVM_DEBUG(Logger->logErrorln("Handling of AtomicRMW not supported yet"));
      break;                                                                                     // TODO implement

      // other operations
    case Instruction::Ret: handleReturn(I); break;
    case Instruction::Br:
      // do nothing
      break;
    case Instruction::ICmp:
    case Instruction::FCmp:   handleCmpInstr(&i); break;
    case Instruction::PHI:    handlePhiNode(&i); break;
    case Instruction::Select: handleSelect(&i); break;
    case Instruction::UserOp1: // TODO implement
    case Instruction::UserOp2: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of UserOp not supported yet"));
      break;
    case Instruction::VAArg: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of VAArg not supported yet"));
      break;
    case Instruction::ExtractElement: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ExtractElement not supported yet"));
      break;
    case Instruction::InsertElement: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of InsertElement not supported yet"));
      break;
    case Instruction::ShuffleVector: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ShuffleVector not supported yet"));
      break;
    case Instruction::ExtractValue: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ExtractValue not supported yet"));
      break;
    case Instruction::InsertValue: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of InsertValue not supported yet"));
      break;
    case Instruction::LandingPad: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of LandingPad not supported yet"));
      break;
    default: LLVM_DEBUG(Logger->logErrorln("unhandled instruction " + std::to_string(OpCode))); break;
    }
  } // end else
}

void VRAnalyzer::setPathLocalInfo(std::shared_ptr<CodeAnalyzer> SuccAnalyzer,
                                  Instruction* TermInstr,
                                  unsigned SuccIdx) {
  // TODO extract more specific ranges from cmp
}

bool VRAnalyzer::requiresInterpretation(Instruction* I) const {
  assert(I);
  if (CallBase* CB = dyn_cast<CallBase>(I)) {
    if (!CB->isIndirectCall()) {
      Function* Called = CB->getCalledFunction();
      return Called
          && !(Called->isIntrinsic() || isMathCallInstruction(Called->getName().str()) || isMallocLike(Called)
               || Called->empty() // function prototypes
          );
    }
    return true;
  }
  // I is not a call.
  return false;
}

void VRAnalyzer::prepareForCall(Instruction* I, std::shared_ptr<AnalysisStore> FunctionStore) {
  CallBase* CB = cast<CallBase>(I);
  assert(!CB->isIndirectCall());

  LLVM_DEBUG(Logger->logInstruction(I));
  LLVM_DEBUG(Logger->logInfoln("preparing for function interpretation..."));

  LLVM_DEBUG(
    Logger->lineHead();
    log() << "Loading argument ranges: ");
  // fetch ranges of arguments
  std::list<std::shared_ptr<ValueInfo>> ArgRanges;
  for (Value* Arg : CB->args()) {
    ArgRanges.push_back(getNode(Arg));
    LLVM_DEBUG(log() << VRALogger::toString(fetchRangeNode(Arg)) << ", ");
  }
  LLVM_DEBUG(log() << "\n");

  std::shared_ptr<VRAFunctionStore> FStore = std::static_ptr_cast<VRAFunctionStore>(FunctionStore);
  FStore->setArgumentRanges(*CB->getCalledFunction(), ArgRanges);
}

void VRAnalyzer::returnFromCall(Instruction* I, std::shared_ptr<AnalysisStore> FunctionStore) {
  CallBase* CB = cast<CallBase>(I);
  assert(!CB->isIndirectCall());

  LLVM_DEBUG(
    Logger->logInstruction(I);
    Logger->logInfo("returning from call"));
}

void VRAnalyzer::prepareForCallPropagation(Instruction* I, std::shared_ptr<AnalysisStore> FunctionStore) {
  CallBase* CB = cast<CallBase>(I);
  assert(!CB->isIndirectCall());

  LLVM_DEBUG(Logger->logInstruction(I));
  LLVM_DEBUG(Logger->logInfoln("preparing for function interpretation..."));

  LLVM_DEBUG(
    Logger->lineHead();
    log() << "Loading argument ranges: ");
  // fetch ranges of arguments
  std::list<std::shared_ptr<ValueInfo>> ArgRanges;
  for (Value* Arg : CB->args()) {
    ArgRanges.push_back(getNode(Arg));
    LLVM_DEBUG(log() << VRALogger::toString(fetchRangeNode(Arg)) << ", ");
  }
  LLVM_DEBUG(log() << "\n");

  std::shared_ptr<VRAFunctionStore> FStore = std::static_ptr_cast<VRAFunctionStore>(FunctionStore);
  FStore->setArgumentRanges(*CB->getCalledFunction(), ArgRanges);
}

void VRAnalyzer::returnFromCallPropagation(Instruction* I, std::shared_ptr<AnalysisStore> FunctionStore) {
  CallBase* CB = cast<CallBase>(I);
  assert(!CB->isIndirectCall());

  LLVM_DEBUG(
    Logger->logInstruction(I);
    Logger->logInfo("returning from call"));

  std::shared_ptr<VRAFunctionStore> FStore = std::static_ptr_cast<VRAFunctionStore>(FunctionStore);
  std::shared_ptr<ValueInfo> Ret = FStore->getRetVal();
  if (!Ret) {
    LLVM_DEBUG(Logger->logInfoln("function returns nothing"));
  }
  else if (std::shared_ptr<ValueInfoWithRange> RetRange = std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(Ret)) {
    saveValueRange(I, RetRange);
    LLVM_DEBUG(logRangeln(I));
  }
  else {
    setNode(I, Ret);
    LLVM_DEBUG(Logger->logRangeln(Ret));
  }
}

std::shared_ptr<Range> VRAnalyzer::getRange(const std::shared_ptr<ValueInfo> VI) {
  if (VI) {
    switch (VI->getKind()) {
    case ValueInfo::K_Scalar: {
      const std::shared_ptr<ScalarInfo> ScalarNode = std::static_ptr_cast<ScalarInfo>(VI);
      return ScalarNode->range;
    }
    case ValueInfo::K_Struct: {
      std::shared_ptr<StructInfo> StructNode = std::static_ptr_cast<StructInfo>(VI);
      std::shared_ptr<Range> summary = nullptr;
      for (const std::shared_ptr<ValueInfo>& Field : *StructNode) {
        const std::shared_ptr<Range> fieldRange = getRange(Field);
        if (!fieldRange)
          continue;
        if (!summary)
          summary = fieldRange->clone();
        else
          summary = summary->join(fieldRange);
      }
      return summary;
    }
    case ValueInfo::K_Array: {
      std::shared_ptr<ArrayInfo> ArrayNode = std::static_ptr_cast<ArrayInfo>(VI);
      std::shared_ptr<Range> summary = nullptr;

      for (unsigned i = 0; i < ArrayNode->getNumElements(); ++i) {
        std::shared_ptr<ValueInfo> Element = ArrayNode->getElement(i);
        const std::shared_ptr<Range> elementRange = getRange(Element);
        if (!elementRange)
          continue;
        if (!summary)
          summary = elementRange->clone();
        else
          summary = summary->join(elementRange);
      }

      if (!summary) {
        // fallback: return generic range
        summary = std::make_shared<Range>(Range::Top());
      }
      return summary;
    }
    case ValueInfo::K_GetElementPointer: {
      return nullptr;
    }
    case ValueInfo::K_Pointer: {
      return nullptr;
    }
    default: llvm_unreachable("Unhandled node type.");
    }
  }
  return nullptr;
}

std::shared_ptr<Range> VRAnalyzer::getBBRange(const llvm::Value *V) {
  return getRange(getNode(V));
}

////////////////////////////////////////////////////////////////////////////////
// Instruction Handlers
////////////////////////////////////////////////////////////////////////////////

void VRAnalyzer::handleSpecialCall(const Instruction* I) {
  const CallBase* CB = cast<CallBase>(I);
  LLVM_DEBUG(Logger->logInstruction(I));

  // fetch function name
  Function* Callee = CB->getCalledFunction();
  if (Callee == nullptr) {
    LLVM_DEBUG(Logger->logInfo("indirect calls not supported"));
    return;
  }

  // check if it's an OMP library function and handle it if so
  if (detectAndHandleLibOMPCall(CB))
    return;

  const StringRef FunctionName = Callee->getName();
  if (isMathCallInstruction(FunctionName.str())) {
    // fetch ranges of arguments
    std::list<std::shared_ptr<Range>> ArgScalarRanges;
    for (Value* Arg : CB->args())
      ArgScalarRanges.push_back(fetchRange(Arg));
    std::shared_ptr<Range> Res = handleMathCallInstruction(ArgScalarRanges, FunctionName.str());
    saveValueRange(I, Res);
    LLVM_DEBUG(Logger->logInfo("whitelisted"));
    LLVM_DEBUG(Logger->logRangeln(Res));
  }
  else if (isMallocLike(Callee)) {
    handleMallocCall(CB);
  }
  else if (Callee->isIntrinsic()) {
    const auto IntrinsicsID = Callee->getIntrinsicID();
    switch (IntrinsicsID) {
    case Intrinsic::memcpy: handleMemCpyIntrinsics(CB); break;
    default: LLVM_DEBUG(Logger->logInfoln("skipping intrinsic " + std::string(FunctionName)));
    }
  }
  else {
    LLVM_DEBUG(Logger->logInfoln("unsupported call"));
  }
}

void VRAnalyzer::handleMemCpyIntrinsics(const Instruction* memcpy) {
  assert(isa<CallInst>(memcpy) || isa<InvokeInst>(memcpy));
  LLVM_DEBUG(Logger->logInfo("llvm.memcpy"));
  const BitCastInst* dest_bitcast = dyn_cast<BitCastInst>(memcpy->getOperand(0U));
  const BitCastInst* src_bitcast = dyn_cast<BitCastInst>(memcpy->getOperand(1U));
  if (!(dest_bitcast && src_bitcast)) {
    LLVM_DEBUG(Logger->logInfo("operand is not bitcast, aborting"));
    return;
  }
  const Value* dest = dest_bitcast->getOperand(0U);
  const Value* src = src_bitcast->getOperand(0U);

  const std::shared_ptr<ValueInfo> src_node = loadNode(getNode(src));
  storeNode(getNode(dest), src_node);
  LLVM_DEBUG(Logger->logRangeln(fetchRangeNode(src)));
}

bool VRAnalyzer::isMallocLike(const Function* F) const {
  const StringRef FName = F->getName();
  // TODO make sure this works in other platforms
  return FName == "malloc" || FName == "calloc" || FName == "_Znwm" || FName == "_Znam";
}

bool VRAnalyzer::isCallocLike(const Function* F) const {
  const StringRef FName = F->getName();
  // TODO make sure this works in other platforms
  return FName == "calloc";
}

void VRAnalyzer::handleMallocCall(const CallBase* CB) {
  LLVM_DEBUG(Logger->logInfo("malloc-like"));
  const Type* AllocatedType = nullptr;

  auto inputInfo = getGlobalStore()->getUserInput(CB);
  if (AllocatedType && AllocatedType->isStructTy()) {
    if (inputInfo && std::isa_ptr<StructInfo>(inputInfo))
      DerivedRanges[CB] = inputInfo->clone();
    else
      DerivedRanges[CB] = std::make_shared<StructInfo>(0);
    LLVM_DEBUG(Logger->logInfoln("struct"));
  }
  else {
    if (!(AllocatedType && AllocatedType->isPointerTy())) {
      if (inputInfo && std::isa_ptr<ScalarInfo>(inputInfo)) {
        DerivedRanges[CB] = std::make_shared<PointerInfo>(inputInfo);
      }
      else if (isCallocLike(CB->getCalledFunction())) {
        DerivedRanges[CB] =
          std::make_shared<PointerInfo>(std::make_shared<ScalarInfo>(nullptr, std::make_shared<Range>(0, 0)));
      }
      else {
        DerivedRanges[CB] = std::make_shared<PointerInfo>(nullptr);
      }
    }
    else {
      DerivedRanges[CB] = std::make_shared<PointerInfo>(nullptr);
    }
    LLVM_DEBUG(Logger->logInfoln("pointer"));
  }
}

bool VRAnalyzer::detectAndHandleLibOMPCall(const CallBase* CB) {
  Function* F = CB->getCalledFunction();
  if (F->getName() == "__kmpc_for_static_init_4") {
    Value* VPLower = CB->getArgOperand(4U);
    Value* VPUpper = CB->getArgOperand(5U);
    std::shared_ptr<Range> PLowerRange = fetchRange(VPLower);
    std::shared_ptr<Range> PUpperRange = fetchRange(VPUpper);
    if (!PLowerRange || !PUpperRange) {
      LLVM_DEBUG(Logger->logInfoln("ranges of plower/pupper unknown, doing nothing"));
      return true;
    }
    std::shared_ptr<Range> Merge = getUnionRange(PLowerRange, PUpperRange);
    saveValueRange(VPLower, Merge);
    saveValueRange(VPUpper, Merge);
    LLVM_DEBUG(Logger->logRange(Merge));
    LLVM_DEBUG(Logger->logInfoln(" set to plower, pupper nodes"));
    return true;
  }
  return false;
}

void VRAnalyzer::handleReturn(const Instruction* ret) {
  const ReturnInst* ret_i = cast<ReturnInst>(ret);
  LLVM_DEBUG(Logger->logInstruction(ret));
  if (const Value* ret_val = ret_i->getReturnValue()) {
    std::shared_ptr<ValueInfo> range = getNode(ret_val);

    std::shared_ptr<VRAFunctionStore> FStore;
    if (CodeInt) {
      FStore = std::static_ptr_cast<VRAFunctionStore>(CodeInt->getFunctionStore());
    } else {
      FStore = std::static_ptr_cast<VRAFunctionStore>(ModInt->getFunctionStore());
    }
    FStore->setRetVal(range);

    LLVM_DEBUG(Logger->logRangeln(range));
  }
  else {
    LLVM_DEBUG(Logger->logInfoln("void return."));
  }
}

void VRAnalyzer::handleAllocaInstr(Instruction* I) {
  auto* allocaInst = cast<AllocaInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const auto inputValueInfo = getGlobalStore()->getUserInput(I);
  auto* allocatedType = TaffoInfo::getInstance().getOrCreateTransparentType(*allocaInst);
  if (allocatedType->isStructTTOrPtrTo()) {
    auto* structType = cast<TransparentStructType>(allocatedType->getFirstNonPtr());
    if (inputValueInfo && std::isa_ptr<StructInfo>(inputValueInfo))
      DerivedRanges[I] = inputValueInfo->clone();
    else
      DerivedRanges[I] = ValueInfoFactory::create(structType);
    LLVM_DEBUG(Logger->logInfoln("struct"));
  }
  else if (allocatedType->isArrayTTOrPtrTo()) {
    auto* arrayType = cast<TransparentArrayType>(allocatedType->getFirstNonPtr());
    if (inputValueInfo && std::isa_ptr<ArrayInfo>(inputValueInfo))
      DerivedRanges[I] = inputValueInfo->clone();
    else
      DerivedRanges[I] = ValueInfoFactory::create(arrayType);
    LLVM_DEBUG(Logger->logInfoln("array"));
  }
  else {
    if (inputValueInfo && std::isa_ptr<ScalarInfo>(inputValueInfo))
      DerivedRanges[I] = std::make_shared<PointerInfo>(inputValueInfo);
    else
      DerivedRanges[I] = std::make_shared<PointerInfo>(nullptr);
    LLVM_DEBUG(Logger->logInfoln("pointer"));
  }
  
}

void VRAnalyzer::handleStoreInstr(const Instruction* I) {
  const StoreInst* Store = cast<StoreInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const Value* AddressParam = Store->getPointerOperand();
  const Value* ValueParam = Store->getValueOperand();

  if (isa<ConstantPointerNull>(ValueParam))
    return;

  std::shared_ptr<ValueInfo> AddressNode = getNode(AddressParam);
  if (!AddressNode) {
    if (const Value* Base = getBaseMemoryObject(AddressParam)) {
      AddressNode = getNode(Base);
    }
  }

  if (const Value* Base = getBaseMemoryObject(AddressParam)) {
    if (auto ArrayNode = std::dynamic_ptr_cast_or_null<ArrayInfo>(loadNode(getNode(Base)))) {
      std::shared_ptr<Range> valRange = fetchRange(ValueParam);
      if (!valRange) valRange = getRange(getNode(ValueParam));
      
      if (valRange) {
        for (unsigned i = 0; i < ArrayNode->getNumElements(); ++i) {
          std::shared_ptr<ValueInfo> el = ArrayNode->getElement(i);
          std::shared_ptr<ValueInfo> valNode = std::make_shared<ScalarInfo>(nullptr, valRange);
          if (el) {
            ArrayNode->setElement(i, assignScalarRange(el, valNode));
          } else {
            ArrayNode->setElement(i, valNode);
          }
        }
      }
      return; 
    }
  }

  // Comportamento originale per gli scalari
  std::shared_ptr<ValueInfo> ValueNode = getNode(ValueParam);
  const std::shared_ptr<Range> oldValueRange = getRange(ValueNode);
  const std::shared_ptr<Range> oldPointedRange = getRange(loadNode(AddressNode));

  if (!ValueNode && !ValueParam->getType()->isPointerTy())
    ValueNode = fetchRangeNode(I);

  if (!ValueParam->getType()->isPointerTy()) {
    std::shared_ptr<Range> currentRange = getRange(ValueNode);
    if (!currentRange) currentRange = fetchRange(ValueParam);
      
    if (oldPointedRange && currentRange) {
      currentRange = currentRange->join(oldPointedRange);
    }
    
    if (auto Scalar = std::dynamic_ptr_cast_or_null<ScalarInfo>(ValueNode)) {
      auto Cloned = std::static_pointer_cast<ScalarInfo>(Scalar->clone());
      Cloned->range = currentRange;
      ValueNode = Cloned;
    } else if (!ValueNode) {
      ValueNode = std::make_shared<ScalarInfo>(nullptr, currentRange);
    }
  }

  storeNode(AddressNode, ValueNode);
  LLVM_DEBUG(Logger->logRangeln(ValueNode));
}

void VRAnalyzer::handleLoadInstr(Instruction* I) {
  LoadInst* Load = cast<LoadInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const Value* PointerOp = Load->getPointerOperand();

  std::shared_ptr<ValueInfo> AddressNode = getNode(PointerOp);
  if (!AddressNode) {
    if (const Value* Base = getBaseMemoryObject(PointerOp)) {
      AddressNode = getNode(Base);
    }
  }

  std::shared_ptr<ValueInfo> Loaded = loadNode(AddressNode);
  std::shared_ptr<ScalarInfo> Scalar = std::dynamic_ptr_cast_or_null<ScalarInfo>(Loaded);

  if (!Scalar) {
    if (const Value* Base = getBaseMemoryObject(PointerOp)) {
      if (auto ArrayNode = std::dynamic_ptr_cast_or_null<ArrayInfo>(loadNode(getNode(Base)))) {
        std::shared_ptr<Range> summaryRange = getRange(ArrayNode);
        if (!summaryRange) {
          summaryRange = std::make_shared<Range>(Range::Top());
        }
        if (Load->getType()->isSingleValueType()) {
          Scalar = std::make_shared<ScalarInfo>(nullptr, summaryRange);
        }
      }
    }
  }

  std::shared_ptr<Range> recurrenceRange = nullptr;
  if (ModInt) {
    if (const Value* Base = getBaseMemoryObject(PointerOp))
      recurrenceRange = ModInt->getLastStoredRange(Base);
  }

  // Comportamento originale MemorySSA
  if (Scalar) {
    llvm::MemorySSAAnalysis::Result *SSARes;
    if (CodeInt) {
      auto& FAM = CodeInt->getMAM().getResult<FunctionAnalysisManagerModuleProxy>(*I->getFunction()->getParent()).getManager();
      SSARes = &(FAM.getResult<MemorySSAAnalysis>(*I->getFunction()));
    } else {
      auto& FAM = ModInt->getMAM().getResult<FunctionAnalysisManagerModuleProxy>(*I->getFunction()->getParent()).getManager();
      SSARes = &(FAM.getResult<MemorySSAAnalysis>(*I->getFunction()));
    }
    
    MemorySSA& memssa = SSARes->getMSSA();
    MemSSAUtils memssa_utils(memssa);
    SmallVectorImpl<Value*>& def_vals = memssa_utils.getDefiningValues(Load);

    Type* load_ty = getFullyUnwrappedType(Load);
    std::shared_ptr<Range> res = Scalar->range;
    for (Value* dval : def_vals)
      if (dval && load_ty->canLosslesslyBitCastTo(getFullyUnwrappedType(dval)))
        res = getUnionRange(res, fetchRange(dval));
    if (recurrenceRange)
      res = getUnionRange(res, recurrenceRange);
    saveValueRange(I, res);

    LLVM_DEBUG(Logger->logRangeln(res));
  }
  else if (Loaded) {
    setNode(I, Loaded);
    LLVM_DEBUG(Logger->logInfoln("pointer load"));
  }
  else {
    LLVM_DEBUG(Logger->logInfoln("unable to retrieve loaded value"));
  }
}

void VRAnalyzer::handleGEPInstr(const Instruction* I) {
  const GetElementPtrInst* gepInst = cast<GetElementPtrInst>(I);
  LLVM_DEBUG(Logger->logInstruction(gepInst));

  std::shared_ptr<ValueInfo> Node = getNode(gepInst);
  if (Node) {
    LLVM_DEBUG(Logger->logInfoln("has node"));
    return;
  }
  
  SmallVector<unsigned, 1> Offset;
  bool success = extractGEPOffset(
        gepInst->getSourceElementType(), iterator_range(gepInst->idx_begin(), gepInst->idx_end()), Offset);

  if (!success) {
    Offset.clear();
    for (auto it = gepInst->idx_begin() + 1; it != gepInst->idx_end(); ++it) {
      Offset.push_back(0);
    }
  }

  std::shared_ptr<ValueInfo> BaseNode = getNode(gepInst->getPointerOperand());
  if (!BaseNode) {
    if (const Value* RootBase = getBaseMemoryObject(gepInst->getPointerOperand())) {
      BaseNode = getNode(RootBase);
    }
  }

  Node = std::make_shared<GEPInfo>(BaseNode, Offset);
  setNode(I, Node);
}

void VRAnalyzer::handleBitCastInstr(Instruction* I) {
  LLVM_DEBUG(Logger->logInstruction(I));
  if (std::shared_ptr<ValueInfo> Node = getNode(I->getOperand(0U))) {
    bool InputIsStruct = getFullyUnwrappedType(I->getOperand(0U))->isStructTy();
    bool OutputIsStruct = getFullyUnwrappedType(I)->isStructTy();
    if (!InputIsStruct && !OutputIsStruct) {
      setNode(I, Node);
      LLVM_DEBUG(Logger->logRangeln(Node));
    }
    else {
      LLVM_DEBUG(Logger->logInfoln("oh shit -> no node"));
      LLVM_DEBUG(
        log()
        << "This instruction is converting to/from a struct type. Ignoring to avoid generating invalid metadata\n");
    }
  }
  else {
    LLVM_DEBUG(Logger->logInfoln("no node"));
  }
}

void VRAnalyzer::handleCmpInstr(const Instruction* cmp) {
  const CmpInst* cmp_i = cast<CmpInst>(cmp);
  LLVM_DEBUG(Logger->logInstruction(cmp));

  const std::shared_ptr<Range> oldInfo = fetchRange(cmp_i);
  
  const CmpInst::Predicate pred = cmp_i->getPredicate();
  std::list<std::shared_ptr<Range>> ranges;
  for (unsigned index = 0; index < cmp_i->getNumOperands(); index++) {
    const Value* op = cmp_i->getOperand(index);
    if (std::shared_ptr<ScalarInfo> op_range = std::dynamic_ptr_cast_or_null<ScalarInfo>(getNode(op)))
      ranges.push_back(op_range->range);
    else
      ranges.push_back(nullptr);
  }
  std::shared_ptr<Range> result = std::dynamic_ptr_cast_or_null<Range>(handleCompare(ranges, pred));
  LLVM_DEBUG(Logger->logRangeln(result));
  saveValueRange(cmp, result);
}

void VRAnalyzer::handlePhiNode(const Instruction* phi) {
  const PHINode* phi_n = cast<PHINode>(phi);
  if (phi_n->getNumIncomingValues() == 0U)
    return;
  LLVM_DEBUG(Logger->logInstruction(phi));
  auto res = copyRange(getGlobalStore()->getUserInput(phi));

  for (unsigned index = 0U; index < phi_n->getNumIncomingValues(); index++) {
    const Value* op = phi_n->getIncomingValue(index);
    std::shared_ptr<ValueInfo> op_node = getNode(op);
    if (!op_node)
      continue;
    if (std::shared_ptr<ValueInfoWithRange> op_range = std::dynamic_ptr_cast<ScalarInfo>(op_node)) {
      res = getUnionRange(res, op_range);
    }
    else {
      setNode(phi, op_node);
      LLVM_DEBUG(Logger->logInfoln("Pointer PHINode"));
      return;
    }
  }
  setNode(phi, res);

  
  LLVM_DEBUG(Logger->logRangeln(res));
}

void VRAnalyzer::analyzePHIStartInstruction(llvm::Instruction* I) {
  const PHINode* phi_n = cast<PHINode>(I);
  if (phi_n->getNumIncomingValues() == 0U)
    return;
  LLVM_DEBUG(Logger->logInstruction(I));

  const Value* op = phi_n->getIncomingValue(0);
  if (!fetchRange(op)) return;
  std::shared_ptr<ValueInfo> op_node = getNode(op);
  if (std::shared_ptr<ValueInfoWithRange> op_range = std::dynamic_ptr_cast<ScalarInfo>(op_node)) {
    setNode(I, copyRange(op_range));
    LLVM_DEBUG(Logger->logRangeln(op_range));
  } else {
    LLVM_DEBUG(tda::log() << "unable to retreve start operand of phi node\n");
  }
}

void VRAnalyzer::handleSelect(const Instruction* i) {
  const SelectInst* sel = cast<SelectInst>(i);
  // TODO handle pointer select
  LLVM_DEBUG(Logger->logInstruction(sel));
  std::shared_ptr<ValueInfoWithRange> res =
    getUnionRange(fetchRangeNode(sel->getFalseValue()), fetchRangeNode(sel->getTrueValue()));
  LLVM_DEBUG(Logger->logRangeln(res));
  saveValueRange(i, res);
}

////////////////////////////////////////////////////////////////////////////////
// Data Handling
////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<Range> VRAnalyzer::fetchRange(const Value* v) {
  if (const std::shared_ptr<Range> Derived = VRAStore::fetchRange(v))
    return Derived;

  if (const std::shared_ptr<ValueInfoWithRange> InputRange = getGlobalStore()->getUserInput(v))
    if (const std::shared_ptr<ScalarInfo> InputScalar = std::dynamic_ptr_cast<ScalarInfo>(InputRange))
      return InputScalar->range;

  if (const ConstantFP* CFP = dyn_cast<ConstantFP>(v)) {
    double val = CFP->getValueAPF().convertToDouble();
    return std::make_shared<Range>(val, val);
  }
  if (const ConstantInt* CI = dyn_cast<ConstantInt>(v)) {
    double val = static_cast<double>(CI->getSExtValue());
    return std::make_shared<Range>(val, val);
  }

  if (auto node = getNode(v)) {
    if (auto scalarInfo = std::dynamic_ptr_cast_or_null<ScalarInfo>(node)) {
      return scalarInfo->range;
    }
  }

  return nullptr;
}

static std::shared_ptr<ValueInfoWithRange> safeMergeRanges(
    std::shared_ptr<ValueInfoWithRange> derived,
    std::shared_ptr<ValueInfoWithRange> input) 
{
  if (!derived) return input;
  if (!input) return derived;

  // Se sono entrambi scalari, uniamo i range
  if (derived->getKind() == ValueInfo::K_Scalar && input->getKind() == ValueInfo::K_Scalar) {
    auto d_scal = std::static_pointer_cast<ScalarInfo>(derived);
    auto i_scal = std::static_pointer_cast<ScalarInfo>(input);
    auto res = std::static_pointer_cast<ScalarInfo>(d_scal->clone());
    if (d_scal->range && i_scal->range) res->range = d_scal->range->join(i_scal->range);
    else if (i_scal->range) res->range = i_scal->range->clone();
    return res;
  }
  // Se sono entrambi array, iteriamo ricorsivamente
  else if (derived->getKind() == ValueInfo::K_Array && input->getKind() == ValueInfo::K_Array) {
    auto d_arr = std::static_pointer_cast<ArrayInfo>(derived);
    auto i_arr = std::static_pointer_cast<ArrayInfo>(input);
    auto res = std::make_shared<ArrayInfo>(d_arr->getNumElements());
    unsigned min_len = std::min(d_arr->getNumElements(), i_arr->getNumElements());
    
    for (unsigned i = 0; i < min_len; ++i) {
      auto d_el = std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(d_arr->getElement(i));
      auto i_el = std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(i_arr->getElement(i));
      res->setElement(i, safeMergeRanges(d_el, i_el));
    }
    for (unsigned i = min_len; i < d_arr->getNumElements(); ++i) {
      if (d_arr->getElement(i)) res->setElement(i, d_arr->getElement(i)->clone());
    }
    return res;
  }
  // Se sono struct, iteriamo sui campi
  else if (derived->getKind() == ValueInfo::K_Struct && input->getKind() == ValueInfo::K_Struct) {
    auto d_str = std::static_pointer_cast<StructInfo>(derived);
    auto i_str = std::static_pointer_cast<StructInfo>(input);
    auto res = std::make_shared<StructInfo>(d_str->getNumFields());
    unsigned min_len = std::min(d_str->getNumFields(), i_str->getNumFields());
    
    for (unsigned i = 0; i < min_len; ++i) {
      auto d_fld = std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(d_str->getField(i));
      auto i_fld = std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(i_str->getField(i));
      res->setField(i, safeMergeRanges(d_fld, i_fld));
    }
    return res;
  }
  // Se TAFFO ha capito che è un aggregato (Array/Struct), ma l'utente ha
  // annotato un semplice "scalar", spalmiamo quel range su tutte le foglie
  else if (input->getKind() == ValueInfo::K_Scalar) {
    auto i_scal = std::static_pointer_cast<ScalarInfo>(input);
    auto res = derived->clone<ValueInfoWithRange>();
    
    std::function<void(std::shared_ptr<ValueInfo>)> smear = [&](std::shared_ptr<ValueInfo> node) {
      if (!node) return;
      if (node->getKind() == ValueInfo::K_Scalar) {
        auto s = std::static_pointer_cast<ScalarInfo>(node);
        if (s->range && i_scal->range) s->range = s->range->join(i_scal->range);
        else if (i_scal->range) s->range = i_scal->range->clone();
      } else if (node->getKind() == ValueInfo::K_Array) {
        auto a = std::static_pointer_cast<ArrayInfo>(node);
        for(unsigned i=0; i<a->getNumElements(); ++i) smear(a->getElement(i));
      } else if (node->getKind() == ValueInfo::K_Struct) {
        auto s = std::static_pointer_cast<StructInfo>(node);
        for(unsigned i=0; i<s->getNumFields(); ++i) smear(s->getField(i));
      } else if (node->getKind() == ValueInfo::K_Pointer) {
        auto p = std::static_pointer_cast<PointerInfo>(node);
        smear(p->getPointed());
      }
    };
    
    smear(res);
    return res;
  }
  
  // Fallback sicuro se i tipi sono incompatibili (es. Pointer vs Array)
  return derived->clone<ValueInfoWithRange>();
}

std::shared_ptr<ValueInfoWithRange> VRAnalyzer::fetchRangeNode(const Value* v) {
  if (const std::shared_ptr<ValueInfoWithRange> Derived = VRAStore::fetchRangeNode(v)) {
    if (auto InputRange = getGlobalStore()->getUserInput(v)) {
      // Usiamo il bypass sicuro invece di fillRangeHoles
      return safeMergeRanges(Derived, InputRange->clone<ValueInfoWithRange>());
    }
    return Derived;
  }

  if (const auto InputRange = getGlobalStore()->getUserInput(v))
    return InputRange->clone<ValueInfoWithRange>();

  return nullptr;
}

std::shared_ptr<ValueInfo> VRAnalyzer::getNode(const Value* v) {
  std::shared_ptr<ValueInfo> Node = VRAStore::getNode(v);

  if (!Node) {
    std::shared_ptr<VRAStore> ExternalStore = getAnalysisStoreForValue(v);
    if (ExternalStore)
      Node = ExternalStore->getNode(v);
  }

  if (Node && Node->getKind() == ValueInfo::K_Scalar) {
    auto UserInput = std::dynamic_ptr_cast_or_null<ScalarInfo>(getGlobalStore()->getUserInput(v));
    if (UserInput && UserInput->isFinal())
      Node = UserInput->clone();
  }

  return Node;
}

void VRAnalyzer::setNode(const Value* V, std::shared_ptr<ValueInfo> Node) {
  if (isa<GlobalVariable>(V)) {
    // set node in global analyzer
    getGlobalStore()->setNode(V, Node);
    return;
  }
  if (isa<Argument>(V)) {
    std::shared_ptr<VRAFunctionStore> FStore;
    if (CodeInt) FStore = std::static_ptr_cast<VRAFunctionStore>(CodeInt->getFunctionStore());
    else FStore = std::static_ptr_cast<VRAFunctionStore>(ModInt->getFunctionStore());
    FStore->setNode(V, Node);
    return;
  }

  VRAStore::setNode(V, Node);
}

void VRAnalyzer::logRangeln(const Value* v) {
  LLVM_DEBUG(
    if (getGlobalStore()->getUserInput(v))
      log() << "(possibly from metadata) ");
  LLVM_DEBUG(Logger->logRangeln(fetchRangeNode(v)));
}

size_t VRAnalyzer::compareLoadStoreDim(VRAFunctionInfo VFI, const llvm::Value *load, const llvm::Value *store) {

  const auto *LoadI = dyn_cast<LoadInst>(load);
  const auto *StoreI = dyn_cast<StoreInst>(store);
  if (!LoadI || !StoreI)
    return 0;
    
  auto getRootPtr = [&](Instruction *I) -> const Value * {
    const Value *Ptr = isa<LoadInst>(I) ? cast<LoadInst>(I)->getPointerOperand()
                                        : cast<StoreInst>(I)->getPointerOperand();
    if (!Ptr)
      return nullptr;

    if (auto *PtrV = const_cast<Value *>(Ptr))
      if (Value *Origin = MemSSAUtils::getOriginPointer(*VFI.MSSA, PtrV))
        return Origin;
    return Ptr;
  };
  
  const Value *LoadPtr = getRootPtr(const_cast<LoadInst*>(LoadI));
  const Value *StorePtr = getRootPtr(const_cast<StoreInst*>(StoreI));
  if (!LoadPtr || !StorePtr)
    return 0;
    
  const int LoadDim = getArrayAccessDimFromPtr(LoadPtr);
  const int StoreDim = getArrayAccessDimFromPtr(StorePtr);
  if (LoadDim < 0 || StoreDim < 0 || LoadDim <= StoreDim)
    return 0;

  return static_cast<size_t>(LoadDim - StoreDim);

}

void VRAnalyzer::resolveRecurrence(VRARecurrenceInfo& VRI, unsigned TripCount) {
  if (!VRI.RR || TripCount == 0) return;

  LLVM_DEBUG(Logger->logInstruction(VRI.root));

  // already solved, solve again just for higher trip count
  if (VRI.lastRange && VRI.lastRangeComputedAt >= TripCount) {
    LLVM_DEBUG(tda::log() << " RR already solved.\n");
    return;
  }

  if (TripCount > 0) {

    auto joinedRange = getRRJoinedRange(VRI.RR.get(), TripCount);
    
    VRI.lastRange = joinedRange;
    VRI.lastRangeComputedAt = TripCount;
    LLVM_DEBUG(tda::log() << " resolved RR " << *VRI.root << ".at("<<TripCount<<") ");

    if (auto* PN = dyn_cast<PHINode>(VRI.root)) {

      const Value* op = PN->getIncomingValue(0);
      std::shared_ptr<ValueInfo> op_node = getNode(op);
      if (std::shared_ptr<ValueInfoWithRange> op_range = std::dynamic_ptr_cast<ScalarInfo>(op_node)) {

        auto phiStart = std::make_shared<ScalarInfo>(nullptr, joinedRange);
        setNode(VRI.root, phiStart);
        LLVM_DEBUG(Logger->logRangeln(phiStart));

        if (auto outerRR = dyn_cast<AffineDeltaRangedRecurrence>(VRI.RR)) {
          LLVM_DEBUG(tda::log() << "\n\tè una delta, necessita update inner RR da ultimo start\n");
          auto curRRStep = outerRR->getInnerRR()->getStep()->clone();

          auto* innerVRI = ModInt->getVRARecurrenceInfo(VRI.innerRR);
          innerVRI->RR = std::make_shared<AffineRangedRecurrence>(getRRJoinedRange(VRI.RR.get(), TripCount - 1), std::move(curRRStep));
          innerVRI->lastRange = getRRJoinedRange(innerVRI->RR.get(), innerVRI->lastRangeComputedAt);
          LLVM_DEBUG(tda::log() << "\t\treplaced with "<<innerVRI->RR->toString()<<"\n");
        }
      }
      return;
    } else if (auto Store = dyn_cast<StoreInst>(VRI.root)) {
      
      const Value* AddressParam = Store->getPointerOperand();
      const Value* ValueParam = Store->getValueOperand();

      if (isa<ConstantPointerNull>(ValueParam)) return;

      std::shared_ptr<ValueInfo> AddressNode = getNode(AddressParam);
      std::shared_ptr<ValueInfo> ValueNode = getNode(ValueParam);

      auto oldRange = getRange(loadNode(AddressNode));
      if (oldRange) { 
        joinedRange = joinedRange->join(oldRange);
        VRI.lastRange = joinedRange;
      }

      if (!ValueNode && !ValueParam->getType()->isPointerTy())
        ValueNode = fetchRangeNode(VRI.root);
        
      if (!ValueParam->getType()->isPointerTy()) {
        if (auto Scalar = std::dynamic_ptr_cast_or_null<ScalarInfo>(ValueNode)) {
          auto Cloned = std::static_pointer_cast<ScalarInfo>(Scalar->clone());
          Cloned->range = joinedRange;
          ValueNode = Cloned;
        } else if (!ValueNode) {
          ValueNode = std::make_shared<ScalarInfo>(nullptr, joinedRange);
        }
      }

      storeNode(AddressNode, ValueNode);
      LLVM_DEBUG(Logger->logRangeln(joinedRange));
      return;
    }
  }
  LLVM_DEBUG(tda::log() << "unable to resolve and store recurrence\n");
}

void VRAnalyzer::retrieveSolvedRecurrence(llvm::Instruction* I, VRARecurrenceInfo& VRI) {
  if (!VRI.lastRange)
    return;

  LLVM_DEBUG(Logger->logInstruction(I));
  auto makeSolvedScalarNode = [&](const std::shared_ptr<ValueInfo>& baseNode) -> std::shared_ptr<ValueInfo> {
    if (baseNode && baseNode->getKind() == ValueInfo::K_Scalar) {
      auto Scalar = std::static_pointer_cast<ScalarInfo>(baseNode);
      auto Cloned = std::static_pointer_cast<ScalarInfo>(Scalar->clone());
      Cloned->range = VRI.lastRange;
      return Cloned;
    }
    return std::make_shared<ScalarInfo>(nullptr, VRI.lastRange);
  };
  
  if (llvm::dyn_cast<llvm::PHINode>(VRI.root)) {
    auto solved = makeSolvedScalarNode(getNode(VRI.root));
    setNode(VRI.root, solved);

    LLVM_DEBUG(Logger->logRangeln(VRI.lastRange));
    return;
  }
  if (auto* Store = llvm::dyn_cast<llvm::StoreInst>(VRI.root)) {
    const llvm::Value* AddressParam = Store->getPointerOperand();
    const llvm::Value* ValueParam   = Store->getValueOperand();

    if (llvm::isa<llvm::ConstantPointerNull>(ValueParam))
      return;

    std::shared_ptr<ValueInfo> AddressNode = getNode(AddressParam);
    std::shared_ptr<ValueInfo> ValueNode = getNode(ValueParam);

    if (!ValueNode && !ValueParam->getType()->isPointerTy())
      ValueNode = fetchRangeNode(VRI.root);

    if (!ValueParam->getType()->isPointerTy()) {
      ValueNode = makeSolvedScalarNode(ValueNode);
    }

    storeNode(AddressNode, ValueNode);
    LLVM_DEBUG(Logger->logRangeln(VRI.lastRange));
    return;
  }
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildAffinePHIRecurrence(const llvm::PHINode *PHI) {

  VRAFunctionInfo* VFI = ModInt->getVRAFunctionInfo(const_cast<PHINode*>(PHI)->getParent()->getParent());
  auto *L = VFI->LI->getLoopFor(PHI->getParent());
  auto *Preheader = L ? L->getLoopPreheader() : nullptr;
  auto *Latch     = L ? L->getLoopLatch() : nullptr;

  auto StartRange = getRange(getNode(PHI->getIncomingValue(0)));

  auto StepRange = extractDeltaFromPhiValue(PHI, Preheader, Latch);
  if (!StepRange) {
    //fallback valid for affine
    auto NextRange = getRange(getNode(PHI->getIncomingValue(1)));
    StepRange = handleSub(NextRange, StartRange);
  }

  return std::make_shared<AffineRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildPHIAffineFlattingRecurrence(VRARecurrenceInfo VRI, const llvm::PHINode* PHI) {

  VRAFunctionInfo* VFI = ModInt->getVRAFunctionInfo(const_cast<PHINode*>(PHI)->getParent()->getParent());
  auto *L = VFI->LI->getLoopFor(PHI->getParent());
  auto *Preheader = L ? L->getLoopPreheader() : nullptr;
  auto *Latch     = L ? L->getLoopLatch() : nullptr;

  auto StartRange = getRange(getNode(PHI->getIncomingValue(0)));

  auto StepRange = extractDeltaFromPhiValue(PHI, Preheader, Latch);
  if (!StepRange) {
    //fallback valid for affine
    auto NextRange = getRange(getNode(VRI.loadHigherDim));
    StepRange = handleSub(NextRange, StartRange);
  }

  return std::make_shared<AffineFlattenedRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildPHIGeometricFlattingRecurrence(VRARecurrenceInfo VRI, const llvm::PHINode* PHI) {

  auto StartRange = getRange(getNode(PHI->getIncomingValue(0)));
  auto RatioRange = getRange(getNode(VRI.loadHigherDim));

  RatioRange = handleDiv(RatioRange, StartRange);

  return std::make_shared<GeometricFlattenedRangedRecurrence>(std::move(StartRange), std::move(RatioRange));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildAffineFlattingRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store) {

  auto StartRange = getRange(getNode(VRI.loadJunction));

  std::shared_ptr<Range> StepRange;

  if (VRI.loadHigherDim) {
    StepRange = getRange(getNode(VRI.loadHigherDim));
  } else {
    StepRange = extractDeltaFromStoreValue(Store->getValueOperand(), VRI.loadJunction, Store);
    if (!StepRange) {
      //fallback valid for affine
      auto NextRange = getRange(getNode(Store->getValueOperand()));
      StepRange = handleSub(NextRange, StartRange);
    }
  }

  return std::make_shared<AffineFlattenedRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildGeometricFlattingRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store) {

  auto StartRange = getRange(getNode(VRI.loadJunction));

  std::shared_ptr<Range> RatioRange;

  if (VRI.loadHigherDim) {
    RatioRange = getRange(getNode(VRI.loadHigherDim));
  } else {
    RatioRange = extractDeltaFromStoreValue(Store->getValueOperand(), VRI.loadJunction, Store);
    if (!RatioRange) {
      //fallback valid for geo
      auto NextRange = getRange(getNode(Store->getValueOperand()));
      RatioRange = handleDiv(NextRange, StartRange);
    }
  }

  return std::make_shared<GeometricFlattenedRangedRecurrence>(std::move(StartRange), std::move(RatioRange));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildLinearFlattingRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store) {
  // Pattern: %res = call @llvm.fmuladd.f32(B, A, Start)
  // Map operands -> Linear(start = arg2, A = arg1, B = arg0)
  auto *Call = dyn_cast<llvm::CallBase>(Store->getValueOperand());
  if (!Call)
    return nullptr;

  auto *CalledF = Call->getCalledFunction();
  if (!CalledF || CalledF->getIntrinsicID() != llvm::Intrinsic::fmuladd)
    return nullptr;

  auto StartRange = getRange(getNode(Call->getArgOperand(2)));
  auto ARng = handleMul(getRange(getNode(Call->getArgOperand(0))), getRange(getNode(Call->getArgOperand(1))));
  auto BRng = getRange(getNode(Call->getArgOperand(2)));

  if (!StartRange) StartRange = Range::Top().clone();
  if (!ARng) ARng = Range::Top().clone();
  if (!BRng) BRng = Range::Top().clone();

  return std::make_shared<LinearRangedRecurrence>(std::move(StartRange), std::move(ARng), std::move(BRng));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildDeltaAffinePHIRecurrence(VRARecurrenceInfo VRI, const llvm::PHINode* phi, VRARecurrenceInfo* InnerVRI) {

  auto InnerAffine = llvm::dyn_cast<AffineRangedRecurrence>(InnerVRI->RR);
  if (!InnerAffine) {
    if (auto InnerFlat = llvm::dyn_cast<AffineFlattenedRangedRecurrence>(InnerVRI->RR)) {
      InnerAffine = std::make_shared<AffineRangedRecurrence>(InnerFlat->getStart(), InnerFlat->getStep());
    } else if (auto InnerCross = llvm::dyn_cast<AffineCrossingRangedRecurrence>(InnerVRI->RR)) {
      InnerAffine = std::make_shared<AffineRangedRecurrence>(InnerCross->getStart(), InnerCross->getStep());
    }
  }

  const Value* op = phi->getIncomingValue(0);
  std::shared_ptr<ValueInfo> op_node = getNode(op);
  std::shared_ptr<Range> StartRange = getRange(op_node);

  const Value* op_1 = phi->getIncomingValue(1);
  std::shared_ptr<ValueInfo> op_node_1 = getNode(op_1);
  std::shared_ptr<Range> OuterIncomingRange = getRange(op_node_1);
  
  std::shared_ptr<Range> InnerComputedRange = nullptr;
  if (InnerAffine && InnerVRI->lastRangeComputedAt > 0) {
    InnerComputedRange = InnerAffine->at(InnerVRI->lastRangeComputedAt);
  }
  if (!InnerComputedRange)
    InnerComputedRange = InnerVRI->lastRange;

  if (OuterIncomingRange) {
    auto InnerRangeUsed = getRange(getNode(InnerVRI->root));
    if (InnerRangeUsed && InnerComputedRange) {
      double dMin = InnerComputedRange->min - InnerRangeUsed->min;
      double dMax = InnerComputedRange->max - InnerRangeUsed->max;
      OuterIncomingRange = std::make_shared<Range>(OuterIncomingRange->min + dMin,
                                                   OuterIncomingRange->max + dMax);
    }
  }

  std::shared_ptr<Range> StepRange = nullptr;
  if (OuterIncomingRange && InnerComputedRange) {
    StepRange = std::make_shared<Range>(OuterIncomingRange->min - InnerComputedRange->min,
                                        OuterIncomingRange->max - InnerComputedRange->max);
  }

  return std::make_shared<AffineDeltaRangedRecurrence>(std::move(StartRange), std::move(StepRange), std::move(InnerAffine), InnerVRI->lastRangeComputedAt);
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildDeltaAffineStoreRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store, VRARecurrenceInfo* InnerVRI) {
  //todo
  return nullptr;
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildDeltaGeometricPHIRecurrence(VRARecurrenceInfo VRI, const llvm::PHINode* phi, VRARecurrenceInfo* InnerVRI) {

  auto InnerGeom = llvm::dyn_cast<GeometricRangedRecurrence>(InnerVRI->RR);
  if (!InnerGeom) {
    if (auto InnerFlat = llvm::dyn_cast<GeometricFlattenedRangedRecurrence>(InnerVRI->RR)) {
      InnerGeom = std::make_shared<GeometricRangedRecurrence>(InnerFlat->getStart(), InnerFlat->getRatio());
    } else if (auto InnerCross = llvm::dyn_cast<GeometricCrossingRangedRecurrence>(InnerVRI->RR)) {
      InnerGeom = std::make_shared<GeometricRangedRecurrence>(InnerCross->getStart(), InnerCross->getRatio());
    }
  }

  const Value* op = phi->getIncomingValue(0);
  std::shared_ptr<ValueInfo> op_node = getNode(op);
  std::shared_ptr<Range> StartRange = getRange(op_node);

  const Value* op_1 = phi->getIncomingValue(1);
  std::shared_ptr<ValueInfo> op_node_1 = getNode(op_1);
  std::shared_ptr<Range> OuterIncomingRange = getRange(op_node_1);
  
  std::shared_ptr<Range> InnerComputedRange = nullptr;
  if (InnerGeom && InnerVRI->lastRangeComputedAt > 0) {
    InnerComputedRange = InnerGeom->at(InnerVRI->lastRangeComputedAt);
  }
  if (!InnerComputedRange)
    InnerComputedRange = InnerVRI->lastRange;

  std::shared_ptr<Range> StepRange = nullptr;

  // Prefer an exact constant multiplier if the delta chain is a single FMul/Mul by a constant.
  if (VRI.chain.size() == 1 && !StepRange) {
    if (const auto *BO = dyn_cast<BinaryOperator>(VRI.chain.front())) {
      if (BO->getOpcode() == Instruction::FMul || BO->getOpcode() == Instruction::Mul) {
        const Value *Op0 = BO->getOperand(0);
        const Value *Op1 = BO->getOperand(1);
        const Value *ConstOp = nullptr;
        if (isa<ConstantFP>(Op0) || isa<ConstantInt>(Op0))
          ConstOp = Op0;
        else if (isa<ConstantFP>(Op1) || isa<ConstantInt>(Op1))
          ConstOp = Op1;

        if (ConstOp) {
          if (const auto *CFP = dyn_cast<ConstantFP>(ConstOp)) {
            StepRange = Range::Point(CFP->getValueAPF()).clone();
          } else if (const auto *CI = dyn_cast<ConstantInt>(ConstOp)) {
            StepRange = Range::Point(llvm::APFloat((double)CI->getSExtValue())).clone();
          } else {
            auto ConstRange = getRange(getNode(ConstOp));
            if (ConstRange)
              StepRange = ConstRange;
          }
        }
      }
    }
  }

  // If the backedge value is exactly the inner recurrence root, the outer loop
  // does not apply any extra operation: force ratio_out = 1.
  if (!StepRange) {
    // Directly extract a constant multiplier from the backedge instruction.
    if (const auto *BO = dyn_cast<BinaryOperator>(op_1)) {
      if (BO->getOpcode() == Instruction::FMul || BO->getOpcode() == Instruction::Mul) {
        const Value *Op0 = BO->getOperand(0);
        const Value *Op1 = BO->getOperand(1);
        const Value *ConstOp = nullptr;
        if (isa<ConstantFP>(Op0) || isa<ConstantInt>(Op0))
          ConstOp = Op0;
        else if (isa<ConstantFP>(Op1) || isa<ConstantInt>(Op1))
          ConstOp = Op1;
        if (ConstOp) {
          if (const auto *CFP = dyn_cast<ConstantFP>(ConstOp)) {
            StepRange = Range::Point(CFP->getValueAPF()).clone();
          } else if (const auto *CI = dyn_cast<ConstantInt>(ConstOp)) {
            StepRange = Range::Point(llvm::APFloat((double)CI->getSExtValue())).clone();
          }
        }
      }
    }
  }

  if (!StepRange) {
    if (VRI.innerRR && op_1 == VRI.innerRR) {
      StepRange = Range::Point(llvm::APFloat(1.0)).clone();
    } else if (VRI.chain.empty()) {
      // No extra ops recorded on the outer latch: ratio_out = 1
      StepRange = Range::Point(llvm::APFloat(1.0)).clone();
    } else if (OuterIncomingRange && InnerComputedRange) {
      // Estimate ratio_out as outer_value / (prev_outer * inner_block).
      if (StartRange) {
        if (auto Den = handleMul(StartRange, InnerComputedRange))
          StepRange = handleDiv(OuterIncomingRange, Den);
      }
      if (!StepRange)
        StepRange = handleDiv(OuterIncomingRange, InnerComputedRange);
    }
  }

  return std::make_shared<GeometricDeltaRangedRecurrence>(std::move(StartRange), std::move(StepRange), std::move(InnerGeom), InnerVRI->lastRangeComputedAt);
}

std::shared_ptr<RangedRecurrence> VRAnalyzer::buildDeltaGeometricStoreRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* store, VRARecurrenceInfo* InnerVRI) {
  return nullptr;
}

// valid when delta index is 1
std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildAffineStoreRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store) {

  auto StartRange = getRange(getNode(VRI.loadJunction));

  std::shared_ptr<Range> StepRange;

  if (VRI.loadHigherDim) {
    StepRange = getRange(getNode(VRI.loadHigherDim));
  } else {
    StepRange = extractDeltaFromStoreValue(Store->getValueOperand(), VRI.loadJunction, Store);
    if (!StepRange) {
      //fallback valid for affine
      auto NextRange = getRange(getNode(Store->getValueOperand()));
      StepRange = handleSub(NextRange, StartRange);
    }
  }

  return std::make_shared<AffineRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildFakeStoreRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store) {

  auto StartRange = getRange(getNode(VRI.loadJunction));
  if (!StartRange) StartRange = Range::Top().clone();

  auto op = Store->getValueOperand();
  auto StepRange = getRange(getNode(op));

  StepRange = StartRange->join(StepRange);

  return std::make_shared<FakeRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::shared_ptr<RangedRecurrence> VRAnalyzer::buildAffinePHIMulAddRecurrence(VRARecurrenceInfo VRI, const llvm::PHINode* PHI) {

  if (!PHI || PHI->getNumIncomingValues() < 2)
    return nullptr;

  auto *Call = dyn_cast<llvm::CallBase>(PHI->getIncomingValue(1));
  if (!Call)
    return nullptr;

  auto *CalledF = Call->getCalledFunction();
  if (!CalledF || CalledF->getIntrinsicID() != llvm::Intrinsic::fmuladd)
    return nullptr;

  auto StartRange = getRange(getNode(PHI->getIncomingValue(0)));
  if (!StartRange)
    StartRange = Range::Top().clone();

  std::shared_ptr<Range> StepRange = nullptr;

  for (unsigned ArgIdx = 0, End = Call->arg_size(); ArgIdx < End; ++ArgIdx) {
    const Value *Arg = Call->getArgOperand(ArgIdx);

    // Skip self-dependence when the accumulator is passed to fmuladd.
    if (Arg == PHI)
      continue;

    auto ArgRange = getRange(getNode(Arg));
    if (!ArgRange)
      ArgRange = Range::Top().clone();

    StepRange = StepRange ? handleMul(StepRange, ArgRange) : ArgRange;
    if (!StepRange)
      StepRange = Range::Top().clone();
  }

  if (!StepRange)
    return nullptr;

  return std::make_shared<AffineFlattenedRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::shared_ptr<RangedRecurrence> VRAnalyzer::buildAffineStoreMulAddRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store) {

  auto *Call = dyn_cast<llvm::CallBase>(Store->getValueOperand());
  if (!Call)
    return nullptr;

  auto *CalledF = Call->getCalledFunction();
  if (!CalledF || CalledF->getIntrinsicID() != llvm::Intrinsic::fmuladd)
    return nullptr;

  auto StartRange = getRange(getNode(VRI.loadJunction));
  if (!StartRange)
    StartRange = Range::Top().clone();

  const Value *StoreBase = getBaseMemoryObject(Store->getPointerOperand());
  std::shared_ptr<Range> StepRange = nullptr;

  for (unsigned ArgIdx = 0, End = Call->arg_size(); ArgIdx < End; ++ArgIdx) {
    const Value *Arg = Call->getArgOperand(ArgIdx);

    // Skip operand if it loads from the same base as the store target.
    if (const auto *LI = dyn_cast<LoadInst>(Arg)) {
      const Value *LoadBase = getBaseMemoryObject(LI->getPointerOperand());
      if (LoadBase && StoreBase && LoadBase == StoreBase)
        continue;
    }

    auto ArgRange = getRange(getNode(Arg));
    if (!ArgRange)
      ArgRange = Range::Top().clone();

    StepRange = StepRange ? handleMul(StepRange, ArgRange) : ArgRange;
    if (!StepRange)
      StepRange = Range::Top().clone();
  }

  if (!StepRange)
    return nullptr;

  return std::make_shared<AffineFlattenedRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::pair<std::shared_ptr<RangedRecurrence>, std::shared_ptr<RangedRecurrence>> VRAnalyzer::buildStoreCrossingAffineRecurrence(VRAAssignationInfo first, VRAAssignationInfo second) {
  //LLVM_DEBUG(tda::log() << "first is "<< first.root << "\nsecond is "<<second.root<<"\n");

  auto FirstStore = dyn_cast<StoreInst>(first.root);
  auto SecondStore = dyn_cast<StoreInst>(second.root);

  auto FirstStartRange = getRange(getNode(FirstStore->getValueOperand()));
  auto SecondStartRange = getRange(getNode(SecondStore->getValueOperand()));

  std::shared_ptr<Range> SecondLoadRange = nullptr;
  if (auto *SecondValLoad = dyn_cast<LoadInst>(SecondStore->getValueOperand())) {
    SecondLoadRange = getRange(getNode(SecondValLoad));
  } else if (auto *SecondValInst = dyn_cast<Instruction>(SecondStore->getValueOperand())) {
    for (const auto &Op : SecondValInst->operands()) {
      if (auto *OpLoad = dyn_cast<LoadInst>(Op)) {
        //LLVM_DEBUG(tda::log() << "break on " << Op << "\n");
        SecondLoadRange = getRange(getNode(OpLoad));
        //LLVM_DEBUG(tda::log() << "range  " << SecondLoadRange->toString() << "\n");
        break;
      }
    }
  }

  auto StepRange = std::make_shared<Range>(FirstStartRange->min - SecondLoadRange->min, FirstStartRange->max - SecondLoadRange->max);

  // LLVM_DEBUG(tda::log() << " first start value op: " << FirstStartRange->toString() << "\n");
  // LLVM_DEBUG(tda::log() << " second start value op: " << SecondStartRange->toString() << "\n");
  // LLVM_DEBUG(tda::log() << " step is: " << StepRange->toString() << "\n");

  return {
    std::make_shared<AffineCrossingRangedRecurrence>(std::move(FirstStartRange), StepRange),
    std::make_shared<AffineCrossingRangedRecurrence>(std::move(SecondStartRange), std::move(StepRange))};
}

std::pair<std::shared_ptr<RangedRecurrence>, std::shared_ptr<RangedRecurrence>> VRAnalyzer::buildStoreCrossingGeometricRecurrence(VRAAssignationInfo first, VRAAssignationInfo second) {
  auto FirstStore = dyn_cast<StoreInst>(first.root);
  auto SecondStore = dyn_cast<StoreInst>(second.root);

  auto FirstStartRange = getRange(getNode(FirstStore->getValueOperand()));
  auto SecondStartRange = getRange(getNode(SecondStore->getValueOperand()));

  if (!FirstStartRange) FirstStartRange = Range::Top().clone();
  if (!SecondStartRange) SecondStartRange = Range::Top().clone();

  std::shared_ptr<Range> SecondLoadRange = nullptr;
  if (auto *SecondValLoad = dyn_cast<LoadInst>(SecondStore->getValueOperand())) {
    SecondLoadRange = getRange(getNode(SecondValLoad));
  } else if (auto *SecondValInst = dyn_cast<Instruction>(SecondStore->getValueOperand())) {
    for (const auto &Op : SecondValInst->operands()) {
      if (auto *OpLoad = dyn_cast<LoadInst>(Op)) {
        SecondLoadRange = getRange(getNode(OpLoad));
        break;
      }
    }
  }

  if (!SecondLoadRange) SecondLoadRange = Range::Top().clone();

  auto RatioRange = handleDiv(FirstStartRange, SecondLoadRange);
  if (!RatioRange) RatioRange = Range::Top().clone();

  return {
    std::make_shared<GeometricCrossingRangedRecurrence>(std::move(FirstStartRange), RatioRange),
    std::make_shared<GeometricCrossingRangedRecurrence>(std::move(SecondStartRange), std::move(RatioRange))};
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildInitRecurrence(std::shared_ptr<Range> LastStoredRange, const llvm::StoreInst* Store) {

  auto op = Store->getValueOperand();
  auto StartRange = getRange(getNode(op));

  if (LastStoredRange)
    StartRange = LastStoredRange->join(StartRange);
  
  return std::make_shared<InitRangedRecurrence>(std::move(StartRange));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildGeometricPHIRecurrence(const llvm::PHINode *phi) {

  const Value* op = phi->getIncomingValue(0);
  std::shared_ptr<ValueInfo> op_node = getNode(op);
  std::shared_ptr<Range> StartRange = getRange(op_node);

  const Value* op_1 = phi->getIncomingValue(1);
  std::shared_ptr<ValueInfo> op_node_1 = getNode(op_1);
  std::shared_ptr<Range> StepRatio = getRange(op_node_1);
  
  StepRatio = handleDiv(StepRatio, StartRange);

  LLVM_DEBUG(tda::log() << "recognized geometric(start= " << (StartRange ? StartRange->toString() : "(none)") << ", ratio= " << StepRatio->toString() << ")\n\n");
  return std::make_shared<GeometricRangedRecurrence>(std::move(StartRange), std::move(StepRatio));
}

std::shared_ptr<taffo::RangedRecurrence> VRAnalyzer::buildGeometricStoreRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* Store) {

  auto StartRange = getRange(getNode(VRI.loadJunction));

  auto op = Store->getValueOperand();
  auto StepRatio = getRange(getNode(op));
  
  StepRatio = handleDiv(StepRatio, StartRange);

  LLVM_DEBUG(tda::log() << "recognized geometric(start= " << (StartRange ? StartRange->toString() : "(none)") << ", ratio= " << StepRatio->toString() << ")\n\n");
  return std::make_shared<GeometricRangedRecurrence>(std::move(StartRange), std::move(StepRatio));
}

std::shared_ptr<RangedRecurrence> VRAnalyzer::buildUnknownRecurrence(const llvm::Value *V) {

  auto StartRange = getRange(getNode(V));
  auto StepRange = Range::Top().clone();

  LLVM_DEBUG(tda::log() << "recognized unknown(start= " << (StartRange ? StartRange->toString() : "(none)") << ", step= " << StepRange->toString() << ")\n\n");
  return std::make_shared<AffineRangedRecurrence>(std::move(StartRange), std::move(StepRange));
}

std::shared_ptr<RangedRecurrence> VRAnalyzer::buildLinearRecurrence(VRARecurrenceInfo VRI, const llvm::StoreInst* store) {
  // Start: value previously stored in the same array (arr[i-1]).
  auto StartRange = getRange(getNode(VRI.loadJunction));
  if (!StartRange)
    StartRange = Range::Top().clone();

  // Expected pattern: add (mul loadA loadArrPrev), loadB
  const Value *StoredVal = store->getValueOperand();
  const auto *Add = dyn_cast<BinaryOperator>(StoredVal);
  std::shared_ptr<Range> ARng = nullptr; // multiplicative term
  std::shared_ptr<Range> BRng = nullptr; // additive term

  if (Add && Add->getOpcode() == Instruction::FAdd) {
    const Value *Op0 = Add->getOperand(0);
    const Value *Op1 = Add->getOperand(1);

    // Identify mul and the other operand
    const BinaryOperator *Mul = nullptr;
    const Value *Other = nullptr;
    if ((Mul = dyn_cast<BinaryOperator>(Op0)) && Mul->getOpcode() == Instruction::FMul) {
      Other = Op1;
    } else if ((Mul = dyn_cast<BinaryOperator>(Op1)) && Mul->getOpcode() == Instruction::FMul) {
      Other = Op0;
    }

    // Mul operands: A load and previous array element load
    if (Mul) {
      const Value *MulOp0 = Mul->getOperand(0);
      const Value *MulOp1 = Mul->getOperand(1);

      // Choose A as operand whose base mem differs from store base
      const Value *StoreBase = getBaseMemoryObject(store->getPointerOperand());
      const Value *MulBase0 = getBaseMemoryObject(MulOp0);
      const Value *MulBase1 = getBaseMemoryObject(MulOp1);

      const Value *AOp = nullptr;
      if (MulBase0 && MulBase0 != StoreBase)
        AOp = MulOp0;
      else if (MulBase1 && MulBase1 != StoreBase)
        AOp = MulOp1;

      if (!AOp)
        AOp = MulOp0;

      ARng = getRange(getNode(AOp));
    }

    // B is the operand of add whose base differs from store base
    if (Other) {
      const Value *StoreBase = getBaseMemoryObject(store->getPointerOperand());
      const Value *OtherBase = getBaseMemoryObject(Other);
      if (!StoreBase || (OtherBase && OtherBase != StoreBase))
        BRng = getRange(getNode(Other));
    }
  }

  if (!ARng) ARng = Range::Top().clone();
  if (!BRng) BRng = Range::Top().clone();

  return std::make_shared<LinearRangedRecurrence>(std::move(StartRange), std::move(ARng), std::move(BRng));
}


std::shared_ptr<Range> VRAnalyzer::getRRJoinedRange(RangedRecurrence* RR, u_int64_t TC) {

  auto rangeAtZero = RR->at(0);
  auto rangeAtTC = RR->at(TC);

  if (!rangeAtZero && !rangeAtTC) {
    return std::make_shared<Range>(Range::Top());
  }

  if (!rangeAtZero || rangeAtZero == Range::Top().clone()) {
    return rangeAtTC;
  }

  if (!rangeAtTC) {
    return rangeAtZero;
  }

  return std::make_shared<Range>(rangeAtZero->join(*rangeAtTC));
}
