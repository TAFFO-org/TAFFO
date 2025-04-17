#include "MemSSAUtils.hpp"
#include "RangeOperations.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"
#include "VRAnalyzer.hpp"

#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Debug.h>

using namespace llvm;
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
  return std::make_shared<VRAnalyzer>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()), CI);
}

std::shared_ptr<AnalysisStore> VRAnalyzer::newFunctionStore(CodeInterpreter& CI) {
  return std::make_shared<VRAFunctionStore>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()));
}

std::shared_ptr<CodeAnalyzer> VRAnalyzer::clone() { return std::make_shared<VRAnalyzer>(*this); }

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
    const std::shared_ptr<Range> info = fetchRange(op);
    const std::shared_ptr<Range> res = handleCastInstruction(info, OpCode, i.getType());
    saveValueRange(&i, res);

    LLVM_DEBUG(if (!info) Logger->logInfo("operand range is null"));
    LLVM_DEBUG(logRangeln(&i));
  }
  else if (Instruction::isBinaryOp(OpCode)) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const Value* op1 = i.getOperand(0);
    const Value* op2 = i.getOperand(1);
    const std::shared_ptr<Range> info1 = fetchRange(op1);
    const std::shared_ptr<Range> info2 = fetchRange(op2);
    const std::shared_ptr<Range> res = handleBinaryInstruction(info1, info2, OpCode);
    saveValueRange(&i, res);

    LLVM_DEBUG(if (!info1) Logger->logInfo("first range is null"));
    LLVM_DEBUG(if (!info2) Logger->logInfo("second range is null"));
    LLVM_DEBUG(logRangeln(&i));
  }
  else if (OpCode == Instruction::FNeg) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const Value* op1 = i.getOperand(0);
    const std::shared_ptr<Range> info1 = fetchRange(op1);
    const auto res = handleUnaryInstruction(info1, OpCode);
    saveValueRange(&i, res);

    LLVM_DEBUG(if (!info1) Logger->logInfo("operand range is null"));
    LLVM_DEBUG(logRangeln(&i));
  }
  else {
    switch (OpCode) {
    // memory operations
    case Instruction::Alloca:
      handleAllocaInstr(I);
      break;
    case Instruction::Load:
      handleLoadInstr(&i);
      break;
    case Instruction::Store:
      handleStoreInstr(&i);
      break;
    case Instruction::GetElementPtr:
      handleGEPInstr(&i);
      break;
    case Instruction::BitCast:
      handleBitCastInstr(I);
      break;
    case Instruction::Fence:
      LLVM_DEBUG(Logger->logErrorln("Handling of Fence not supported yet"));
      break; // TODO implement
    case Instruction::AtomicCmpXchg:
      LLVM_DEBUG(Logger->logErrorln("Handling of AtomicCmpXchg not supported yet"));
      break; // TODO implement
    case Instruction::AtomicRMW:
      LLVM_DEBUG(Logger->logErrorln("Handling of AtomicRMW not supported yet"));
      break; // TODO implement

      // other operations
    case Instruction::Ret:
      handleReturn(I);
      break;
    case Instruction::Br:
      // do nothing
      break;
    case Instruction::ICmp:
    case Instruction::FCmp:
      handleCmpInstr(&i);
      break;
    case Instruction::PHI:
      handlePhiNode(&i);
      break;
    case Instruction::Select:
      handleSelect(&i);
      break;
    case Instruction::UserOp1:        // TODO implement
    case Instruction::UserOp2:        // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of UserOp not supported yet"));
      break;
    case Instruction::VAArg:          // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of VAArg not supported yet"));
      break;
    case Instruction::ExtractElement: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ExtractElement not supported yet"));
      break;
    case Instruction::InsertElement:  // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of InsertElement not supported yet"));
      break;
    case Instruction::ShuffleVector:  // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ShuffleVector not supported yet"));
      break;
    case Instruction::ExtractValue:   // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ExtractValue not supported yet"));
      break;
    case Instruction::InsertValue:    // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of InsertValue not supported yet"));
      break;
    case Instruction::LandingPad:     // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of LandingPad not supported yet"));
      break;
    default:
      LLVM_DEBUG(Logger->logErrorln("unhandled instruction " + std::to_string(OpCode)));
      break;
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

  LLVM_DEBUG(Logger->lineHead(); dbgs() << "Loading argument ranges: ");
  // fetch ranges of arguments
  std::list<std::shared_ptr<ValueInfo>> ArgRanges;
  for (Value* Arg : CB->args()) {
    ArgRanges.push_back(getNode(Arg));

    LLVM_DEBUG(dbgs() << VRALogger::toString(fetchRangeNode(Arg)) << ", ");
  }
  LLVM_DEBUG(dbgs() << "\n");

  std::shared_ptr<VRAFunctionStore> FStore = std::static_ptr_cast<VRAFunctionStore>(FunctionStore);
  FStore->setArgumentRanges(*CB->getCalledFunction(), ArgRanges);
}

void VRAnalyzer::returnFromCall(Instruction* I, std::shared_ptr<AnalysisStore> FunctionStore) {
  CallBase* CB = cast<CallBase>(I);
  assert(!CB->isIndirectCall());

  LLVM_DEBUG(Logger->logInstruction(I); Logger->logInfo("returning from call"));

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
    case Intrinsic::memcpy:
      handleMemCpyIntrinsics(CB);
      break;
    default:
      LLVM_DEBUG(Logger->logInfoln("skipping intrinsic " + std::string(FunctionName)));
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

    std::shared_ptr<VRAFunctionStore> FStore = std::static_ptr_cast<VRAFunctionStore>(CodeInt.getFunctionStore());
    FStore->setRetVal(range);

    LLVM_DEBUG(Logger->logRangeln(range));
  }
  else {
    LLVM_DEBUG(Logger->logInfoln("void return."));
  }
}

void VRAnalyzer::handleAllocaInstr(Instruction* I) {
  AllocaInst* allocaInst = cast<AllocaInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const auto inputValueInfo = getGlobalStore()->getUserInput(I);
  auto allocatedType = TaffoInfo::getInstance().getOrCreateTransparentType(*allocaInst);
  if (auto structType = std::dynamic_ptr_cast<TransparentStructType>(allocatedType)) {
    if (inputValueInfo && std::isa_ptr<StructInfo>(inputValueInfo))
      DerivedRanges[I] = inputValueInfo->clone();
    else
      DerivedRanges[I] = ValueInfoFactory::create(structType);
    LLVM_DEBUG(Logger->logInfoln("struct"));
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
  std::shared_ptr<ValueInfo> ValueNode = getNode(ValueParam);

  if (!ValueNode && !ValueParam->getType()->isPointerTy())
    ValueNode = fetchRangeNode(I);

  storeNode(AddressNode, ValueNode);

  LLVM_DEBUG(Logger->logRangeln(ValueNode));
}

void VRAnalyzer::handleLoadInstr(Instruction* I) {
  LoadInst* Load = cast<LoadInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const Value* PointerOp = Load->getPointerOperand();

  std::shared_ptr<ValueInfo> Loaded = loadNode(getNode(PointerOp));

  if (std::shared_ptr<ScalarInfo> Scalar = std::dynamic_ptr_cast_or_null<ScalarInfo>(Loaded)) {
    auto& FAM =
      CodeInt.getMAM().getResult<FunctionAnalysisManagerModuleProxy>(*I->getFunction()->getParent()).getManager();
    auto* SSARes = &(FAM.getResult<MemorySSAAnalysis>(*I->getFunction()));
    MemorySSA& memssa = SSARes->getMSSA();
    MemSSAUtils memssa_utils(memssa);
    SmallVectorImpl<Value*>& def_vals = memssa_utils.getDefiningValues(Load);

    Type* load_ty = getUnwrappedType(Load);
    std::shared_ptr<Range> res = Scalar->range;
    for (Value* dval : def_vals)
      if (dval && load_ty->canLosslesslyBitCastTo(getUnwrappedType(dval)))
        res = getUnionRange(res, fetchRange(dval));
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
  if (!extractGEPOffset(
        gepInst->getSourceElementType(), iterator_range(gepInst->idx_begin(), gepInst->idx_end()), Offset)) {
    return;
  }
  Node = std::make_shared<GEPInfo>(getNode(gepInst->getPointerOperand()), Offset);
  setNode(I, Node);
}

void VRAnalyzer::handleBitCastInstr(Instruction* I) {
  LLVM_DEBUG(Logger->logInstruction(I));
  if (std::shared_ptr<ValueInfo> Node = getNode(I->getOperand(0U))) {
    bool InputIsStruct = getUnwrappedType(I->getOperand(0U))->isStructTy();
    bool OutputIsStruct = getUnwrappedType(I)->isStructTy();
    if (!InputIsStruct && !OutputIsStruct) {
      setNode(I, Node);
      LLVM_DEBUG(Logger->logRangeln(Node));
    }
    else {
      LLVM_DEBUG(Logger->logInfoln("oh shit -> no node"));
      LLVM_DEBUG(
        dbgs()
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

  return nullptr;
}

std::shared_ptr<ValueInfoWithRange> VRAnalyzer::fetchRangeNode(const Value* v) {
  if (const std::shared_ptr<ValueInfoWithRange> Derived = VRAStore::fetchRangeNode(v)) {
    if (std::isa_ptr<StructInfo>(Derived)) {
      if (auto InputRange = getGlobalStore()->getUserInput(v)) {
        // fill null input_range fields with corresponding derived fields
        return fillRangeHoles(Derived, InputRange->clone<ValueInfoWithRange>());
      }
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
    std::shared_ptr<VRAFunctionStore> FStore = std::static_ptr_cast<VRAFunctionStore>(CodeInt.getFunctionStore());
    FStore->setNode(V, Node);
    return;
  }

  VRAStore::setNode(V, Node);
}

void VRAnalyzer::logRangeln(const Value* v) {
  LLVM_DEBUG(if (getGlobalStore()->getUserInput(v)) dbgs() << "(possibly from metadata) ");
  LLVM_DEBUG(Logger->logRangeln(fetchRangeNode(v)));
}
