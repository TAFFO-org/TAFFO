#include "VRAnalyzer.hpp"

#include "MemSSAUtils.hpp"
#include "RangeOperations.hpp"
#include "TypeUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace taffo;

#define DEBUG_TYPE "taffo-vra"

void VRAnalyzer::convexMerge(const AnalysisStore &Other)
{
  // Since llvm::dyn_cast<T>() does not do cross-casting, we must do this:
  if (llvm::isa<VRAnalyzer>(Other)) {
    VRAStore::convexMerge(llvm::cast<VRAStore>(llvm::cast<VRAnalyzer>(Other)));
  } else if (llvm::isa<VRAGlobalStore>(Other)) {
    VRAStore::convexMerge(llvm::cast<VRAStore>(llvm::cast<VRAGlobalStore>(Other)));
  } else {
    VRAStore::convexMerge(llvm::cast<VRAStore>(llvm::cast<VRAFunctionStore>(Other)));
  }
}

std::shared_ptr<CodeAnalyzer>
VRAnalyzer::newCodeAnalyzer(CodeInterpreter &CI)
{
  return std::make_shared<VRAnalyzer>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()), CI);
}

std::shared_ptr<AnalysisStore>
VRAnalyzer::newFunctionStore(CodeInterpreter &CI)
{
  return std::make_shared<VRAFunctionStore>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()));
}

std::shared_ptr<CodeAnalyzer>
VRAnalyzer::clone()
{
  return std::make_shared<VRAnalyzer>(*this);
}

void VRAnalyzer::analyzeInstruction(llvm::Instruction *I)
{
  assert(I);
  Instruction &i = *I;
  const unsigned OpCode = i.getOpcode();
  if (OpCode == Instruction::Call || OpCode == Instruction::Invoke) {
    handleSpecialCall(&i);
  } else if (Instruction::isCast(OpCode) && OpCode != llvm::Instruction::BitCast) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const llvm::Value *op = i.getOperand(0);
    const range_ptr_t info = fetchRange(op);
    const range_ptr_t res = handleCastInstruction(info, OpCode, i.getType());
    saveValueRange(&i, res);

    LLVM_DEBUG(if (!info) Logger->logInfo("operand range is null"));
    LLVM_DEBUG(logRangeln(&i));
  } else if (Instruction::isBinaryOp(OpCode)) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const llvm::Value *op1 = i.getOperand(0);
    const llvm::Value *op2 = i.getOperand(1);
    const range_ptr_t info1 = fetchRange(op1);
    const range_ptr_t info2 = fetchRange(op2);
    const range_ptr_t res = handleBinaryInstruction(info1, info2, OpCode);
    saveValueRange(&i, res);

    LLVM_DEBUG(if (!info1) Logger->logInfo("first range is null"));
    LLVM_DEBUG(if (!info2) Logger->logInfo("second range is null"));
    LLVM_DEBUG(logRangeln(&i));
  } else if (OpCode == llvm::Instruction::FNeg) {
    LLVM_DEBUG(Logger->logInstruction(&i));
    const llvm::Value *op1 = i.getOperand(0);
    const range_ptr_t info1 = fetchRange(op1);
    const auto res = handleUnaryInstruction(info1, OpCode);
    saveValueRange(&i, res);

    LLVM_DEBUG(if (!info1) Logger->logInfo("operand range is null"));
    LLVM_DEBUG(logRangeln(&i));
  } else {
    switch (OpCode) {
    // memory operations
    case llvm::Instruction::Alloca:
      handleAllocaInstr(I);
      break;
    case llvm::Instruction::Load:
      handleLoadInstr(&i);
      break;
    case llvm::Instruction::Store:
      handleStoreInstr(&i);
      break;
    case llvm::Instruction::GetElementPtr:
      handleGEPInstr(&i);
      break;
    case llvm::Instruction::BitCast:
      handleBitCastInstr(I);
      break;
    case llvm::Instruction::Fence:
      LLVM_DEBUG(Logger->logErrorln("Handling of Fence not supported yet"));
      break; // TODO implement
    case llvm::Instruction::AtomicCmpXchg:
      LLVM_DEBUG(Logger->logErrorln("Handling of AtomicCmpXchg not supported yet"));
      break; // TODO implement
    case llvm::Instruction::AtomicRMW:
      LLVM_DEBUG(Logger->logErrorln("Handling of AtomicRMW not supported yet"));
      break; // TODO implement

      // other operations
    case llvm::Instruction::Ret:
      handleReturn(I);
      break;
    case llvm::Instruction::Br:
      // do nothing
      break;
    case llvm::Instruction::ICmp:
    case llvm::Instruction::FCmp:
      handleCmpInstr(&i);
      break;
    case llvm::Instruction::PHI:
      handlePhiNode(&i);
      break;
    case llvm::Instruction::Select:
      handleSelect(&i);
      break;
    case llvm::Instruction::UserOp1: // TODO implement
    case llvm::Instruction::UserOp2: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of UserOp not supported yet"));
      break;
    case llvm::Instruction::VAArg: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of VAArg not supported yet"));
      break;
    case llvm::Instruction::ExtractElement: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ExtractElement not supported yet"));
      break;
    case llvm::Instruction::InsertElement: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of InsertElement not supported yet"));
      break;
    case llvm::Instruction::ShuffleVector: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ShuffleVector not supported yet"));
      break;
    case llvm::Instruction::ExtractValue: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of ExtractValue not supported yet"));
      break;
    case llvm::Instruction::InsertValue: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of InsertValue not supported yet"));
      break;
    case llvm::Instruction::LandingPad: // TODO implement
      LLVM_DEBUG(Logger->logErrorln("Handling of LandingPad not supported yet"));
      break;
    default:
      LLVM_DEBUG(Logger->logErrorln("unhandled instruction " + std::to_string(OpCode)));
      break;
    }
  } // end else
}

void VRAnalyzer::setPathLocalInfo(std::shared_ptr<CodeAnalyzer> SuccAnalyzer,
                                  llvm::Instruction *TermInstr, unsigned SuccIdx)
{
  // TODO extract more specific ranges from cmp
}

bool VRAnalyzer::requiresInterpretation(llvm::Instruction *I) const
{
  assert(I);
  if (llvm::CallBase *CB = llvm::dyn_cast<llvm::CallBase>(I)) {
    if (!CB->isIndirectCall()) {
      llvm::Function *Called = CB->getCalledFunction();
      return Called && !(Called->isIntrinsic() 
          || isMathCallInstruction(Called->getName().str()) 
          || isMallocLike(Called) 
          || Called->empty() // function prototypes
        );
    }
    return true;
  }
  // I is not a call.
  return false;
}

void VRAnalyzer::prepareForCall(llvm::Instruction *I,
                                std::shared_ptr<AnalysisStore> FunctionStore)
{
  llvm::CallBase *CB = llvm::cast<llvm::CallBase>(I);
  assert(!CB->isIndirectCall());

  LLVM_DEBUG(Logger->logInstruction(I));
  LLVM_DEBUG(Logger->logInfoln("preparing for function interpretation..."));

  LLVM_DEBUG(Logger->lineHead(); llvm::dbgs() << "Loading argument ranges: ");
  // fetch ranges of arguments
  std::list<NodePtrT> ArgRanges;
  for (Value *Arg : CB->args()) {
    ArgRanges.push_back(getNode(Arg));

    LLVM_DEBUG(llvm::dbgs() << VRALogger::toString(fetchRangeNode(Arg)) << ", ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");

  std::shared_ptr<VRAFunctionStore> FStore =
      std::static_ptr_cast<VRAFunctionStore>(FunctionStore);
  FStore->setArgumentRanges(*CB->getCalledFunction(), ArgRanges);
}

void VRAnalyzer::returnFromCall(llvm::Instruction *I,
                                std::shared_ptr<AnalysisStore> FunctionStore)
{
  llvm::CallBase *CB = llvm::cast<llvm::CallBase>(I);
  assert(!CB->isIndirectCall());

  LLVM_DEBUG(Logger->logInstruction(I); Logger->logInfo("returning from call"));

  std::shared_ptr<VRAFunctionStore> FStore =
      std::static_ptr_cast<VRAFunctionStore>(FunctionStore);
  NodePtrT Ret = FStore->getRetVal();
  if (!Ret) {
    LLVM_DEBUG(Logger->logInfoln("function returns nothing"));
  } else if (RangeNodePtrT RetRange = std::dynamic_ptr_cast_or_null<VRARangeNode>(Ret)) {
    saveValueRange(I, RetRange);
    LLVM_DEBUG(logRangeln(I));
  } else {
    setNode(I, Ret);
    LLVM_DEBUG(Logger->logRangeln(Ret));
  }
}


////////////////////////////////////////////////////////////////////////////////
// Instruction Handlers
////////////////////////////////////////////////////////////////////////////////

void VRAnalyzer::handleSpecialCall(const llvm::Instruction *I)
{
  const CallBase *CB = llvm::cast<CallBase>(I);
  LLVM_DEBUG(Logger->logInstruction(I));

  // fetch function name
  llvm::Function *Callee = CB->getCalledFunction();
  if (Callee == nullptr) {
    LLVM_DEBUG(Logger->logInfo("indirect calls not supported"));
    return;
  }

  // check if it's an OMP library function and handle it if so
  if (detectAndHandleLibOMPCall(CB)) 
    return;

  const llvm::StringRef FunctionName = Callee->getName();
  if (isMathCallInstruction(FunctionName.str())) {
    // fetch ranges of arguments
    std::list<range_ptr_t> ArgScalarRanges;
    for (Value *Arg : CB->args()) {
      ArgScalarRanges.push_back(fetchRange(Arg));
    }
    range_ptr_t Res = handleMathCallInstruction(ArgScalarRanges, FunctionName.str());
    saveValueRange(I, Res);
    LLVM_DEBUG(Logger->logInfo("whitelisted"));
    LLVM_DEBUG(Logger->logRangeln(Res));
  } else if (isMallocLike(Callee)) {
    handleMallocCall(CB);
  } else if (Callee->isIntrinsic()) {
    const auto IntrinsicsID = Callee->getIntrinsicID();
    switch (IntrinsicsID) {
    case llvm::Intrinsic::memcpy:
      handleMemCpyIntrinsics(CB);
      break;
    default:
      LLVM_DEBUG(Logger->logInfoln("skipping intrinsic " + std::string(FunctionName)));
    }
  } else {
    LLVM_DEBUG(Logger->logInfo("unsupported call"));
  }
}

void VRAnalyzer::handleMemCpyIntrinsics(const llvm::Instruction *memcpy)
{
  assert(isa<CallInst>(memcpy) || isa<InvokeInst>(memcpy));
  LLVM_DEBUG(Logger->logInfo("llvm.memcpy"));
  const BitCastInst *dest_bitcast =
      dyn_cast<BitCastInst>(memcpy->getOperand(0U));
  const BitCastInst *src_bitcast =
      dyn_cast<BitCastInst>(memcpy->getOperand(1U));
  if (!(dest_bitcast && src_bitcast)) {
    LLVM_DEBUG(Logger->logInfo("operand is not bitcast, aborting"));
    return;
  }
  const Value *dest = dest_bitcast->getOperand(0U);
  const Value *src = src_bitcast->getOperand(0U);

  const NodePtrT src_node = loadNode(getNode(src));
  storeNode(getNode(dest), src_node);
  LLVM_DEBUG(Logger->logRangeln(fetchRangeNode(src)));
}

bool VRAnalyzer::isMallocLike(const llvm::Function *F) const
{
  const llvm::StringRef FName = F->getName();
  // TODO make sure this works in other platforms
  return FName == "malloc" || FName == "calloc" || FName == "_Znwm" || FName == "_Znam";
}

bool VRAnalyzer::isCallocLike(const llvm::Function *F) const
{
  const llvm::StringRef FName = F->getName();
  // TODO make sure this works in other platforms
  return FName == "calloc";
}

void VRAnalyzer::handleMallocCall(const llvm::CallBase *CB)
{
  LLVM_DEBUG(Logger->logInfo("malloc-like"));
  const llvm::Type *AllocatedType = nullptr;
  for (const llvm::Value *User : CB->users()) {
    if (const llvm::BitCastInst *BCI = llvm::dyn_cast<llvm::BitCastInst>(User)) {
      AllocatedType = BCI->getDestTy()->getPointerElementType();
      break;
    }
  }

  const RangeNodePtrT InputRange = getGlobalStore()->getUserInput(CB);
  if (AllocatedType && AllocatedType->isStructTy()) {
    if (InputRange && std::isa_ptr<VRAStructNode>(InputRange)) {
      DerivedRanges[CB] = InputRange;
    } else {
      DerivedRanges[CB] = std::make_shared<VRAStructNode>();
    }
    LLVM_DEBUG(Logger->logInfoln("struct"));
  } else {
    if (!(AllocatedType && AllocatedType->isPointerTy())) {
      if (InputRange && std::isa_ptr<VRAScalarNode>(InputRange)) {
        DerivedRanges[CB] = std::make_shared<VRAPtrNode>(InputRange);
      } else if (isCallocLike(CB->getCalledFunction())) {
        DerivedRanges[CB] =
            std::make_shared<VRAPtrNode>(std::make_shared<VRAScalarNode>(make_range(0, 0)));
      } else {
        DerivedRanges[CB] = std::make_shared<VRAPtrNode>();
      }
    } else {
      DerivedRanges[CB] = std::make_shared<VRAPtrNode>();
    }
    LLVM_DEBUG(Logger->logInfoln("pointer"));
  }
}

bool VRAnalyzer::detectAndHandleLibOMPCall(const llvm::CallBase *CB)
{
  llvm::Function *F = CB->getCalledFunction();
  if (F->getName() == "__kmpc_for_static_init_4") {
    Value *VPLower = CB->getArgOperand(4U);
    Value *VPUpper = CB->getArgOperand(5U);
    range_ptr_t PLowerRange = fetchRange(VPLower);
    range_ptr_t PUpperRange = fetchRange(VPUpper);
    if (!PLowerRange || !PUpperRange) {
      LLVM_DEBUG(Logger->logInfoln("ranges of plower/pupper unknown, doing nothing"));
      return true;
    }
    range_ptr_t Merge = getUnionRange(PLowerRange, PUpperRange);
    saveValueRange(VPLower, Merge);
    saveValueRange(VPUpper, Merge);
    LLVM_DEBUG(Logger->logRange(Merge));
    LLVM_DEBUG(Logger->logInfoln(" set to plower, pupper nodes"));
    return true;
  }
  return false;
}

void VRAnalyzer::handleReturn(const llvm::Instruction *ret)
{
  const llvm::ReturnInst *ret_i = cast<llvm::ReturnInst>(ret);
  LLVM_DEBUG(Logger->logInstruction(ret));
  if (const llvm::Value *ret_val = ret_i->getReturnValue()) {
    NodePtrT range = getNode(ret_val);

    std::shared_ptr<VRAFunctionStore> FStore =
        std::static_ptr_cast<VRAFunctionStore>(CodeInt.getFunctionStore());
    FStore->setRetVal(range);

    LLVM_DEBUG(Logger->logRangeln(range));
  } else {
    LLVM_DEBUG(Logger->logInfoln("void return."));
  }
}

void VRAnalyzer::handleAllocaInstr(const llvm::Instruction *I)
{
  const llvm::AllocaInst *AI = llvm::cast<AllocaInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const RangeNodePtrT InputRange = getGlobalStore()->getUserInput(I);
  if (AI->getAllocatedType()->isStructTy()) {
    if (InputRange && std::isa_ptr<VRAStructNode>(InputRange)) {
      DerivedRanges[I] = InputRange;
    } else {
      DerivedRanges[I] = std::make_shared<VRAStructNode>();
    }
    LLVM_DEBUG(Logger->logInfoln("struct"));
  } else {
    if (InputRange && std::isa_ptr<VRAScalarNode>(InputRange)) {
      DerivedRanges[I] = std::make_shared<VRAPtrNode>(InputRange);
    } else {
      DerivedRanges[I] = std::make_shared<VRAPtrNode>();
    }
    LLVM_DEBUG(Logger->logInfoln("pointer"));
  }
}

void VRAnalyzer::handleStoreInstr(const llvm::Instruction *I)
{
  const llvm::StoreInst *Store = llvm::cast<llvm::StoreInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const llvm::Value *AddressParam = Store->getPointerOperand();
  const llvm::Value *ValueParam = Store->getValueOperand();

  if (llvm::isa<llvm::ConstantPointerNull>(ValueParam))
    return;

  NodePtrT AddressNode = getNode(AddressParam);
  NodePtrT ValueNode = getNode(ValueParam);

  if (!ValueNode && !ValueParam->getType()->isPointerTy()) {
    ValueNode = fetchRangeNode(I);
  }

  storeNode(AddressNode, ValueNode);

  LLVM_DEBUG(Logger->logRangeln(ValueNode));
}

void VRAnalyzer::handleLoadInstr(llvm::Instruction *I)
{
  llvm::LoadInst *Load = llvm::cast<llvm::LoadInst>(I);
  LLVM_DEBUG(Logger->logInstruction(I));
  const llvm::Value *PointerOp = Load->getPointerOperand();

  NodePtrT Loaded = loadNode(getNode(PointerOp));

  if (std::shared_ptr<VRAScalarNode> Scalar =
          std::dynamic_ptr_cast_or_null<VRAScalarNode>(Loaded)) {
    auto &FAM = CodeInt.getMAM().getResult<FunctionAnalysisManagerModuleProxy>(*I->getFunction()->getParent()).getManager();
    auto *SSARes = &(FAM.getResult<MemorySSAAnalysis>(*I->getFunction()));
    MemorySSA &memssa = SSARes->getMSSA();
    MemSSAUtils memssa_utils(memssa);
    SmallVectorImpl<Value *> &def_vals = memssa_utils.getDefiningValues(Load);

    Type *load_ty = fullyUnwrapPointerOrArrayType(Load->getType());
    range_ptr_t res = Scalar->getRange();
    for (Value *dval : def_vals) {
      if (dval &&
          load_ty->canLosslesslyBitCastTo(fullyUnwrapPointerOrArrayType(dval->getType())))
        res = getUnionRange(res, fetchRange(dval));
    }
    saveValueRange(I, res);
    LLVM_DEBUG(Logger->logRangeln(res));
  } else if (Loaded) {
    setNode(I, Loaded);
    LLVM_DEBUG(Logger->logInfoln("pointer load"));
  } else {
    LLVM_DEBUG(Logger->logInfoln("unable to retrieve loaded value"));
  }
}

void VRAnalyzer::handleGEPInstr(const llvm::Instruction *I)
{
  const llvm::GetElementPtrInst *Gep = llvm::cast<llvm::GetElementPtrInst>(I);
  LLVM_DEBUG(Logger->logInstruction(Gep));

  NodePtrT Node = getNode(Gep);
  if (Node) {
    LLVM_DEBUG(Logger->logInfoln("has node"));
    return;
  }
  llvm::SmallVector<unsigned, 1U> Offset;
  if (!extractGEPOffset(Gep->getSourceElementType(),
                        iterator_range<User::const_op_iterator>(Gep->idx_begin(),
                                                                Gep->idx_end()),
                        Offset)) {
    return;
  }
  Node = std::make_shared<VRAGEPNode>(getNode(Gep->getPointerOperand()), Offset);
  setNode(I, Node);
}

void VRAnalyzer::handleBitCastInstr(const llvm::Instruction *I)
{
  LLVM_DEBUG(Logger->logInstruction(I));
  if (NodePtrT Node = getNode(I->getOperand(0U))) {
    llvm::Type *InputT = I->getOperand(0U)->getType();
    llvm::Type *OutputT = I->getType();
    bool InputIsStruct = fullyUnwrapPointerOrArrayType(InputT)->isStructTy();
    bool OutputIsStruct = fullyUnwrapPointerOrArrayType(OutputT)->isStructTy();
    if (!InputIsStruct && !OutputIsStruct) {
      setNode(I, Node);
      LLVM_DEBUG(Logger->logRangeln(Node));
    } else {
      LLVM_DEBUG(Logger->logInfoln("oh shit -> no node"));
      LLVM_DEBUG(dbgs() << "This instruction is converting to/from a struct type. Ignoring to avoid generating invalid metadata\n");
    }
  } else {
    LLVM_DEBUG(Logger->logInfoln("no node"));
  }
}

void VRAnalyzer::handleCmpInstr(const llvm::Instruction *cmp)
{
  const llvm::CmpInst *cmp_i = llvm::cast<llvm::CmpInst>(cmp);
  LLVM_DEBUG(Logger->logInstruction(cmp));
  const llvm::CmpInst::Predicate pred = cmp_i->getPredicate();
  std::list<range_ptr_t> ranges;
  for (unsigned index = 0; index < cmp_i->getNumOperands(); index++) {
    const llvm::Value *op = cmp_i->getOperand(index);
    if (std::shared_ptr<VRAScalarNode> op_range =
            std::dynamic_ptr_cast_or_null<VRAScalarNode>(getNode(op))) {
      ranges.push_back(op_range->getRange());
    } else {
      ranges.push_back(nullptr);
    }
  }
  range_ptr_t result = std::dynamic_ptr_cast_or_null<range_t>(handleCompare(ranges, pred));
  LLVM_DEBUG(Logger->logRangeln(result));
  saveValueRange(cmp, result);
}

void VRAnalyzer::handlePhiNode(const llvm::Instruction *phi)
{
  const llvm::PHINode *phi_n = llvm::cast<llvm::PHINode>(phi);
  if (phi_n->getNumIncomingValues() == 0U) {
    return;
  }
  LLVM_DEBUG(Logger->logInstruction(phi));
  RangeNodePtrT res = copyRange(getGlobalStore()->getUserInput(phi));
  for (unsigned index = 0U; index < phi_n->getNumIncomingValues(); index++) {
    const llvm::Value *op = phi_n->getIncomingValue(index);
    NodePtrT op_node = getNode(op);
    if (!op_node)
      continue;
    if (RangeNodePtrT op_range =
            std::dynamic_ptr_cast<VRAScalarNode>(op_node)) {
      res = getUnionRange(res, op_range);
    } else {
      setNode(phi, op_node);
      LLVM_DEBUG(Logger->logInfoln("Pointer PHINode"));
      return;
    }
  }
  setNode(phi, res);
  LLVM_DEBUG(Logger->logRangeln(res));
}

void VRAnalyzer::handleSelect(const llvm::Instruction *i)
{
  const llvm::SelectInst *sel = cast<llvm::SelectInst>(i);
  // TODO handle pointer select
  LLVM_DEBUG(Logger->logInstruction(sel));
  RangeNodePtrT res = getUnionRange(fetchRangeNode(sel->getFalseValue()),
                                    fetchRangeNode(sel->getTrueValue()));
  LLVM_DEBUG(Logger->logRangeln(res));
  saveValueRange(i, res);
}


////////////////////////////////////////////////////////////////////////////////
// Data Handling
////////////////////////////////////////////////////////////////////////////////

const range_ptr_t
VRAnalyzer::fetchRange(const llvm::Value *V)
{
  if (const range_ptr_t Derived = VRAStore::fetchRange(V)) {
    return Derived;
  }

  if (const RangeNodePtrT InputRange = getGlobalStore()->getUserInput(V)) {
    if (const std::shared_ptr<VRAScalarNode> InputScalar =
            std::dynamic_ptr_cast<VRAScalarNode>(InputRange)) {
      return InputScalar->getRange();
    }
  }

  return nullptr;
}

const RangeNodePtrT
VRAnalyzer::fetchRangeNode(const llvm::Value *V)
{
  if (const RangeNodePtrT Derived = VRAStore::fetchRangeNode(V)) {
    if (std::isa_ptr<VRAStructNode>(Derived)) {
      if (RangeNodePtrT InputRange = getGlobalStore()->getUserInput(V)) {
        // fill null input_range fields with corresponding derived fields
        return fillRangeHoles(Derived, InputRange);
      }
    }
    return Derived;
  }

  if (const RangeNodePtrT InputRange = getGlobalStore()->getUserInput(V)) {
    return InputRange;
  }

  return nullptr;
}

NodePtrT
VRAnalyzer::getNode(const llvm::Value *v)
{
  NodePtrT Node = VRAStore::getNode(v);

  if (!Node) {
    std::shared_ptr<VRAStore> ExternalStore = getAnalysisStoreForValue(v);
    if (ExternalStore) {
      Node = ExternalStore->getNode(v);
    }
  }

  if (Node && Node->getKind() == VRANode::VRAScalarNodeK) {
    auto UserInput =
        std::dynamic_ptr_cast_or_null<VRAScalarNode>(getGlobalStore()->getUserInput(v));
    if (UserInput && UserInput->isFinal()) {
      Node = UserInput;
    }
  }

  return Node;
}

void VRAnalyzer::setNode(const llvm::Value *V, NodePtrT Node)
{
  if (isa<GlobalVariable>(V)) {
    // set node in global analyzer
    getGlobalStore()->setNode(V, Node);
    return;
  }
  if (isa<Argument>(V)) {
    std::shared_ptr<VRAFunctionStore> FStore =
        std::static_ptr_cast<VRAFunctionStore>(CodeInt.getFunctionStore());
    FStore->setNode(V, Node);
    return;
  }

  VRAStore::setNode(V, Node);
}

void VRAnalyzer::logRangeln(const llvm::Value *v)
{
  LLVM_DEBUG(if (getGlobalStore()->getUserInput(v)) dbgs() << "(possibly from metadata) ");
  LLVM_DEBUG(Logger->logRangeln(fetchRangeNode(v)));
}
