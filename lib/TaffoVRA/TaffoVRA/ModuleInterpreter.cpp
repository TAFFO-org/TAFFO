#include "ModuleInterpreter.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "VRAFunctionStore.hpp"
#include "VRAGlobalStore.hpp"
#include "VRAStore.hpp"
#include "VRAnalyzer.hpp"
#include "ValueRangeAnalysisPass.hpp"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/IVDescriptors.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <deque>
#include <sstream>
#include <string>
#include <vector>

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;

namespace taffo {

STATISTIC(NumRecurrenceUnsolved, "Number of unsolved recurrences");

#define VRA_RR_KIND_LIST \
  X(Affine)              \
  X(AffineFlattened)     \
  X(AffineDelta)         \
  X(AffineCrossing)      \
  X(Geometric)           \
  X(GeometricFlattened)  \
  X(GeometricDelta)      \
  X(GeometricCrossing)   \
  X(Linear)              \
  X(Init)                \
  X(Fake)                \
  X(Unknown)

#define DEFINE_RR_STATS_FOR_ORIGIN(ORIGIN, LABEL) \
  namespace {                                     \
  VRA_RR_KIND_LIST                                \
  }

#define X(K) STATISTIC(NumPhi##K##Recurrence, "Number of phi " #K " recurrences detected");
DEFINE_RR_STATS_FOR_ORIGIN(Phi, "phi");
#undef X

#define X(K) STATISTIC(NumMem##K##Recurrence, "Number of mem " #K " recurrences detected");
DEFINE_RR_STATS_FOR_ORIGIN(Mem, "mem");
#undef X

#undef DEFINE_RR_STATS_FOR_ORIGIN
#undef VRA_RR_KIND_LIST

static Statistic& getRecurrenceStat(const std::string& Origin, taffo::RangedRecurrence::Kind K) {
  using Kind = taffo::RangedRecurrence::Kind;

  if (Origin == "phi") {
    switch (K) {
    case Kind::Affine:             return NumPhiAffineRecurrence;
    case Kind::AffineFlattened:    return NumPhiAffineFlattenedRecurrence;
    case Kind::AffineDelta:        return NumPhiAffineDeltaRecurrence;
    case Kind::AffineCrossing:     return NumPhiAffineCrossingRecurrence;
    case Kind::Geometric:          return NumPhiGeometricRecurrence;
    case Kind::GeometricFlattened: return NumPhiGeometricFlattenedRecurrence;
    case Kind::GeometricDelta:     return NumPhiGeometricDeltaRecurrence;
    case Kind::GeometricCrossing:  return NumPhiGeometricCrossingRecurrence;
    case Kind::Linear:             return NumPhiLinearRecurrence;
    case Kind::Init:               return NumPhiInitRecurrence;
    case Kind::Fake:               return NumPhiFakeRecurrence;
    default:                       return NumPhiUnknownRecurrence;
    }
  }

  if (Origin == "mem") {
    switch (K) {
    case Kind::Affine:             return NumMemAffineRecurrence;
    case Kind::AffineFlattened:    return NumMemAffineFlattenedRecurrence;
    case Kind::AffineDelta:        return NumMemAffineDeltaRecurrence;
    case Kind::AffineCrossing:     return NumMemAffineCrossingRecurrence;
    case Kind::Geometric:          return NumMemGeometricRecurrence;
    case Kind::GeometricFlattened: return NumMemGeometricFlattenedRecurrence;
    case Kind::GeometricDelta:     return NumMemGeometricDeltaRecurrence;
    case Kind::GeometricCrossing:  return NumMemGeometricCrossingRecurrence;
    case Kind::Linear:             return NumMemLinearRecurrence;
    case Kind::Init:               return NumMemInitRecurrence;
    case Kind::Fake:               return NumMemFakeRecurrence;
    default:                       return NumMemUnknownRecurrence;
    }
  }

  llvm_unreachable("Unhandled recurrence origin in getRecurrenceStat");
}

static bool isLoopLatch(const llvm::Loop* L, llvm::BasicBlock* candidate) {
  llvm::SmallVector<llvm::BasicBlock*, 4> latches;
  L->getLoopLatches(latches);

  auto it = std::find(latches.begin(), latches.end(), candidate);
  return it != latches.end();
}

static bool isLoopExit(const llvm::Loop* L, llvm::BasicBlock* candidate) {
  llvm::SmallVector<llvm::BasicBlock*, 4> exits;
  L->getExitBlocks(exits);

  auto it = std::find(exits.begin(), exits.end(), candidate);
  return it != exits.end();
}

static void collectLoopsPostOrder(Loop* L, SmallVectorImpl<Loop*>& Out) {
  for (Loop* SubL : L->getSubLoops())
    collectLoopsPostOrder(SubL, Out);
  Out.push_back(L);
}

static SmallVector<Loop*> getLoopsInnermostFirst(Function* F, LoopInfo& LI) {
  SmallVector<Loop*> Result;
  for (Loop* TopL : LI)
    collectLoopsPostOrder(TopL, Result);
  return Result;
}

/// @brief check if binary is valid for affine (on recurrence main path)
/// @param V value (binary instr) to analyze
/// @return true if this op is affine compatible
static bool isAffineBinaryOp(const Value* V) {
  const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(V);
  if (!BO)
    return false;

  const unsigned opc = BO->getOpcode();
  return opc == llvm::Instruction::Add || opc == llvm::Instruction::Sub || opc == llvm::Instruction::FAdd
      || opc == llvm::Instruction::FSub;
}

/// @brief check if binary is valid for geometric (on recurrence main path)
/// @param V value (binary instr) to analyze
/// @return true if this op is geometric compatible
static bool isGeometricBinaryOp(const Value* V) {
  const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(V);
  if (!BO)
    return false;

  switch (BO->getOpcode()) {
  case llvm::Instruction::Mul:
  case llvm::Instruction::FMul:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::FDiv: return true;
  default:                      return false;
  }
}

/// @brief Be careful, muladd is often affine ( a[i] += b[i] * x)
/// @param V value (binary instr) to analyze
/// @return true if this op is linear compatible op
static bool isLinearOp(const llvm::Value* V) {
  const auto* Call = llvm::dyn_cast<llvm::CallBase>(V);
  if (!Call)
    return false;

  if (const auto* IF = Call->getCalledFunction()) {
    if (IF->isIntrinsic() && IF->getIntrinsicID() == llvm::Intrinsic::fmuladd) {
      const llvm::Type* Ty = Call->getType();
      if (Ty && (Ty->isFloatTy() || Ty->isDoubleTy()))
        return true;
    }
  }

  return false;
}

const Value* getBaseMemoryObject(const Value* Ptr) {
  if (!Ptr)
    return nullptr;
  Ptr = Ptr->stripPointerCasts();

  const Value* Base = getUnderlyingObject(Ptr, 6);
  if (!Base)
    Base = Ptr;

  return Base->stripPointerCasts();
}

static const Value* getIndexOperand(const Value* Ptr) {
  Ptr = Ptr ? Ptr->stripPointerCasts() : nullptr;
  const auto* GEP = dyn_cast_or_null<GEPOperator>(Ptr);
  if (!GEP)
    return nullptr;
  if (GEP->getNumOperands() < 2)
    return nullptr;
  return GEP->getOperand(GEP->getNumOperands() - 1);
}

static const Value* stripCasts(const Value* V) {
  const Value* Cur = V;
  while (Cur) {
    if (auto* Cast = dyn_cast<CastInst>(Cur)) {
      Cur = Cast->getOperand(0);
      continue;
    }
    if (auto* CE = dyn_cast<ConstantExpr>(Cur)) {
      if (CE->isCast()) {
        Cur = CE->getOperand(0);
        continue;
      }
    }
    if (auto* Op = dyn_cast<Operator>(Cur)) {
      unsigned opc = Op->getOpcode();
      if (opc == Instruction::BitCast || opc == Instruction::AddrSpaceCast) {
        Cur = Op->getOperand(0);
        continue;
      }
    }
    break;
  }
  return Cur;
}

static void printValueName(llvm::raw_ostream& OS, const llvm::Value* V) {
  if (!V)
    return;
  if (!V->getName().empty())
    OS << V->getName();
  else
    V->printAsOperand(OS, false);
}

static void printBaseOp(llvm::raw_ostream& OS, const llvm::Value* Ptr) {
  const llvm::Value* Base = getBaseMemoryObject(Ptr);
  if (Base)
    printValueName(OS, Base);
  else
    OS << "<unknown>";
}

/// @brief Clear mamory print => load/store(basemem)
/// @param OS
/// @param V
static void printValue(llvm::raw_ostream& OS, const llvm::Value* V) {
  if (!V)
    return;

  if (const auto* SI = llvm::dyn_cast<llvm::StoreInst>(V)) {
    OS << "store(";
    printBaseOp(OS, SI->getPointerOperand());
    OS << ")";
    return;
  }

  if (const auto* LI = llvm::dyn_cast<llvm::LoadInst>(V)) {
    OS << "load(";
    printBaseOp(OS, LI->getPointerOperand());
    OS << ")";
    return;
  }

  printValueName(OS, V);
}

static std::string printInstrName(const llvm::Value* V) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  printValue(OS, V);
  return OS.str();
}

static std::string recurrenceOrigin(const llvm::Value* Root) {
  if (llvm::isa<llvm::PHINode>(Root))
    return "phi";
  if (llvm::isa<llvm::StoreInst>(Root))
    return "mem";
  if (llvm::isa<llvm::LoadInst>(Root))
    return "load";
  return "instr";
}

static std::string recurrenceKindLabel(taffo::RangedRecurrence::Kind K) {
  using Kind = taffo::RangedRecurrence::Kind;
  switch (K) {
  case Kind::Affine:             return "affine";
  case Kind::AffineFlattened:    return "affine flattened";
  case Kind::AffineDelta:        return "affine delta";
  case Kind::AffineCrossing:     return "affine crossing";
  case Kind::Geometric:          return "geometric";
  case Kind::GeometricFlattened: return "geometric flattened";
  case Kind::GeometricDelta:     return "geometric delta";
  case Kind::GeometricCrossing:  return "geometric crossing";
  case Kind::Linear:             return "linear";
  case Kind::Init:               return "init";
  case Kind::Fake:               return "fake";
  default:                       return "unknown";
  }
}

/// @brief Implemented until 2D: retrieve all PHI Values refer to vectors indexes. Walk over parent loops.
/// @param LI load instruction
/// @param LIInfo llvm::LoopInfo
/// @return vector with all retrieved PHI Values
static std::vector<const llvm::Value*> getInductionFromLoad(const llvm::LoadInst* LI, const llvm::LoopInfo* LIInfo) {
  std::vector<const llvm::Value*> Result;
  if (!LIInfo)
    return Result;

  const llvm::Loop* CurLoop = LIInfo->getLoopFor(LI->getParent());
  if (!CurLoop)
    return Result;

  llvm::SmallVector<const llvm::BasicBlock*, 4> AllowedHeaders;
  for (const llvm::Loop* L = CurLoop; L; L = L->getParentLoop())
    AllowedHeaders.push_back(L->getHeader());

  const llvm::Value* Ptr = LI->getPointerOperand()->stripPointerCasts();

  // Collect indexes that correspond to array dimensions (up to 2D) walking
  // through chained GEPs (inner to outer).
  llvm::SmallVector<const llvm::Value*, 2> DimIdx;
  while (DimIdx.size() < 2) {
    const auto* GEP = llvm::dyn_cast<llvm::GEPOperator>(Ptr);
    if (!GEP)
      break;

    Type* Ty = GEP->getSourceElementType();
    unsigned Pos = 0;
    for (auto It = GEP->idx_begin(), E = GEP->idx_end(); It != E && DimIdx.size() < 2; ++It, ++Pos) {
      if (Pos == 0) {

        bool IsZeroIdx = false;
        if (auto* CI = llvm::dyn_cast<llvm::ConstantInt>(It->get()))
          IsZeroIdx = CI->isZero();

        if (DimIdx.size() < 2 && ((!llvm::isa<llvm::ArrayType>(Ty) && !llvm::isa<llvm::StructType>(Ty)) || !IsZeroIdx))
          DimIdx.push_back(It->get());

        // Still advance through nested element types when available.
        if (auto* ArrTy0 = llvm::dyn_cast<llvm::ArrayType>(Ty))
          Ty = ArrTy0->getElementType();

        continue;
      }

      if (auto* ArrTy = llvm::dyn_cast<llvm::ArrayType>(Ty)) {
        DimIdx.push_back(It->get());
        Ty = ArrTy->getElementType();
        continue;
      }

      if (auto* ST = llvm::dyn_cast<llvm::StructType>(Ty)) {
        auto* CI = llvm::dyn_cast<llvm::ConstantInt>(It->get());
        if (!CI)
          break;
        unsigned Field = CI->getZExtValue();
        if (Field >= ST->getNumElements())
          break;
        Ty = ST->getElementType(Field);
        continue;
      }

      if (DimIdx.size() < 2)
        DimIdx.push_back(It->get());
    }

    Ptr = GEP->getPointerOperand()->stripPointerCasts();
  }

  if (DimIdx.size() == 2)
    std::reverse(DimIdx.begin(), DimIdx.end());

  // LLVM_DEBUG(tda::log() << "  getInductionFromLoad collected " << DimIdx.size() << " dim idx\n");

  if (DimIdx.empty())
    return Result;

  auto IsHeaderPHI = [&](const llvm::PHINode* PN) {
    const llvm::BasicBlock* BB = PN->getParent();
    return llvm::is_contained(AllowedHeaders, BB);
  };

  auto FindInduction = [&](const llvm::Value* Start) -> const llvm::Value* {
    llvm::SmallVector<const llvm::Value*, 8> Stack;
    llvm::SmallPtrSet<const llvm::Value*, 16> Seen;
    Stack.push_back(Start);

    while (!Stack.empty()) {
      const llvm::Value* V = Stack.pop_back_val();
      if (!Seen.insert(V).second)
        continue;

      if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(V)) {
        if (IsHeaderPHI(PN))
          return PN;
      }

      if (const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
        const llvm::Value* Op0 = BO->getOperand(0);
        const llvm::Value* Op1 = BO->getOperand(1);
        if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(Op0); PN && IsHeaderPHI(PN))
          return PN;
        if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(Op1); PN && IsHeaderPHI(PN))
          return PN;
        if (!llvm::isa<llvm::Constant>(Op0))
          Stack.push_back(Op0);
        if (!llvm::isa<llvm::Constant>(Op1))
          Stack.push_back(Op1);
        continue;
      }

      if (const auto* SI = llvm::dyn_cast<llvm::SelectInst>(V)) {
        for (const llvm::Value* Op : {SI->getTrueValue(), SI->getFalseValue()}) {
          if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(Op); PN && IsHeaderPHI(PN))
            return PN;
          if (!llvm::isa<llvm::Constant>(Op))
            Stack.push_back(Op);
        }
        continue;
      }

      if (const auto* CI = llvm::dyn_cast<llvm::CastInst>(V)) {
        Stack.push_back(CI->getOperand(0));
        continue;
      }

      if (const auto* GEPIdx = llvm::dyn_cast<llvm::GEPOperator>(V)) {
        Stack.push_back(GEPIdx->getOperand(GEPIdx->getNumOperands() - 1));
        continue;
      }

      if (const auto* Cmp = llvm::dyn_cast<llvm::CmpInst>(V)) {
        Stack.push_back(Cmp->getOperand(0));
        Stack.push_back(Cmp->getOperand(1));
        continue;
      }
    }

    return nullptr;
  };

  for (const llvm::Value* Idx : DimIdx)
    if (const llvm::Value* IV = FindInduction(Idx))
      Result.push_back(IV);

  return Result;
}

std::shared_ptr<AnalysisStore> ModuleInterpreter::getStoreForValue(const llvm::Value* V) const {
  assert(V && "Trying to get AnalysisStore for null value.");

  if (llvm::isa<llvm::Constant>(V))
    return GlobalStore;

  for (auto [F, info] : FNs) {
    if (llvm::isa<llvm::Argument>(V) && info.scope.FunctionStore->hasValue(V))
      return info.scope.FunctionStore;

    if (const llvm::Instruction* I = llvm::dyn_cast<llvm::Instruction>(V)) {

      auto BBAIt = info.scope.BBAnalyzers.find(I->getParent());
      if (BBAIt != info.scope.BBAnalyzers.end() && BBAIt->second->hasValue(I))
        return BBAIt->second;
    }
  }

  return nullptr;
}

ModuleInterpreter::ModuleInterpreter(llvm::Module& M, llvm::ModuleAnalysisManager& MAM)
: M(M), GlobalStore(nullptr), curFn(), MAM(MAM), FNs() {}

void ModuleInterpreter::interpret() {

  preSeed();
  if (FNs.size() == 0)
    return;

  LLVM_DEBUG(tda::log() << "\n\n------------------------------------------------------------------------------\n");
  LLVM_DEBUG(tda::log() << "preseed completed: found " << FNs.size() << " visitable functions with " << countLoops()
                        << " loops.\n");
  LLVM_DEBUG(tda::log() << "------------------------------------------------------------------------------\n\n");

  inspect();

  remainingUnsolvedRR = countPotentialRecurrences();
  LLVM_DEBUG(tda::log() << "\n\n------------------------------------------------------------------------------\n");
  LLVM_DEBUG(tda::log() << "inspection completed: found " << remainingUnsolvedRR
                        << " potential detectable recurrences.\n");
  LLVM_DEBUG(tda::log() << "------------------------------------------------------------------------------\n\n");

  resolve();
}

void ModuleInterpreter::computeRecurrenceSummary() {
  RecSummary = RecurrenceSummary();

  for (auto& Entry : FNs) {
    auto& VFI = Entry.second;
    for (auto& RREntry : VFI.RRs) {
      const VRARecurrenceInfo& VRI = RREntry.second;
      if (VRI.RR) {
        const std::string origin = recurrenceOrigin(VRI.root);
        const auto kind = VRI.RR->kind();

        const std::string key = origin + " " + recurrenceKindLabel(kind);
        ++RecSummary.counts[key];

        // Update STATISTIC counters so they are reported in the
        // standard "Statistics Collected" block produced by -stats.
        getRecurrenceStat(origin, kind) += 1;
      }
      else {
        ++RecSummary.unsolved;
        ++NumRecurrenceUnsolved;
      }
    }
  }
}

void ModuleInterpreter::printRecurrenceSummary(llvm::raw_ostream& OS) const {
  if (llvm::AreStatisticsEnabled())
    return;

  OS << "===-------------------------------------------------------------------------===\n";
  OS << "                      ... VRA RR Statistics Collected ...\n";
  OS << "===-------------------------------------------------------------------------===\n\n";

  for (const auto& Entry : RecSummary.counts)
    OS << "    " << Entry.second << " " << Entry.first << " recurrence detected\n";

  if (!RecSummary.counts.empty())
    OS << "\n";

  OS << "    " << RecSummary.unsolved << " recurrence unsolved\n\n";
}

void ModuleInterpreter::resolve() {
  size_t iteration = 1;
  isFallback = false;

  do {

    assemble();

    remainingUnsolvedRR -= solvedRR.size();
    LLVM_DEBUG(tda::log() << "\n\n------------------------------------------------------------------------------\n");
    LLVM_DEBUG(tda::log() << "assembling iter " << iteration << " completed: solved " << solvedRR.size()
                          << " recurrences, remainings " << remainingUnsolvedRR << ".\n");
    LLVM_DEBUG(tda::log() << "------------------------------------------------------------------------------\n\n");

    if (existAtLeastOneLoopWithoutTripCount()) {
      tripCount();

      LLVM_DEBUG(tda::log() << "\n\n------------------------------------------------------------------------------\n");
      LLVM_DEBUG(tda::log() << "trip count iter " << iteration << " completed: found " << solvedTC << " trip count.\n");
      LLVM_DEBUG(tda::log() << "------------------------------------------------------------------------------\n\n");
    }

    ++iteration;
    if ((MaxPropagation && iteration > MaxPropagation)
        || (solvedRR.size() + solvedTC == 0 && remainingUnsolvedRR > 0)) {
      LLVM_DEBUG(
        tda::log() << "Propagation interrupted: after " << iteration
                   << " iteration(s) no fixed point reached: widening falling back remaining RR and last iteration\n");
      fallback();
      isFallback = true;
    }

    propagate();

    if (isFallback) {
      LLVM_DEBUG(tda::log() << "\n\n------------------------------------------------------------------------------\n");
      LLVM_DEBUG(tda::log() << "Fallback completed at iter " << iteration << " \n");
      LLVM_DEBUG(tda::log() << "------------------------------------------------------------------------------\n\n");
    }
    else {
      LLVM_DEBUG(tda::log() << "\n\n------------------------------------------------------------------------------\n");
      LLVM_DEBUG(tda::log() << "propagation iter " << iteration << " completed.\n");
      LLVM_DEBUG(tda::log() << "------------------------------------------------------------------------------\n\n");
    }
  }
  while (solvedRR.size() + solvedTC != 0 && !isFallback);

  LLVM_DEBUG(tda::log() << "saving results...\n");
  GlobalStore->saveResults(M);

  computeRecurrenceSummary();
}

//==================================================================================================
//======================= PRESEEDING METHODS =======================================================
//==================================================================================================

void ModuleInterpreter::preSeed() {

  for (Function& F : M) {
    if (!F.empty() && (TaffoInfo::getInstance().isStartingPoint(F))) {

      // empty scope
      GlobalStore = std::make_shared<VRAGlobalStore>();
      GlobalStore->harvestValueInfo(M);

      interpretFunction(&F);
      EntryFn = &F;
    }
  }

  if (!EntryFn)
    LLVM_DEBUG(tda::log() << " No visitable functions found.\n");
}

/**
 * Before enetering we can consider scopes at state S0: globals and annotation from TAFFO
 * At the end of preseed we have all functions and block scopes at state S1
 */
void ModuleInterpreter::interpretFunction(llvm::Function* F, std::shared_ptr<AnalysisStore> FunctionStore) {

  if (FNs.count(F) && FNs[F].bbFlow.size() > 0) {
    LLVM_DEBUG(tda::log() << "FN[" << F->getName() << "] already interpreted\n");
    return;
  }

  FNs.try_emplace(F, VRAFunctionInfo(F, getMAM()));
  VRAFunctionInfo& VFI = FNs[F];

  if (!FunctionStore)
    FunctionStore = GlobalStore->newFnStore(*this);
  VFI.scope = FunctionScope(FunctionStore);
  curFn.push_back(F);

  llvm::BasicBlock* EntryBlock = &F->getEntryBlock();
  llvm::SmallPtrSet<llvm::BasicBlock*, 4U> VisitedSuccs;
  std::deque<llvm::BasicBlock*> worklist;
  llvm::SmallVector<llvm::Loop*> curLoop;

  worklist.push_back(EntryBlock);
  VFI.scope.BBAnalyzers[EntryBlock] = GlobalStore->newInstructionAnalyzer(*this);

  while (!worklist.empty()) {
    llvm::BasicBlock* curBlock = worklist.front();
    worklist.pop_front();

    auto CAIt = VFI.scope.BBAnalyzers.find(curBlock);
    assert(CAIt != VFI.scope.BBAnalyzers.end());
    std::shared_ptr<CodeAnalyzer> CurAnalyzer = CAIt->second;

    for (llvm::Instruction& I : *curBlock) {
      unsigned curPos = InstrPos.size();
      InstrPos.try_emplace(&I, curPos);
      if (CurAnalyzer->requiresInterpretation(&I))
        interpretCall(CurAnalyzer, &I);
      else if (!curLoop.empty() && isa<PHINode>(&I) && curBlock == VFI.LI->getLoopFor(curBlock)->getHeader())
        CurAnalyzer->analyzePHIStartInstruction(&I);
      else
        CurAnalyzer->analyzeInstruction(&I);
    }

    if (curLoop.empty()) {
      LLVM_DEBUG(tda::log() << "BB[" << curBlock->getName() << "] marked as visited\n");
      VFI.bbFlow.push_back(curBlock);
    }
    else {
      LLVM_DEBUG(tda::log() << "BB[" << curBlock->getName() << "] marked as visited into the loop "
                            << curLoop.back()->getName() << "\n");
      VFI.loops[curLoop.back()].bbFlow.push_back(curBlock);
    }

    llvm::Instruction* Term = curBlock->getTerminator();
    VisitedSuccs.clear();
    for (unsigned NS = 0; NS < Term->getNumSuccessors(); ++NS) {
      llvm::BasicBlock* nextBlock = Term->getSuccessor(NS);

      // Needed just for terminators where the same successor appears twice
      if (VisitedSuccs.count(nextBlock))
        continue;
      else
        VisitedSuccs.insert(nextBlock);

      LLVM_DEBUG(tda::log() << "FN[" << F->getName() << "] >> Follow path " << curBlock->getName() << " ==> "
                            << nextBlock->getName() << ": ");
      switch (followPath(VFI, curBlock, nextBlock, curLoop)) {
      case FollowingPathResponse::ENQUE_BLOCK: {
        LLVM_DEBUG(tda::log() << "ENQUE_BLOCK.\n");
        worklist.push_front(nextBlock);
        break;
      }
      case FollowingPathResponse::LOOP_FORK: {
        LLVM_DEBUG(tda::log() << "LOOP_FORK. New VRALoopInfo created, new context, header enqueued ==> ");

        // add also loop header to trace when go into loop into propagate phase()
        if (curLoop.empty()) {
          LLVM_DEBUG(tda::log() << "added loop header to bbFlow\n");
          VFI.bbFlow.push_back(nextBlock);
        }
        else {
          LLVM_DEBUG(tda::log() << "added loop header to bbFlow of the loop " << curLoop.back()->getName() << "\n");
          VFI.loops[curLoop.back()].bbFlow.push_back(nextBlock);
        }

        // here nextBlock is the new loop header
        Loop* dstLoop = VFI.LI->getLoopFor(nextBlock);
        curLoop.push_back(dstLoop);
        VFI.loops.try_emplace(dstLoop, VRALoopInfo(dstLoop));
        worklist.push_front(nextBlock);

        break;
      }
      case FollowingPathResponse::LOOP_JOIN: {
        LLVM_DEBUG(tda::log() << "LOOP_JOIN. Old context restored, exits enqueued\n");
        llvm::Loop* dstLoop = curLoop.back();
        curLoop.pop_back();

        // insert once, avoid exit block duplicated
        SmallVector<BasicBlock*, 4> exits;
        dstLoop->getExitBlocks(exits);

        llvm::SmallPtrSet<BasicBlock*, 8> queuedBlocks;
        for (llvm::BasicBlock* QB : worklist)
          queuedBlocks.insert(QB);

        // allow enqueue only if every predecessor has already been visited in the current flow
        const llvm::SmallVector<llvm::BasicBlock*>& curFlow =
          curLoop.empty() ? VFI.bbFlow : VFI.loops.lookup(curLoop.back()).bbFlow;

        llvm::SmallPtrSet<BasicBlock*, 32> visitedBlocks;
        visitedBlocks.insert(curFlow.begin(), curFlow.end());

        for (BasicBlock* EB : exits) {
          bool allPredVisited = true;
          for (BasicBlock* Pred : predecessors(EB)) {
            if (!visitedBlocks.count(Pred) && !dstLoop->contains(Pred)) {
              allPredVisited = false;
              LLVM_DEBUG(tda::log() << "skip exit " << EB->getName() << " because predecessor " << Pred->getName()
                                    << " not in current flow\n");
              break;
            }
          }

          if (allPredVisited && queuedBlocks.insert(EB).second) {
            LLVM_DEBUG(tda::log() << "enqueue exit " << EB->getName() << "\n");
            worklist.push_back(EB);
          }
          else if (allPredVisited) {
            LLVM_DEBUG(tda::log() << "skip exit " << EB->getName() << " already queued\n");
          }
        }

        break;
      }
      default: {
        LLVM_DEBUG(tda::log() << "NO ACTION.\n");
        break;
      }
      }

      updateSuccessorAnalyzer(CurAnalyzer, Term, NS);
    }

    GlobalStore->convexMerge(*CurAnalyzer);
  }
  GlobalStore->convexMerge(*FunctionStore);

  curFn.pop_back();
}

FollowingPathResponse ModuleInterpreter::followPath(VRAFunctionInfo info,
                                                    llvm::BasicBlock* src,
                                                    llvm::BasicBlock* dst,
                                                    llvm::SmallVector<llvm::Loop*> nesting) const {

  llvm::Loop* srcLoop = info.LI->getLoopFor(src);
  llvm::Loop* dstLoop = info.LI->getLoopFor(dst);

  if (srcLoop && isLoopExit(srcLoop, dst))
    return FollowingPathResponse::NO_ENQUE;

  llvm::SmallVector<llvm::BasicBlock*> curFlow;
  if (!srcLoop)
    curFlow = info.bbFlow;                       // flow preso dal corpo della funzione
  else if (nesting.back() == srcLoop)
    curFlow = info.loops.lookup(srcLoop).bbFlow; // flow da analizzare preso dal loop corrente

  for (BasicBlock* pred : predecessors(dst)) {

    // do not analyze header predecessors if latch
    llvm::Loop* predLoop = info.LI->getLoopFor(pred);
    if (predLoop && isLoopLatch(predLoop, pred) && dst == predLoop->getHeader())
      continue;
    if (dstLoop && predLoop == dstLoop->getParentLoop() && dstLoop->getHeader() == dst)
      continue; // blocco prima del loop più esterno per forza visitato

    // typically loop exits path
    if (predLoop && dstLoop && info.loops.count(predLoop) && dstLoop == predLoop->getParentLoop()) {
      VRALoopInfo loopInfo = info.loops.lookup(predLoop);
      if (!loopInfo.isEntirelyVisited()) {
        LLVM_DEBUG(tda::log() << "pred loop " << predLoop->getName() << " is not entirely visited yet ==> ");
        return FollowingPathResponse::NO_ENQUE;
      }
    }

    auto it = std::find(curFlow.begin(), curFlow.end(), pred);
    if (it == curFlow.end()) {
      LLVM_DEBUG(tda::log() << "pred block " << pred->getName() << " is not visited yet ==> ");
      return FollowingPathResponse::NO_ENQUE;
    }
  }

  // percorso da esterno a nuovo loop
  if (dstLoop && dstLoop->getHeader() == dst && srcLoop != dstLoop) {
    if (!info.loops.count(dstLoop))
      return FollowingPathResponse::LOOP_FORK;
  }

  // latch del loop che punta al suo header:
  if (dstLoop && dstLoop->getHeader() == dst && srcLoop == dstLoop && isLoopLatch(srcLoop, src)) {
    // torna loop join se il loop risulta completamente visitato, altrimenti no_enque
    LLVM_DEBUG(tda::log() << "path latch -> header (same loop) ==> ");
    return info.loops.lookup(srcLoop).isEntirelyVisited() ? FollowingPathResponse::LOOP_JOIN
                                                          : FollowingPathResponse::NO_ENQUE;
  }

  // path standard: se non visitato permetti, altrimenti evita
  auto it = std::find(curFlow.begin(), curFlow.end(), dst);
  return it == curFlow.end() ? FollowingPathResponse::ENQUE_BLOCK : FollowingPathResponse::NO_ENQUE;
}

void ModuleInterpreter::updateSuccessorAnalyzer(std::shared_ptr<CodeAnalyzer> CurrentAnalyzer,
                                                llvm::Instruction* TermInstr,
                                                unsigned SuccIdx) {
  llvm::DenseMap<llvm::BasicBlock*, std::shared_ptr<CodeAnalyzer>>& BBAnalyzers = FNs[curFn.back()].scope.BBAnalyzers;
  llvm::BasicBlock* SuccBB = TermInstr->getSuccessor(SuccIdx);

  std::shared_ptr<CodeAnalyzer> SuccAnalyzer;
  auto SAIt = BBAnalyzers.find(SuccBB);
  if (SAIt == BBAnalyzers.end()) {
    SuccAnalyzer = CurrentAnalyzer->clone();
    BBAnalyzers[SuccBB] = SuccAnalyzer;
  }
  else {
    SuccAnalyzer = SAIt->second;
    SuccAnalyzer->convexMerge(*CurrentAnalyzer);
  }

  CurrentAnalyzer->setPathLocalInfo(SuccAnalyzer, TermInstr, SuccIdx);
}

void ModuleInterpreter::updateKnownSuccessorAnalyzer(std::shared_ptr<CodeAnalyzer> CurrentAnalyzer,
                                                     llvm::BasicBlock* nextBlock,
                                                     const llvm::BasicBlock* curBlock) {
  llvm::DenseMap<llvm::BasicBlock*, std::shared_ptr<CodeAnalyzer>>& BBAnalyzers = FNs[curFn.back()].scope.BBAnalyzers;
  std::shared_ptr<CodeAnalyzer> SuccAnalyzer;
  auto SAIt = BBAnalyzers.find(nextBlock);
  if (SAIt == BBAnalyzers.end()) {
    SuccAnalyzer = CurrentAnalyzer->clone();
    BBAnalyzers[nextBlock] = SuccAnalyzer;
  }
  else {
    SuccAnalyzer = SAIt->second;
    SuccAnalyzer->convexMerge(*CurrentAnalyzer);
  }
}

void ModuleInterpreter::interpretCall(std::shared_ptr<CodeAnalyzer> CurAnalyzer, llvm::Instruction* I) {
  llvm::CallBase* CB = llvm::cast<llvm::CallBase>(I);
  llvm::Function* F = CB->getCalledFunction();
  if (!F || F->empty())
    return;

  std::shared_ptr<AnalysisStore> FunctionStore = GlobalStore->newFnStore(*this);

  CurAnalyzer->prepareForCall(I, FunctionStore);
  interpretFunction(F, FunctionStore);
  CurAnalyzer->returnFromCall(I, FunctionStore);
}

void ModuleInterpreter::resolveCall(std::shared_ptr<CodeAnalyzer> CurAnalyzer, llvm::Instruction* I) {
  llvm::CallBase* CB = llvm::cast<llvm::CallBase>(I);
  llvm::Function* F = CB->getCalledFunction();
  if (!F || F->empty())
    return;

  std::shared_ptr<AnalysisStore> FunctionStore = GlobalStore->newFnStore(*this);
  FNs[F].scope.BBAnalyzers.clear();
  FNs[F].scope.FunctionStore = nullptr;
  FNs[F].scope = FunctionScope(FunctionStore);
  FNs[F].scope.BBAnalyzers[&F->getEntryBlock()] = GlobalStore->newInstructionAnalyzer(*this);

  CurAnalyzer->prepareForCallPropagation(I, FunctionStore);
  propagateFunction(F, FunctionStore);
  CurAnalyzer->returnFromCallPropagation(I, FunctionStore);
}

//==================================================================================================
//======================= INSPECTION METHODS =======================================================
//==================================================================================================

void ModuleInterpreter::inspect() {

  for (auto& Entry : FNs) {
    llvm::Function* F = Entry.first;
    auto& VFI = Entry.second;
    if (VFI.loops.size() == 0)
      continue;

    for (llvm::Loop* L : getLoopsInnermostFirst(F, *VFI.LI)) {
      for (BasicBlock* loopBlock : L->blocks()) {
        for (Instruction& I : *loopBlock) {
          if (!isa<PHINode>(I) && !isa<StoreInst>(I))
            continue;

          VRARecurrenceInfo VRI(static_cast<const llvm::Value*>(&I));
          if (isa<PHINode>(I) && loopBlock == L->getHeader()) {
            handlePHIChain(VFI, L, dyn_cast<PHINode>(&I), VRI);

            if (isInductionVariable(F, L, dyn_cast<PHINode>(&I)))
              VFI.loops[L].InductionVariable = dyn_cast<PHINode>(&I);
          }
          else if (isa<StoreInst>(I)) {
            handleStoreChain(VFI, L, dyn_cast<StoreInst>(&I), VRI);
            if (VRI.kind == VRAInspectionKind::ASSIGN) {
              VRAAssignationInfo VAI(dyn_cast<StoreInst>(&I), VRI.chain, VRI.loads);
              VFI.addAssignmentInfo(VAI);
              continue;
            }
            else if (VRI.kind != VRAInspectionKind::REC)
              continue;
          }

          VFI.addRecurrenceInfo(VRI);
        }
      }
    }
  }
}

bool ModuleInterpreter::isInductionVariable(llvm::Function* F, llvm::Loop* L, const llvm::PHINode* PHI) {
  if (!L || !PHI)
    return false;

  // 1) Canonical IV (fast path).
  if (const PHINode* CIV = L->getCanonicalInductionVariable())
    if (CIV == PHI)
      return true;

  // Must be a PHI in the loop header.
  if (PHI->getParent() != L->getHeader())
    return false;

  auto SE = FNs[F].SE;

  // 2) Use IVDescriptors when SCEV is available (handles many non-canonical cases).
  if (SE && SE->isSCEVable(PHI->getType())) {
    InductionDescriptor D;
    auto* PN = const_cast<llvm::PHINode*>(PHI);
    if (InductionDescriptor::isInductionPHI(PN, L, SE, D))
      return true;

    // 3) SCEV fallback: PN is an add recurrence on L with invariant start/step.
    const SCEV* S = SE->getSCEV(const_cast<PHINode*>(PHI));
    if (const auto* AR = dyn_cast<SCEVAddRecExpr>(S)) {
      if (AR->getLoop() == L) {
        const SCEV* Start = AR->getStart();
        const SCEV* Step = AR->getStepRecurrence(*SE);
        if (SE->isLoopInvariant(Start, L) && SE->isLoopInvariant(Step, L))
          return true;
      }
    }
  }

  // todo: expand here new way to detect IV
  return false;
}

void ModuleInterpreter::handlePHIChain(VRAFunctionInfo VFI, Loop* L, const PHINode* PHI, VRARecurrenceInfo& VRI) {

  VRI.kind = VRAInspectionKind::UNKNOWN;

  // retrieve PHI latch
  const Value* latchIncoming = nullptr;
  if (auto* latch = L->getLoopLatch())
    latchIncoming = PHI->getIncomingValueForBlock(latch);
  if (!latchIncoming) {
    LLVM_DEBUG(tda::log() << " missing latch incoming block, abort\n");
    VRI.chain.clear();
    return;
  }

  SmallVector<const Value*, 32> worklist;
  DenseMap<const Value*, const Value*> preds;
  auto enqueue = [&](const Value* from, const Value* to) {
    if (preds.find(to) != preds.end())
      return;
    preds[to] = from;
    worklist.push_back(to);
  };

  preds[PHI] = nullptr;
  worklist.push_back(PHI);

  while (!worklist.empty()) {
    const Value* cur = worklist.pop_back_val();

    if (cur == latchIncoming) {
      SmallVector<const Value*, 32> Rev;
      VRI.chain.clear();

      const Value* V = cur;
      while (V) {
        Rev.push_back(V);
        auto It = preds.find(V);
        if (It == preds.end())
          break;
        V = It->second;
      }

      // reverse
      for (auto It = Rev.rbegin(); It != Rev.rend(); ++It) {
        if (!isa<BinaryOperator>(*It) && !VFI.RRs.count(*It) && !isLinearOp(*It))
          continue;
        if (VFI.RRs.count(*It))
          VRI.innerRR = *It;
        VRI.chain.push_back(*It);
      }

      VRI.kind = VRAInspectionKind::REC; // ring found

      LLVM_DEBUG(tda::log() << "FOUND REC: " << VRI.chainToString());
      return;
    }

    for (const User* U : cur->users()) {
      auto* I = dyn_cast<Instruction>(U);
      if (!I)
        continue;

      // list of plausible instruction which can continue the flow
      if (isa<CastInst>(I) || isa<BinaryOperator>(I) || isa<CallInst>(I) || VFI.RRs.count(I)) {
        enqueue(cur, I);
        continue;
      }
    }
  }

  LLVM_DEBUG(tda::log() << "UNNKONW REC: " << VRI.chainToString() << "\n");
  VRI.chain.clear();
}

void ModuleInterpreter::handleStoreChain(VRAFunctionInfo VFI, Loop* L, const StoreInst* Store, VRARecurrenceInfo& VRI) {

  VRI.chain.clear();
  VRI.kind = VRAInspectionKind::UNKNOWN;

  const Value* StoreValue = Store->getValueOperand();
  const Value* StoreBase = getBaseMemoryObject(Store->getPointerOperand());
  if (!StoreBase || !StoreValue)
    return;

  bool couldBeInit = false;
  SmallVector<const Value*, 32> worklist;
  DenseMap<const Value*, const Value*> preds;
  DenseSet<const Value*> visited;

  auto enqueue = [&](const Value* from, const Value* to) {
    if (visited.contains(to) || preds.find(to) != preds.end())
      return;
    preds[to] = from;
    worklist.push_back(to);
  };

  auto buildChain = [&](const Value* start) {
    SmallVector<const Value*, 32> Rev;
    VRI.chain.clear();

    const Value* V = start;
    while (V) {
      Rev.push_back(V);
      auto It = preds.find(V);
      if (It == preds.end())
        break;
      V = It->second;
    }

    for (auto It = Rev.rbegin(); It != Rev.rend(); ++It) {
      if (!isa<BinaryOperator>(*It) && !VFI.RRs.count(*It) && !isLinearOp(*It))
        continue;
      if (VFI.RRs.count(*It))
        VRI.innerRR = *It;
      VRI.chain.push_back(*It);
    }
  };

  SmallVector<const llvm::LoadInst*> loads; // collect load for crossing

  preds[StoreValue] = nullptr;              // root of the back-trace
  worklist.push_back(StoreValue);

  const Value* last = nullptr;

  while (!worklist.empty()) {
    const Value* cur = worklist.pop_back_val();
    last = cur;
    if (!visited.insert(cur).second)
      continue;

    if (isa<Constant>(cur)) {
      couldBeInit = true;
      continue;
    }

    if (auto* Load = dyn_cast<LoadInst>(cur)) {

      const Value* LoadBase = getBaseMemoryObject(Load->getPointerOperand());
      if (StoreBase == LoadBase) {
        buildChain(cur);

        VRI.kind = VRAInspectionKind::REC; // ring found
        VRI.loadJunction = Load;
        VRI.loads = loads;

        LLVM_DEBUG(tda::log() << "FOUND REC: " << VRI.chainToString());
        return;
      }

      loads.push_back(Load);
      couldBeInit = true;
      continue; // load from a different base: treat as init candidate
    }

    if (auto* callInstr = dyn_cast<CallInst>(cur)) {
      if (const Function* callee = callInstr->getCalledFunction();
          callee && callee->isIntrinsic() && callee->getIntrinsicID() == llvm::Intrinsic::fmuladd) {
        for (const Value* Op : callInstr->args()) {
          if (isa<Constant>(Op)) {
            couldBeInit = true;
            continue;
          }
          enqueue(cur, Op);
        }
        continue;
      }
      Type* retTy = callInstr->getType();
      couldBeInit = !retTy->isVoidTy();
      continue; // stop: call result is a source
    }

    // Walk backwards through operands to reach defining loads.
    if (auto* I = dyn_cast<Instruction>(cur)) {
      for (const Value* Op : I->operands()) {
        if (isa<Constant>(Op)) {
          couldBeInit = true;
          continue;
        }
        enqueue(cur, Op);
      }
    }

    if (auto* I = dyn_cast<CastInst>(cur)) {
      for (const Value* Op : I->operands()) {
        if (isa<Constant>(Op)) {
          couldBeInit = true;
          continue;
        }
        enqueue(cur, Op);
      }
    }

    if (auto* PHI = dyn_cast<PHINode>(cur)) {
      auto* I = dyn_cast<Instruction>(PHI);
      if (I->getParent() == L->getHeader())
        couldBeInit = true;
    }
  }

  // other operands can bring to init instead of unknown

  if (couldBeInit) {
    VRI.kind = VRAInspectionKind::ASSIGN;
    VRI.loads = loads;

    if (last)
      buildChain(last);

    LLVM_DEBUG(tda::log() << "FOUND ASSIGN: " << VRI.chainToString());
  }
  else {
    LLVM_DEBUG(tda::log() << "UNNKONW REC: " << VRI.chainToString());
  }
}

std::string VRARecurrenceInfo::chainToString() {
  std::string S;
  llvm::raw_string_ostream OS(S);

  OS << "  flow: ";
  printValue(OS, root);
  OS << " => ";

  for (unsigned i = 0; i < chain.size(); ++i) {
    const llvm::Value* V = chain[i];
    printValue(OS, V);

    if (i + 1 < chain.size())
      OS << " => ";
  }

  if (kind == VRAInspectionKind::REC) {
    OS << " => ";
    printValueName(OS, root);
    OS << " || RECURRENCE";
  }
  else if (kind == VRAInspectionKind::ASSIGN) {
    OS << " || ASSIGNATION";
  }
  else {
    OS << " || UNKNOWN";
  }

  OS << "\n";

  return OS.str();
}

//==================================================================================================
//======================= ASSEMBLING METHODS =======================================================
//==================================================================================================

void ModuleInterpreter::assemble() {
  solvedRR.clear();

  for (auto& Entry : FNs) {
    llvm::Function* F = Entry.first;
    auto& VFI = Entry.second;

    for (auto RREntry = VFI.RRs.begin(); RREntry != VFI.RRs.end();) {
      const llvm::Value* root = RREntry->first;
      auto& VRI = RREntry->second;

      if (VRI.RR) {
        RREntry++;
        continue; // already solved
      }

      LLVM_DEBUG(tda::log() << "\n\n[VRA] >> [ASSEMBLE] >> FN[" << F->getName() << "] - Recognization of "
                            << printInstrName(root) << " instr: " << root << "\n");

      // add here new recurrences
      if (isLinearRecurrence(VRI)) {
        RREntry++;
        continue;
      }
      else if (isDeltaAffineRecurrence(VRI)) {
        RREntry++;
        continue;
      }
      else if (isAffineRecurrence(VRI)) {
        RREntry++;
        continue;
      }
      else if (isDeltaGeometricRecurrence(VRI)) {
        RREntry++;
        continue;
      }
      else if (isGeometricRecurrence(VRI)) {
        RREntry++;
        continue;
      }
      else if (isFakeRecurrence(VRI)) {
        RREntry++;
        continue;
      }

      LLVM_DEBUG(tda::log() << " no implemented recurrence detected: removed\n");
      auto Next = std::next(RREntry);
      VFI.RRs.erase(RREntry);
      RREntry = Next;
    }

    for (auto ASEntry = VFI.ASs.begin(); ASEntry != VFI.ASs.end();) {
      const llvm::Value* root = ASEntry->first;
      auto& VAI = ASEntry->second;

      LLVM_DEBUG(tda::log() << "\n\n[VRA] >> [ASSEMBLE] >> FN[" << F->getName() << "] - Recognization of "
                            << printInstrName(root) << " instr: " << root << "\n");

      if (isCrossingAffineRecurrence(VAI)) {
        if (VAI.RR) {
          auto Next = std::next(ASEntry);
          VFI.ASs.erase(ASEntry);
          ASEntry = Next;
          continue;
        }
        else {
          ++ASEntry;
          continue;
        }
      }
      else if (isCrossingGeometricRecurrence(VAI)) {
        if (VAI.RR) {
          auto Next = std::next(ASEntry);
          VFI.ASs.erase(ASEntry);
          ASEntry = Next;
          continue;
        }
        else {
          ++ASEntry;
          continue;
        }
      }

      LLVM_DEBUG(tda::log() << " no implemented recurrence detected: removed\n");
      auto Next = std::next(ASEntry);
      VFI.ASs.erase(ASEntry);
      ASEntry = Next;
    }
  }
}

/// Strip di cast/wrappers comuni per risalire al "vero" producer.
/// - CastInst copre: sext/zext/trunc/bitcast/sitofp/fptosi/ptrtoint/inttoptr, etc.
static const Value* stripCastsAndWrappers(const Value* V) {
  while (V) {
    if (auto* CI = dyn_cast<CastInst>(V)) {
      V = CI->getOperand(0);
      continue;
    }
    if (auto* FI = dyn_cast<FreezeInst>(V)) {
      V = FI->getOperand(0);
      continue;
    }
    if (auto* UO = dyn_cast<UnaryOperator>(V)) {
      if (UO->getOpcode() == Instruction::FNeg) {
        V = UO->getOperand(0);
        continue;
      }
    }
    break;
  }
  return V;
}

/// True se PhiLoop è uguale a UseLoop o un suo parent (risalendo la catena dei parent).
static bool isSameOrParentLoop(const Loop* UseLoop, const Loop* PhiLoop) {
  if (!UseLoop || !PhiLoop)
    return false;
  for (const Loop* L = UseLoop; L; L = L->getParentLoop())
    if (L == PhiLoop)
      return true;
  return false;
}

/// Colleziona i PHI che influenzano Root (def-use backward),
/// filtrando SOLO i PHI presenti nei loop header (BB == Loop->getHeader()).
/// Include i PHI di loop padri del loop che contiene UseCtx.
static SmallVector<const PHINode*, 4>
collectInfluencingHeaderPHIs(const Value* Root, const Instruction* UseCtx, const LoopInfo& LI) {
  SmallVector<const PHINode*, 4> Result;
  if (!Root || !UseCtx)
    return Result;

  const Loop* UseLoop = LI.getLoopFor(UseCtx->getParent());

  SmallVector<const Value*, 32> Stack;
  SmallPtrSet<const Value*, 32> Visited;
  SmallPtrSet<const PHINode*, 16> SeenPHI;

  Stack.push_back(Root);

  while (!Stack.empty()) {
    const Value* V = Stack.pop_back_val();
    V = stripCastsAndWrappers(V);
    if (!V)
      continue;

    if (!Visited.insert(V).second)
      continue;

    // Caso PHI
    if (auto* PN = dyn_cast<PHINode>(V)) {
      const BasicBlock* PhiBB = PN->getParent();
      const Loop* PhiLoop = LI.getLoopFor(PhiBB);

      bool ok = false;

      // 1) Deve appartenere a un loop (per essere "loop header PHI")
      if (PhiLoop) {
        // 2) Deve essere NELL'HEADER del SUO loop
        if (PhiBB == PhiLoop->getHeader()) {
          if (!UseLoop) {
            // Uso fuori da loop: qui decidi policy.
            // Scelta: accettiamo comunque PHI header (di qualsiasi loop).
            ok = true;
          }
          else {
            // 3) Deve essere in UseLoop o in un suo parent (loop esterni),
            //    NON in subloop.
            ok = isSameOrParentLoop(UseLoop, PhiLoop);
          }
        }
      }

      if (ok && SeenPHI.insert(PN).second)
        Result.push_back(PN);

      // Nota: NON espandiamo gli incoming del PHI per evitare di “risalire”
      // dentro la logica di update; qui vogliamo solo le dipendenze “IV-like”.
      continue;
    }

    // Caso istruzione generica: espandi gli operandi
    if (auto* I = dyn_cast<Instruction>(V)) {
      for (const Value* Op : I->operands())
        Stack.push_back(Op);
      continue;
    }

    // Constant/Argument/Global: fine ramo
  }

  return Result;
}

bool ModuleInterpreter::analyzeSolvability(const llvm::Value* cur,
                                           VRAFunctionInfo& VFI,
                                           VRARecurrenceInfo& VRI,
                                           VRALoopInfo& VLI) {
  if (cur == VRI.root)
    return true;

  if (auto CB = dyn_cast<CallBase>(cur)) {
    if (!VLI.isInvariant(cur)) {
      bool atLeastOneUnsolvedArg = false;
      for (const Use& Arg : CB->args()) {
        const llvm::Value* ArgV = Arg.get();
        if (VLI.isInvariant(ArgV))
          continue;
        if (VFI.RRs.count(ArgV) && !VFI.RRs[ArgV].lastRange)
          atLeastOneUnsolvedArg = true;
      }
      if (atLeastOneUnsolvedArg) {
        VRI.depsOnFn.push_back(CB->getCalledFunction());
        return false;
      }

      VRI.depsOnFn.clear();
    }
    else {
      // LLVM_DEBUG(tda::log() << " call invariant operand, cur range usable\n");
    }
  }
  else if (!VLI.isInvariant(cur)) {
    if (auto Load = dyn_cast<LoadInst>(cur)) {
      auto IVs = getInductionFromLoad(Load, VFI.LI);
      // LLVM_DEBUG(if (!IVs.empty()) { tda::log() << " (IVs: "; for (auto *IV : IVs) tda::log() << IV->getName() << "
      // "; tda::log() << ") "; });
      bool IVSolved = false;
      for (const auto* IV : IVs) {
        if (VFI.RRs.count(IV) && VFI.RRs[IV].lastRange) {
          IVSolved = true;
          break;
        }
      }
      if (IVSolved) {

        // if idx solved check base BE CAREFUL: IMPLEMENT DOMINANCE BETWEEN L/S INSTR
        const Value* LoadBase = getBaseMemoryObject(Load->getPointerOperand());
        if (LoadBase) {

          if ((VRI.loadJunction && IVs.size() > getInductionFromLoad(VRI.loadJunction, VFI.LI).size())
              || isa<PHINode>(VRI.root)) {
            // LLVM_DEBUG(tda::log() << " found load from array with higher dim: " << Load << "\n");
            VRI.loadHigherDim = dyn_cast<LoadInst>(Load);
          }

          for (auto [_, VF] : FNs)
            for (auto [R, RR] : VF.RRs) {
              const auto* RootStore = dyn_cast<StoreInst>(R);
              if (!RootStore || R == VRI.root)
                continue;

              if (!RR.lastRange && isBefore(R, Load)) {
                VRI.depsOnRR.push_back(const_cast<llvm::Value*>(R));
                // LLVM_DEBUG(tda::log() << " dep on past store unsolved ");
                return false;
              }
            }
        }
        return true;
      }
    }
    else

      if (!VFI.RRs.count(cur)) {
      auto PHIs = collectInfluencingHeaderPHIs(cur, dyn_cast<Instruction>(cur), *VFI.LI);
      for (auto PHI : PHIs) {
        // LLVM_DEBUG(tda::log() << " trovato phi radice: "<<PHI->getName() << "| ");
        if (VFI.RRs.count(PHI) && VFI.RRs[PHI].lastRange)
          return true;
        VRI.depsOnRR.push_back(const_cast<llvm::Value*>(dyn_cast<Value>(PHI)));
        return false;
      }
    }
    else {
      if (VFI.RRs.count(cur) && VFI.RRs[cur].lastRange)
        return true;
      VRI.depsOnRR.push_back(const_cast<llvm::Value*>(cur));
      return false;
    }
  }

  return true;
}

bool ModuleInterpreter::isSolvableDependenceTreeBackwark(const llvm::Value* V, llvm::Loop* L, VRARecurrenceInfo& VRI) {
  if (!V || !L)
    return false;
  if (V == VRI.root || isa<Constant>(V))
    return true;

  const auto* I = llvm::dyn_cast<llvm::Instruction>(V);
  if (!I)
    return true;

  llvm::Function* F = const_cast<llvm::Function*>(I->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  VRALoopInfo& VLI = FNs[F].loops[L];

  SmallVector<const Value*, 32> worklist;
  DenseMap<const Value*, const Value*> preds;
  DenseSet<const Value*> visited;

  auto enqueue = [&](const Value* from, const Value* to) {
    if (!to || visited.contains(to) || preds.find(to) != preds.end())
      return;
    preds[to] = from;
    worklist.push_back(to);
  };

  preds[V] = nullptr;
  worklist.push_back(V);

  while (!worklist.empty()) {
    const Value* cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    // LLVM_DEBUG(tda::log() << " (analyzing backward " << printInstrName(cur) << ") => \n");

    if (!analyzeSolvability(cur, VFI, VRI, VLI))
      return false;

    if (auto* Inst = dyn_cast<Instruction>(cur)) {
      for (const Value* Op : Inst->operands()) {
        if (isa<Constant>(Op))
          continue;
        enqueue(cur, Op);
      }
    }
  }

  return true;
}

bool ModuleInterpreter::isAffineRecurrence(VRARecurrenceInfo& VRI) {
  if (VRI.kind != VRAInspectionKind::REC)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as affine recurrence... \n");

  bool isMulAdd = false;
  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VRI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(VRI.root)) {

    bool isSolvable = VRI.chain.size() > 0;
    for (const auto* RRNode : VRI.chain) {

      // handling of muladd affine
      if (isLinearOp(RRNode) && VRI.chain.size() == 1) {
        const auto* Call = llvm::cast<llvm::CallBase>(RRNode);

        for (unsigned ArgIdx = 0, End = Call->arg_size(); ArgIdx < End && isSolvable; ++ArgIdx) {
          const Value* Arg = Call->getArgOperand(ArgIdx);

          isSolvable &= isSolvableDependenceTreeBackwark(Arg, L, VRI);
        }
        isMulAdd = true;
        break;
      }

      if (!isAffineBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VRI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VRI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    // pattern sum += a[i] * x
    if (VRI.loadHigherDim && isMulAdd) {
      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildAffinePHIMulAddRecurrence(VRI, PN);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
        }
      }
    }
    // pattern sum += a[i]
    else if (VRI.loadHigherDim) {
      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildPHIAffineFlattingRecurrence(VRI, PN);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "recognized (spatial) " << RR->toString() << " \n\n");
        }
      }
    }

    // pattern a += x
    else if (auto* latch = L->getLoopLatch()) {
      auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
      std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildAffinePHIRecurrence(PN);
      if (RR) {
        VRI.RR = RR;
        solvedRR.push_back(VRI.root);
        LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
      }
    }
  }
  else if (const auto* Store = llvm::dyn_cast<llvm::StoreInst>(VRI.root)) {

    bool isSolvable = VRI.chain.size() > 0;
    for (const auto* RRNode : VRI.chain) {

      if (isLinearOp(RRNode) && VRI.chain.size() == 1) {
        const auto* Call = llvm::cast<llvm::CallBase>(RRNode);

        for (unsigned ArgIdx = 0, End = Call->arg_size(); ArgIdx < End && isSolvable; ++ArgIdx) {
          const Value* Arg = Call->getArgOperand(ArgIdx);

          isSolvable &= isSolvableDependenceTreeBackwark(Arg, L, VRI);
        }
        isMulAdd = true;
        break;
      }

      if (!isAffineBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VRI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VRI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    // pattern a[i] += b[i][j] * x
    if (VRI.loadHigherDim && isMulAdd) {
      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildAffineStoreMulAddRecurrence(VRI, Store);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
        }
      }
    }
    else

      // pattern a[i] += b[i][j]
      if (VRI.loadHigherDim) {
        if (auto* latch = L->getLoopLatch()) {
          auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
          std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildAffineFlattingRecurrence(VRI, Store);
          if (RR) {
            VRI.RR = RR;
            solvedRR.push_back(VRI.root);
            LLVM_DEBUG(tda::log() << "recognized (spatial) " << RR->toString() << " \n\n");
          }
        }
      }

      else if (getBaseMemoryObject(Store->getPointerOperand())
               == getBaseMemoryObject(VRI.loadJunction->getPointerOperand())) {
        const Value* StoreIdx = getIndexOperand(Store->getPointerOperand());
        const Value* LoadIdx = getIndexOperand(VRI.loadJunction->getPointerOperand());

        // CASE loop extra beyond IV: for (k) A[i] = A[i] + C
        int maxDistance = 0;
        const llvm::Loop* maxD_L;
        auto IVs = getInductionFromLoad(VRI.loadJunction, VFI.LI);
        for (auto IV : IVs) {
          auto IV_Loop = VFI.LI->getLoopFor(llvm::dyn_cast<llvm::Instruction>(IV)->getParent());
          if (L == IV_Loop)
            continue; // current loop IV

          int distance = 0;
          const llvm::Loop* Cur = L;
          while (Cur && Cur != IV_Loop) {
            ++distance;
            Cur = Cur->getParentLoop();
          }
          if (distance > maxDistance) {
            maxDistance = distance;
            maxD_L = Cur;
          }
        }

        // currently handle just only one getParentLoop
        if (maxDistance == 2 || (maxD_L && maxD_L->getParentLoop())) {

          if (auto* latch = L->getLoopLatch()) {
            auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
            std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildAffineFlattingRecurrence(VRI, Store);
            if (RR) {
              VRI.RR = RR;
              solvedRR.push_back(VRI.root);
              LLVM_DEBUG(tda::log() << "recognized (temporal) " << RR->toString() << " \n\n");
            }
          }
          return true;
        }

        // LAST CASE: A[i] = A[i - 1] + C

        int64_t StoreOff = 0;
        int64_t LoadOff = 0;

        const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
        const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
        if (!StoreIV || !LoadIV || StoreIV != LoadIV)
          return false;

        const int64_t delta = StoreOff - LoadOff;
        if (std::abs(delta) != 1)
          return false;

        if (auto* latch = L->getLoopLatch()) {
          auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
          std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildAffineStoreRecurrence(VRI, Store);
          if (RR) {
            VRI.RR = RR;
            solvedRR.push_back(VRI.root);
            LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
          }
        }
      }
  }
  return true;
}

bool ModuleInterpreter::isDeltaAffineRecurrence(VRARecurrenceInfo& VRI) {
  if (VRI.kind != VRAInspectionKind::REC)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as delta affine recurrence... \n");

  if (!VRI.innerRR)
    return false;

  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VRI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(VRI.root)) {

    bool isSolvable = VRI.chain.size() > 0;
    for (const auto* RRNode : VRI.chain) {
      if (VFI.RRs.count(RRNode)) {
        isSolvable &= VFI.RRs[RRNode].lastRange ? true : false;
        if (VFI.RRs[RRNode].lastRange && !isa<AffineRangedRecurrence>(VFI.RRs[RRNode].RR)
            && !isa<AffineFlattenedRangedRecurrence>(VFI.RRs[RRNode].RR)
            && !isa<AffineCrossingRangedRecurrence>(VFI.RRs[RRNode].RR))
          return false;
        continue;
      }

      if (!isAffineBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);

      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VRI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VRI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    if (auto* latch = L->getLoopLatch()) {
      auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
      std::shared_ptr<RangedRecurrence> RR =
        LatchAnalyzer->buildDeltaAffinePHIRecurrence(VRI, PN, &VFI.RRs[VRI.innerRR]);
      if (RR) {
        VRI.RR = RR;
        solvedRR.push_back(VRI.root);
        LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
      }
    }
  }

  return true;
}

bool ModuleInterpreter::isDeltaGeometricRecurrence(VRARecurrenceInfo& VRI) {
  if (VRI.kind != VRAInspectionKind::REC)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as delta geometric recurrence... \n");

  if (!VRI.innerRR)
    return false;

  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VRI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(VRI.root)) {

    bool isSolvable = VRI.chain.size() > 0;
    for (const auto* RRNode : VRI.chain) {
      if (VFI.RRs.count(RRNode)) {
        isSolvable &= VFI.RRs[RRNode].lastRange ? true : false;
        if (VFI.RRs[RRNode].lastRange && !isa<GeometricRangedRecurrence>(VFI.RRs[RRNode].RR)
            && !isa<GeometricFlattenedRangedRecurrence>(VFI.RRs[RRNode].RR)
            && !isa<GeometricCrossingRangedRecurrence>(VFI.RRs[RRNode].RR))
          return false;
        continue;
      }

      if (!isGeometricBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);

      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VRI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VRI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    if (auto* latch = L->getLoopLatch()) {
      auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
      std::shared_ptr<RangedRecurrence> RR =
        LatchAnalyzer->buildDeltaGeometricPHIRecurrence(VRI, PN, &VFI.RRs[VRI.innerRR]);
      if (RR) {
        VRI.RR = RR;
        solvedRR.push_back(VRI.root);
        LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
      }
    }
  }

  return true;
}

bool ModuleInterpreter::isCrossingAffineRecurrence(VRAAssignationInfo& VAI) {
  if (VAI.kind != VRAInspectionKind::ASSIGN || VAI.chain.size() == 0)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as crossing affine recurrence... \n");

  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VAI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (llvm::dyn_cast<llvm::PHINode>(VAI.root)) {
    // todo: add if needed
  }
  else if (const auto* Store = llvm::dyn_cast<llvm::StoreInst>(VAI.root)) {
    if (VAI.loads.size() == 0)
      return false;

    bool isSolvable = true;
    for (const auto* RRNode : VAI.chain) {
      if (!isAffineBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VAI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VAI);
    }

    // extra check for crossing: check that all loads of the current VAI on base X refers to the same base Y and that
    // base Y has another VAI which has all loads with the same base X
    const llvm::Value* curLoad = nullptr;
    const llvm::LoadInst* curLoadInst = nullptr;
    for (const auto* loadPtr : VAI.loads) {
      if (!curLoad)
        curLoad = getBaseMemoryObject(loadPtr->getPointerOperand());
      else if (getBaseMemoryObject(loadPtr->getPointerOperand()) != curLoad)
        return false;
      curLoadInst = loadPtr;
    }

    // check delta store/load = 1
    const Value* StoreIdx = getIndexOperand(Store->getPointerOperand());
    const Value* LoadIdx = getIndexOperand(dyn_cast<LoadInst>(curLoadInst)->getPointerOperand());

    int64_t StoreOff = 0;
    int64_t LoadOff = 0;

    const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
    const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
    if (!StoreIV || !LoadIV || StoreIV != LoadIV)
      return false;

    const int64_t delta = StoreOff - LoadOff;
    if (std::abs(delta) != 1)
      return false;

    const llvm::Value* complementaryStore = nullptr;
    for (auto [root, AS] : VFI.ASs) {
      if (isa<PHINode>(root) || root == VAI.root)
        continue;

      const llvm::Value* otherLoad = nullptr;
      const llvm::LoadInst* otherLoadInst = nullptr;
      for (const auto* loadPtr : AS.loads) {
        if (!otherLoad)
          otherLoad = getBaseMemoryObject(loadPtr->getPointerOperand());
        else if (getBaseMemoryObject(loadPtr->getPointerOperand()) != otherLoad)
          return false;
        otherLoadInst = loadPtr;
      }

      if (!otherLoad || otherLoad != getBaseMemoryObject(Store->getPointerOperand()))
        continue;

      const auto* ComplementaryInstr = llvm::dyn_cast<llvm::Instruction>(root);
      if (VFI.LI->getLoopFor(ComplementaryInstr->getParent()) != L)
        continue;

      // check delta store/load = 1
      const Value* StoreIdx = getIndexOperand(dyn_cast<StoreInst>(root)->getPointerOperand());
      const Value* LoadIdx = getIndexOperand(dyn_cast<LoadInst>(otherLoadInst)->getPointerOperand());

      int64_t StoreOff = 0;
      int64_t LoadOff = 0;

      const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
      const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
      if (!StoreIV || !LoadIV || StoreIV != LoadIV)
        return false;

      const int64_t delta = StoreOff - LoadOff;
      if (std::abs(delta) != 1)
        return false;

      complementaryStore = root;
    }

    if (!complementaryStore)
      return false;

    // check dependency also for complementary
    for (const auto* RRNode : VFI.ASs[complementaryStore].chain) {
      if (!isAffineBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VAI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VAI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    if (auto* latch = L->getLoopLatch()) {
      auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
      auto RRCoupled = LatchAnalyzer->buildStoreCrossingAffineRecurrence(
        isBefore(VAI.root, complementaryStore) ? VFI.ASs[complementaryStore] : VAI,
        isBefore(VAI.root, complementaryStore) ? VAI : VFI.ASs[complementaryStore]);
      if (RRCoupled.first && RRCoupled.second) {
        VAI.RR = RRCoupled.first; // to mark as resolved

        VRARecurrenceInfo VRA1(VAI.root);
        VRA1.kind = VRAInspectionKind::REC;
        VRA1.RR = RRCoupled.first;

        VRARecurrenceInfo VRA2(complementaryStore);
        VRA2.kind = VRAInspectionKind::REC;
        VRA2.RR = RRCoupled.second;

        remainingUnsolvedRR += 2;
        VFI.addRecurrenceInfo(VRA1);
        VFI.addRecurrenceInfo(VRA2);

        solvedRR.push_back(VRA1.root);
        solvedRR.push_back(VRA2.root);

        LLVM_DEBUG(tda::log() << "recognized " << VRA1.RR->toString() << " \n");
        LLVM_DEBUG(tda::log() << "recognized " << VRA2.RR->toString() << " \n\n");
      }
    }
  }
  return true;
}

bool ModuleInterpreter::isCrossingGeometricRecurrence(VRAAssignationInfo& VAI) {
  if (VAI.kind != VRAInspectionKind::ASSIGN || VAI.chain.size() == 0)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as crossing geometric recurrence... \n");

  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VAI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (llvm::dyn_cast<llvm::PHINode>(VAI.root)) {
    // todo if needed
  }
  else if (const auto* Store = llvm::dyn_cast<llvm::StoreInst>(VAI.root)) {
    if (VAI.loads.size() == 0)
      return false;

    bool isSolvable = true;
    for (const auto* RRNode : VAI.chain) {
      if (!isGeometricBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VAI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VAI);
    }

    // extra check for crossing: check that all loads of the current VAI on base X refers to the same base Y and that
    // base Y has another VAI which has all loads with the same base X
    const llvm::Value* curLoad = nullptr;
    const llvm::LoadInst* curLoadInst = nullptr;
    for (const auto* loadPtr : VAI.loads) {
      if (!curLoad)
        curLoad = getBaseMemoryObject(loadPtr->getPointerOperand());
      else if (getBaseMemoryObject(loadPtr->getPointerOperand()) != curLoad)
        return false;
      curLoadInst = loadPtr;
    }

    // check delta store/load = 1
    const Value* StoreIdx = getIndexOperand(Store->getPointerOperand());
    const Value* LoadIdx = getIndexOperand(dyn_cast<LoadInst>(curLoadInst)->getPointerOperand());

    int64_t StoreOff = 0;
    int64_t LoadOff = 0;

    const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
    const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
    if (!StoreIV || !LoadIV || StoreIV != LoadIV)
      return false;

    const int64_t delta = StoreOff - LoadOff;
    if (std::abs(delta) != 1)
      return false;

    const llvm::Value* complementaryStore = nullptr;
    for (auto [root, AS] : VFI.ASs) {
      if (isa<PHINode>(root) || root == VAI.root)
        continue;

      const llvm::Value* otherLoad = nullptr;
      const llvm::LoadInst* otherLoadInst = nullptr;
      for (const auto* loadPtr : AS.loads) {
        if (!otherLoad)
          otherLoad = getBaseMemoryObject(loadPtr->getPointerOperand());
        else if (getBaseMemoryObject(loadPtr->getPointerOperand()) != otherLoad)
          return false;
        otherLoadInst = loadPtr;
      }

      if (!otherLoad || otherLoad != getBaseMemoryObject(Store->getPointerOperand()))
        continue;

      const auto* ComplementaryInstr = llvm::dyn_cast<llvm::Instruction>(root);
      if (VFI.LI->getLoopFor(ComplementaryInstr->getParent()) != L)
        continue;

      // check delta store/load = 1
      const Value* StoreIdx = getIndexOperand(dyn_cast<StoreInst>(root)->getPointerOperand());
      const Value* LoadIdx = getIndexOperand(dyn_cast<LoadInst>(otherLoadInst)->getPointerOperand());

      int64_t StoreOff = 0;
      int64_t LoadOff = 0;

      const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
      const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
      if (!StoreIV || !LoadIV || StoreIV != LoadIV)
        return false;

      const int64_t delta = StoreOff - LoadOff;
      if (std::abs(delta) != 1)
        return false;

      complementaryStore = root;
    }

    if (!complementaryStore)
      return false;

    // check dependency also for complementary
    for (const auto* RRNode : VFI.ASs[complementaryStore].chain) {
      if (!isGeometricBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VAI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VAI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    if (auto* latch = L->getLoopLatch()) {
      auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
      auto RRCoupled = LatchAnalyzer->buildStoreCrossingGeometricRecurrence(
        isBefore(VAI.root, complementaryStore) ? VFI.ASs[complementaryStore] : VAI,
        isBefore(VAI.root, complementaryStore) ? VAI : VFI.ASs[complementaryStore]);
      if (RRCoupled.first && RRCoupled.second) {
        VAI.RR = RRCoupled.first; // to mark as resolved

        VRARecurrenceInfo VRA1(VAI.root);
        VRA1.kind = VRAInspectionKind::REC;
        VRA1.RR = RRCoupled.first;

        VRARecurrenceInfo VRA2(complementaryStore);
        VRA2.kind = VRAInspectionKind::REC;
        VRA2.RR = RRCoupled.second;

        remainingUnsolvedRR += 2;
        VFI.addRecurrenceInfo(VRA1);
        VFI.addRecurrenceInfo(VRA2);

        solvedRR.push_back(VRA1.root);
        solvedRR.push_back(VRA2.root);

        LLVM_DEBUG(tda::log() << "recognized " << VRA1.RR->toString() << " \n");
        LLVM_DEBUG(tda::log() << "recognized " << VRA2.RR->toString() << " \n\n");
      }
    }
  }
  return true;
}

std::shared_ptr<Range> ModuleInterpreter::getLastStoredRange(const llvm::Value* BaseStore) {
  std::shared_ptr<Range> joinedR = nullptr;
  for (auto [_, FN] : FNs) {
    for (auto [R, RR] : FN.RRs) {
      if (auto Store = dyn_cast<StoreInst>(R)) {
        if (getBaseMemoryObject(Store->getPointerOperand()) != BaseStore || !RR.lastRange)
          continue;
        if (!joinedR)
          joinedR = RR.lastRange;
        else
          joinedR = joinedR->join(RR.lastRange);
      }
    }
  }
  return joinedR;
}

bool ModuleInterpreter::isGeometricRecurrence(VRARecurrenceInfo& VRI) {
  if (VRI.kind != VRAInspectionKind::REC)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as geometric recurrence... \n");

  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VRI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (const auto* PN = llvm::dyn_cast<llvm::PHINode>(VRI.root)) {

    bool isSolvable = VRI.chain.size() > 0;
    for (const auto* RRNode : VRI.chain) {
      if (!isGeometricBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VRI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VRI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    // pattern acc *= a[i];
    if (VRI.loadHigherDim) {
      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildPHIGeometricFlattingRecurrence(VRI, PN);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
        }
      }
    }

    // pattern acc *= x;
    else if (auto* latch = L->getLoopLatch()) {
      auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
      std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildGeometricPHIRecurrence(PN);
      if (RR) {
        VRI.RR = RR;
        solvedRR.push_back(VRI.root);
        LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
      }
    }
  }
  else if (const auto* Store = llvm::dyn_cast<llvm::StoreInst>(VRI.root)) {

    bool isSolvable = VRI.chain.size() > 0;
    for (const auto* RRNode : VRI.chain) {
      if (!isGeometricBinaryOp(RRNode))
        return false;
      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VRI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VRI);
    }

    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    // pattern a[i] *= b[i][j];
    if (VRI.loadHigherDim) {
      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildGeometricFlattingRecurrence(VRI, Store);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
        }
      }
    }

    else if (getBaseMemoryObject(Store->getPointerOperand())
             == getBaseMemoryObject(VRI.loadJunction->getPointerOperand())) {
      const Value* StoreIdx = getIndexOperand(Store->getPointerOperand());
      const Value* LoadIdx = getIndexOperand(VRI.loadJunction->getPointerOperand());

      // CASE loop extra beyond IV: for (k) A[i] = A[i] * C
      int maxDistance = 0;
      const llvm::Loop* maxD_L;
      auto IVs = getInductionFromLoad(VRI.loadJunction, VFI.LI);
      for (auto IV : IVs) {
        auto IV_Loop = VFI.LI->getLoopFor(llvm::dyn_cast<llvm::Instruction>(IV)->getParent());
        if (L == IV_Loop)
          continue; // current loop IV

        int distance = 0;
        const llvm::Loop* Cur = L;
        while (Cur && Cur != IV_Loop) {
          ++distance;
          Cur = Cur->getParentLoop();
        }
        if (distance > maxDistance) {
          maxDistance = distance;
          maxD_L = Cur;
        }
      }

      // currently handle just only one getParentLoop
      if (maxDistance == 2 || (maxD_L && maxD_L->getParentLoop())) {

        if (auto* latch = L->getLoopLatch()) {
          auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
          std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildGeometricFlattingRecurrence(VRI, Store);
          if (RR) {
            VRI.RR = RR;
            solvedRR.push_back(VRI.root);
            LLVM_DEBUG(tda::log() << "recognized (temporal) " << RR->toString() << " \n\n");
          }
        }
        return true;
      }

      int64_t StoreOff = 0;
      int64_t LoadOff = 0;

      const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
      const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
      if (!StoreIV || !LoadIV || StoreIV != LoadIV)
        return false;

      const int64_t delta = StoreOff - LoadOff;
      if (std::abs(delta) != 1)
        return false;

      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildGeometricStoreRecurrence(VRI, Store);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
        }
      }
    }
  }
  return true;
}

const Value* ModuleInterpreter::matchIVOffset(VRAFunctionInfo VFI, const Value* Idx, int64_t& Offset, llvm::Loop* L) {

  Offset = 0;
  const Value* Cur = stripCasts(Idx);
  int64_t Acc = 0;
  int64_t ScaleNum = 1; // numerator of the IV coefficient
  int64_t ScaleDen = 1; // denominator of the IV coefficient

  const auto getConstInt = [](const Value* V) { return dyn_cast<ConstantInt>(V); };
  const auto mulNoOverflow = [](int64_t A, int64_t B, int64_t& Res) {
    __int128 Prod = static_cast<__int128>(A) * static_cast<__int128>(B);
    if (Prod > std::numeric_limits<int64_t>::max() || Prod < std::numeric_limits<int64_t>::min())
      return false;
    Res = static_cast<int64_t>(Prod);
    return true;
  };

  while (true) {
    Cur = stripCasts(Cur);
    const auto* BO = dyn_cast<BinaryOperator>(Cur);
    if (!BO)
      break;

    const unsigned Opc = BO->getOpcode();

    if (Opc == llvm::Instruction::Add || Opc == llvm::Instruction::Sub) {
      const ConstantInt* CI = nullptr;
      const Value* Next = nullptr;

      if ((CI = getConstInt(BO->getOperand(1))))
        Next = BO->getOperand(0);
      else if (Opc == llvm::Instruction::Add && (CI = getConstInt(BO->getOperand(0))))
        Next = BO->getOperand(1);

      if (!CI)
        break;

      int64_t Delta = 0;
      const int64_t C = CI->getSExtValue();
      if (!mulNoOverflow(ScaleNum, Opc == llvm::Instruction::Sub ? -C : C, Delta))
        break;
      Acc += Delta;
      Cur = Next;
      continue;
    }

    if (Opc == llvm::Instruction::Mul) {
      const ConstantInt* CI = getConstInt(BO->getOperand(1));
      const Value* Next = BO->getOperand(0);

      if (!CI) {
        CI = getConstInt(BO->getOperand(0));
        Next = BO->getOperand(1);
      }

      if (!CI)
        break;

      const int64_t Factor = CI->getSExtValue();
      if (Factor == 0)
        break;

      int64_t NewScale = 0;
      if (!mulNoOverflow(ScaleNum, Factor, NewScale))
        break;
      ScaleNum = NewScale;
      Cur = Next;
      continue;
    }

    if (Opc == llvm::Instruction::UDiv || Opc == llvm::Instruction::SDiv) {
      const auto* CI = getConstInt(BO->getOperand(1));
      if (!CI)
        break;

      const int64_t Divisor = CI->getSExtValue();
      if (Divisor == 0)
        break;

      int64_t NewAcc = 0;
      int64_t NewDen = 0;
      if (!mulNoOverflow(Acc, Divisor, NewAcc))
        break;
      if (!mulNoOverflow(ScaleDen, Divisor, NewDen))
        break;

      Acc = NewAcc;
      ScaleDen = NewDen;
      Cur = BO->getOperand(0);
      continue;
    }

    break;
  }

  Cur = stripCasts(Cur);

  if (ScaleDen < 0) {
    ScaleDen = -ScaleDen;
    ScaleNum = -ScaleNum;
    Acc = -Acc;
  }

  if ((ScaleNum % ScaleDen) != 0 || (Acc % ScaleDen) != 0)
    return nullptr;

  if (const auto* PHI = dyn_cast<PHINode>(Cur)) {
    if (VFI.loops[L].InductionVariable == const_cast<Value*>(Cur) && VFI.LI->getLoopFor(PHI->getParent()) == L) {
      Offset = Acc / ScaleDen;
      return PHI;
    }
  }

  return nullptr;
}

bool ModuleInterpreter::isFakeRecurrence(VRARecurrenceInfo& VRI) {
  if (VRI.kind != VRAInspectionKind::REC)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as fake recurrence... \n");

  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VRI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (const auto* Store = llvm::dyn_cast<llvm::StoreInst>(VRI.root)) {

    // main path analyzing
    if (getBaseMemoryObject(Store->getPointerOperand()) == getBaseMemoryObject(VRI.loadJunction->getPointerOperand())) {

      const Value* StoreIdx = getIndexOperand(Store->getPointerOperand());
      const Value* LoadIdx = getIndexOperand(VRI.loadJunction->getPointerOperand());

      int64_t StoreOff = 0;
      int64_t LoadOff = 0;

      const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
      const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
      if (!StoreIV || !LoadIV || StoreIV != LoadIV)
        return false;

      const int64_t delta = StoreOff - LoadOff;
      if (std::abs(delta) != 0)
        return false;

      // sec path analyzing...
      if (!isSolvableDependenceTreeBackwark(Store->getValueOperand(), L, VRI)) {
        LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvale yet: it depends on other unsolved recurrences\n");
        return true;
      }

      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];
        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildFakeStoreRecurrence(VRI, Store);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "\t\t\trecognized " << RR->toString() << "\n");
        }
      }
      return true;
    }
  }
  return false;
}

bool ModuleInterpreter::isLinearRecurrence(VRARecurrenceInfo& VRI) {
  if (VRI.kind != VRAInspectionKind::REC)
    return false;
  LLVM_DEBUG(tda::log() << "\t\ttry to recognize as linear recurrence... \n");

  const auto* InstrRoot = llvm::dyn_cast<llvm::Instruction>(VRI.root);
  llvm::Function* F = const_cast<llvm::Function*>(InstrRoot->getParent()->getParent());
  VRAFunctionInfo& VFI = FNs[F];
  llvm::Loop* L = VFI.LI->getLoopFor(InstrRoot->getParent());

  if (const auto* Store = llvm::dyn_cast<llvm::StoreInst>(VRI.root)) {

    bool isSolvable = VRI.chain.size() > 0;
    bool isFoundGeoOp = false;
    bool isFoundAffineOp = false;
    for (const auto* RRNode : VRI.chain) {

      // backwark analysis: no geo without previous affine (at least one), no affine when geo is found
      if (isAffineBinaryOp(RRNode) && isFoundGeoOp)
        return false;
      if (isGeometricBinaryOp(RRNode) && !isFoundAffineOp)
        return false;
      if (isAffineBinaryOp(RRNode))
        isFoundAffineOp = true;
      if (isGeometricBinaryOp(RRNode))
        isFoundGeoOp = true;

      if (!isAffineBinaryOp(RRNode) && !isGeometricBinaryOp(RRNode))
        return false;

      const auto* BO = llvm::dyn_cast<llvm::BinaryOperator>(RRNode);
      isSolvable &= isSolvableDependenceTreeBackwark(BO->getOperand(0), L, VRI)
                 && isSolvableDependenceTreeBackwark(BO->getOperand(1), L, VRI);
    }

    if (!isFoundAffineOp && !isFoundGeoOp)
      return false;
    if (!isSolvable) {
      LLVM_DEBUG(tda::log() << "\t\t\tRR is not solvable yet: it depends on other unsolved recurrences\n");
      return true;
    }

    // implementation of arr[i] = arr[i - 1] case
    if (getBaseMemoryObject(Store->getPointerOperand()) == getBaseMemoryObject(VRI.loadJunction->getPointerOperand())) {
      const Value* StoreIdx = getIndexOperand(Store->getPointerOperand());
      const Value* LoadIdx = getIndexOperand(VRI.loadJunction->getPointerOperand());

      int64_t StoreOff = 0;
      int64_t LoadOff = 0;

      const Value* StoreIV = matchIVOffset(VFI, StoreIdx, StoreOff, L);
      const Value* LoadIV = matchIVOffset(VFI, LoadIdx, LoadOff, L);
      if (!StoreIV || !LoadIV || StoreIV != LoadIV)
        return false;

      const int64_t delta = StoreOff - LoadOff;
      if (std::abs(delta) != 1)
        return false;

      if (auto* latch = L->getLoopLatch()) {
        auto LatchAnalyzer = VFI.scope.BBAnalyzers[latch];

        std::shared_ptr<RangedRecurrence> RR = LatchAnalyzer->buildLinearRecurrence(VRI, Store);
        if (RR) {
          VRI.RR = RR;
          solvedRR.push_back(VRI.root);
          LLVM_DEBUG(tda::log() << "recognized " << RR->toString() << " \n\n");
          return true;
        }
      }
    }
  }
  return false;
}

//==================================================================================================
//======================= TRIP COUNT METHODS =======================================================
//==================================================================================================

void ModuleInterpreter::tripCount() {
  solvedTC = 0;

  for (auto& Entry : FNs) {
    llvm::Function* F = Entry.first;
    auto& VFI = Entry.second;

    for (auto& LEntry : VFI.loops) {
      const llvm::Loop* L = LEntry.first;
      auto& VLI = LEntry.second;

      if (VLI.TripCount > 0 || !VLI.InductionVariable)
        continue;

      auto It = VFI.RRs.find(VLI.InductionVariable);
      if (It == VFI.RRs.end()) {
        LLVM_DEBUG(tda::log() << "[VRA] >> [TripCount] >> FN[" << F->getName() << "] - RR entry NOT FOUND for IV "
                              << VLI.InductionVariable->getName() << " ptr=" << (const void*) VLI.InductionVariable
                              << "\n");
        continue;
      }

      auto& VRI = It->second;
      if (!VRI.RR) {
        LLVM_DEBUG(tda::log() << "[VRA] >> [TripCount] >> FN[" << F->getName()
                              << "] - RR entry found but RR==nullptr for IV " << VLI.InductionVariable->getName()
                              << " ptr=" << (const void*) VLI.InductionVariable << "\n");
        continue;
      }

      if (auto* ARR = dyn_cast<AffineRangedRecurrence>(VRI.RR.get())) {
        const auto& Start = ARR->getStart();
        const auto& Step = ARR->getStep();
        if (!Start || !Step)
          continue;

        if (!VLI.exitCmp) {
          // looking for block with exit successor, then get cmp with handle branch
          for (auto BB : VLI.bbFlow) {
            for (auto& I : *BB) {
              if (auto BR = dyn_cast<BranchInst>(&I)) {
                if (!BR->isConditional())
                  continue;
                for (unsigned succ_idx = 0; succ_idx < BR->getNumSuccessors(); succ_idx++) {
                  if (isLoopExit(L, BR->getSuccessor(succ_idx))) {
                    auto cmp = BR->getCondition();
                    VLI.exitCmp = dyn_cast<CmpInst>(cmp);
                  }
                }
              }
            }
          }
        }

        if (!VLI.exitCmp)
          continue;

        CmpInst::Predicate Pred = VLI.exitCmp->getPredicate();
        llvm::Value* invariantOp;
        if (VLI.isInvariant(VLI.exitCmp->getOperand(0))) {
          Pred = CmpInst::getSwappedPredicate(Pred);
          invariantOp = VLI.exitCmp->getOperand(0);
        }
        else if (VLI.isInvariant(VLI.exitCmp->getOperand(1))) {
          invariantOp = VLI.exitCmp->getOperand(1);
        }
        else
          continue;

        const bool isLT = Pred == CmpInst::ICMP_SLT || Pred == CmpInst::ICMP_ULT || Pred == CmpInst::FCMP_OLT
                       || Pred == CmpInst::FCMP_ULT;
        const bool isLE = Pred == CmpInst::ICMP_SLE || Pred == CmpInst::ICMP_ULE || Pred == CmpInst::FCMP_OLE
                       || Pred == CmpInst::FCMP_ULE;
        const bool isGT = Pred == CmpInst::ICMP_SGT || Pred == CmpInst::ICMP_UGT || Pred == CmpInst::FCMP_OGT
                       || Pred == CmpInst::FCMP_UGT;
        const bool isGE = Pred == CmpInst::ICMP_SGE || Pred == CmpInst::ICMP_UGE || Pred == CmpInst::FCMP_OGE
                       || Pred == CmpInst::FCMP_UGE;

        uint64_t TripC = 0;
        bool computed = false;

        auto InvariantStore = std::static_pointer_cast<VRAnalyzer>(getStoreForValue(invariantOp));

        if (!InvariantStore)
          continue;
        auto InvariantRange = InvariantStore->getBBRange(invariantOp);

        if (!isa<Constant>(invariantOp)) {
          const auto* I = llvm::dyn_cast<llvm::Instruction>(invariantOp);
          if (VFI.LI && VFI.RRs.count(I)) {
            if (!isSolvableDependenceTreeBackwark(invariantOp, VFI.LI->getLoopFor(I->getParent()), VFI.RRs[I]))
              continue;
          }
        }

        if ((isLT || isLE) && Step->min > 0.0) {
          const double num = InvariantRange->max - Start->min;
          const double den = Step->min;
          if (den > 0.0 && std::isfinite(num)) {
            TripC = static_cast<uint64_t>(std::ceil(num / den));
            computed = true;
          }
        }
        else if ((isGT || isGE) && Step->max < 0.0) {
          const double num = Start->max - InvariantRange->min;
          const double den = -Step->max;
          if (den > 0.0 && std::isfinite(num)) {
            TripC = static_cast<uint64_t>(std::ceil(num / den));
            computed = true;
          }
        }
        else {
          LLVM_DEBUG(tda::log() << "todo: add here other heuristics\n");
        }

        if (computed && TripC > 0) {
          VLI.TripCount = TripC;
          VLI.Reason = TripCountReason::HeuristicFallback;
          LLVM_DEBUG(tda::log() << "[VRA] >> [TripCount] >> FN[" << F->getName() << "] - trip count for loop "
                                << VLI.L->getName() << " (affine growth) = " << TripC << "\n");
          solvedTC++;
        }
        else if (isFallback) {
          VLI.TripCount = 1; // cerotto: eseguire comunque una iterazione per evitare null ranges
          VLI.Reason = TripCountReason::Unknown;
          LLVM_DEBUG(tda::log() << "[VRA] >> [TripCount] >> FN[" << F->getName() << "] - trip count for loop "
                                << VLI.L->getName() << " (FALLBACK) = 1\n");
          solvedTC++;
        }
      }
    }
  }
}

//==================================================================================================
//======================= PROPAGATE METHODS ========================================================
//==================================================================================================

void ModuleInterpreter::walk(llvm::Loop* L) {

  VRAFunctionInfo& VFI = FNs[curFn.back()];
  llvm::SmallVector<llvm::BasicBlock*> curFlow =
    L && FNs[curFn.back()].loops.count(L) ? FNs[curFn.back()].loops[L].bbFlow : FNs[curFn.back()].bbFlow;

  for (auto it = curFlow.begin(); it != curFlow.end(); ++it) {
    const llvm::BasicBlock* curBlock = *it;
    llvm::Loop* curLoop = VFI.LI ? VFI.LI->getLoopFor(curBlock) : nullptr;

    if ((!L && curLoop && curBlock == curLoop->getHeader())
        || (L && curBlock == curLoop->getHeader() && L != curLoop)) {
      walk(curLoop);
      continue;
    }

    LLVM_DEBUG(
      {
        auto& dbg = tda::log();
        dbg << "\n[VRA] >> [Propagation] >> FN[" << curFn.back()->getName() << "] - walking on ";
        if (curLoop)
          dbg << " loop " << curLoop->getName() << ",";
        dbg << " block " << curBlock->getName()
            << " \n----------------------------------------------------------------------------------------------------"
               "--\n";
      });

    auto CAIt = VFI.scope.BBAnalyzers.find(curBlock);
    assert(CAIt != VFI.scope.BBAnalyzers.end());
    std::shared_ptr<CodeAnalyzer> CurAnalyzer = CAIt->second;

    for (const llvm::Instruction& CI : *curBlock) {
      llvm::Instruction& I = const_cast<llvm::Instruction&>(CI);

      if (FNs[curFn.back()].RRs.count(&I) && curLoop) {
        VRARecurrenceInfo& VRI = FNs[curFn.back()].RRs[&I];

        if (VRI.lastRange && VRI.lastRangeComputedAt >= FNs[curFn.back()].loops[curLoop].TripCount) {
          CurAnalyzer->retrieveSolvedRecurrence(&I, VFI.RRs[&I]);
        }
        else {
          if (!VRI.RR) {
            CurAnalyzer->analyzeInstruction(&I);
            LLVM_DEBUG(tda::log() << " STILL NOT SOLVED\n");
            continue;
          }
          CurAnalyzer->resolveRecurrence(VRI, FNs[curFn.back()].loops[curLoop].TripCount);
        }
      }
      else if (CurAnalyzer->requiresInterpretation(&I)) {
        resolveCall(CurAnalyzer, &I);
      }
      else {
        CurAnalyzer->analyzeInstruction(&I);
      }
    }

    auto nextIt = std::next(it);
    if (nextIt != curFlow.end())
      updateKnownSuccessorAnalyzer(CurAnalyzer, *nextIt, curBlock);
    else if (L) {
      // end flow but into the loop
      llvm::SmallVector<llvm::BasicBlock*, 4> exits;
      L->getExitBlocks(exits);

      llvm::SmallPtrSet<llvm::BasicBlock*, 32> visitedBlocks;
      visitedBlocks.insert(curFlow.begin(), curFlow.end());

      llvm::BasicBlock* firstEligible = nullptr;
      for (llvm::BasicBlock* EB : exits) {
        bool allPredVisited = true;
        for (llvm::BasicBlock* Pred : predecessors(EB)) {
          if (!visitedBlocks.count(Pred) && !L->contains(Pred)) {
            allPredVisited = false;
            break;
          }
        }

        if (allPredVisited) {
          firstEligible = EB;
          break;
        }
      }

      if (firstEligible) {
        LLVM_DEBUG(tda::log() << " end of loop " << L->getName() << ", enqueuing " << firstEligible->getName()
                              << " \n");
        updateKnownSuccessorAnalyzer(CurAnalyzer, firstEligible, curBlock);
      }
    }

    GlobalStore->convexMerge(*CurAnalyzer);
  }
}

void ModuleInterpreter::propagateFunction(llvm::Function* F, std::shared_ptr<AnalysisStore> FunctionStore) {
  LLVM_DEBUG(tda::log() << "\n\n[VRA] >> [Propagation] >> FN[" << F->getName() << "] - START PROPAGATION\n");

  VRAFunctionInfo& VFI = FNs[F];
  curFn.push_back(F);

  if (!FunctionStore) {
    LLVM_DEBUG(tda::log() << "\n\n[VRA] >> [Propagation] >> FN[" << F->getName() << "] - SCOPE RESET\n");
    FunctionStore = GlobalStore->newFnStore(*this);
    FNs[F].scope.BBAnalyzers.clear();
    FNs[F].scope.FunctionStore = nullptr;
    VFI.scope = FunctionScope(FunctionStore);
    VFI.scope.BBAnalyzers[&F->getEntryBlock()] = GlobalStore->newInstructionAnalyzer(*this);
  }
  else {
    if (FNs.count(F))
      FunctionStore = FNs[F].scope.FunctionStore;
  }
  walk();

  GlobalStore->convexMerge(*FunctionStore);
  curFn.pop_back();

  LLVM_DEBUG(tda::log() << "\n\n[VRA] >> [Propagation] >> FN[" << F->getName() << "] - PROPAGATION ENDED\n");
}

void ModuleInterpreter::propagate() {
  curFn.clear();

  // reset global scope
  GlobalStore = std::make_shared<VRAGlobalStore>();
  GlobalStore->harvestValueInfo(M);

  propagateFunction(EntryFn, nullptr);
}

void ModuleInterpreter::fallback() {

  for (auto& Entry : FNs) {
    auto& VFI = Entry.second;

    for (auto RREntry = VFI.RRs.begin(); RREntry != VFI.RRs.end();) {
      const llvm::Value* root = RREntry->first;
      auto& VRI = RREntry->second;

      if (VRI.RR) {
        RREntry++;
        continue; // already solved
      }

      auto Next = std::next(RREntry);
      VFI.RRs.erase(RREntry);
      RREntry = Next;
      LLVM_DEBUG(tda::log() << "fallback applied on RR " << printInstrName(root)
                            << " due to unrecognized or dependency unsolvable.\n");
      solvedRR.push_back(root);

      if (isa<PHINode>(root))
        phiFalledBack.push_back({Entry.first, root});
    }
  }
}

} // end of namespace taffo
