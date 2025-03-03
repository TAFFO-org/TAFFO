#include "CodeInterpreter.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include <Metadata.h>
#include <cassert>
#include <deque>

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

void CodeInterpreter::interpretFunction(llvm::Function *F,
                                        std::shared_ptr<AnalysisStore> FunctionStore)
{
  DEBUG_WITH_TYPE(GlobalStore->getLogger()->getDebugType(),
                  GlobalStore->getLogger()->logStartFunction(F));
  if (!FunctionStore) {
    FunctionStore = GlobalStore->newFunctionStore(*this);
  }
  Scopes.push_back(FunctionScope(FunctionStore));

  updateLoopInfo(F);
  retrieveLoopTripCount(F);

  llvm::BasicBlock *EntryBlock = &F->getEntryBlock();
  llvm::SmallPtrSet<llvm::BasicBlock *, 4U> VisitedSuccs;
  std::deque<llvm::BasicBlock *> Worklist;
  Worklist.push_back(EntryBlock);
  Scopes.back().EvalCount[EntryBlock] = 1U;
  Scopes.back().BBAnalyzers[EntryBlock] = GlobalStore->newCodeAnalyzer(*this);

  while (!Worklist.empty()) {
    llvm::BasicBlock *BB = Worklist.front();
    Worklist.pop_front();

    auto CAIt = Scopes.back().BBAnalyzers.find(BB);
    assert(CAIt != Scopes.back().BBAnalyzers.end());
    std::shared_ptr<CodeAnalyzer> CurAnalyzer = CAIt->second;

    DEBUG_WITH_TYPE(GlobalStore->getLogger()->getDebugType(),
                    GlobalStore->getLogger()->logBasicBlock(BB));
    for (llvm::Instruction &I : *BB) {
      if (CurAnalyzer->requiresInterpretation(&I)) {
        interpretCall(CurAnalyzer, &I);
      } else {
        CurAnalyzer->analyzeInstruction(&I);
      }
    }

    assert(Scopes.back().EvalCount[BB] > 0 && "Trying to evaluate block with 0 EvalCount.");
    --(Scopes.back().EvalCount[BB]);

    llvm::Instruction *Term = BB->getTerminator();
    VisitedSuccs.clear();
    for (unsigned NS = 0; NS < Term->getNumSuccessors(); ++NS) {
      llvm::BasicBlock *Succ = Term->getSuccessor(NS);

      // Needed just for terminators where the same successor appears twice
      if (VisitedSuccs.count(Succ)) {
        continue;
      } else {
        VisitedSuccs.insert(Succ);
      }

      if (followEdge(BB, Succ)) {
        Worklist.push_front(Succ);
      }
      // TODO: only propagate pathlocal info for better efficiency.
      updateSuccessorAnalyzer(CurAnalyzer, Term, NS);
    }

    GlobalStore->convexMerge(*CurAnalyzer);
  }

  GlobalStore->convexMerge(*FunctionStore);
  Scopes.pop_back();

  DEBUG_WITH_TYPE(GlobalStore->getLogger()->getDebugType(),
                  GlobalStore->getLogger()->logEndFunction(F));
}

std::shared_ptr<AnalysisStore>
CodeInterpreter::getStoreForValue(const llvm::Value *V) const
{
  assert(V && "Trying to get AnalysisStore for null value.");

  if (llvm::isa<llvm::Constant>(V))
    return GlobalStore;

  if (llvm::isa<llvm::Argument>(V)) {
    for (const FunctionScope &Scope : llvm::make_range(Scopes.rbegin(), Scopes.rend())) {
      if (Scope.FunctionStore->hasValue(V))
        return Scope.FunctionStore;
    }
  }

  if (const llvm::Instruction *I = llvm::dyn_cast<llvm::Instruction>(V)) {
    for (const FunctionScope &Scope : llvm::make_range(Scopes.rbegin(), Scopes.rend())) {
      auto BBAIt = Scope.BBAnalyzers.find(I->getParent());
      if (BBAIt != Scope.BBAnalyzers.end() && BBAIt->second->hasValue(I))
        return BBAIt->second;
    }
  }

  return nullptr;
}

bool CodeInterpreter::isLoopBackEdge(llvm::BasicBlock *Src, llvm::BasicBlock *Dst) const
{
  assert(LoopInfo);
  return LoopInfo->isLoopHeader(Dst) && getLoopForBackEdge(Src, Dst);
}

llvm::Loop *
CodeInterpreter::getLoopForBackEdge(llvm::BasicBlock *Src, llvm::BasicBlock *Dst) const
{
  assert(LoopInfo);
  llvm::Loop *L = LoopInfo->getLoopFor(Dst);
  while (L && !L->contains(Src))
    L = L->getParentLoop();

  return L;
}

bool CodeInterpreter::followEdge(llvm::BasicBlock *Src, llvm::BasicBlock *Dst)
{
  llvm::DenseMap<llvm::BasicBlock *, unsigned> &EvalCount = Scopes.back().EvalCount;
  // Don't follow edge if Dst has unvisited predecessors.
  unsigned SrcEC = EvalCount[Src];
  for (llvm::BasicBlock *Pred : predecessors(Dst)) {
    auto PredECIt = EvalCount.find(Pred);
    if ((PredECIt == EvalCount.end() || PredECIt->second > SrcEC) && !isLoopBackEdge(Pred, Dst))
      return false;
  }

  assert(LoopInfo);
  llvm::Loop *DstLoop = LoopInfo->getLoopFor(Dst);
  if (DstLoop && !DstLoop->contains(Src)) {
    // Entering new loop.
    assert(DstLoop->getHeader() == Dst && "Dst must be Loop header.");
    unsigned TripCount = 1U;
    if (llvm::BasicBlock *Latch = DstLoop->getLoopLatch()) {
      TripCount = LoopTripCount[Latch];
    }
    for (llvm::BasicBlock *LBB : DstLoop->blocks()) {
      EvalCount[LBB] = TripCount;
    }
    if (DstLoop->isLoopExiting(Dst)) {
      ++EvalCount[Dst];
    }
    return true;
  }
  llvm::Loop *SrcLoop = LoopInfo->getLoopFor(Src);
  if (SrcLoop) {
    if (SrcEC == 0U && SrcLoop->isLoopExiting(Src)) {
      // We are in the last evaluation of this loop.
      if (SrcLoop->contains(Dst)) {
        // We follow an internal edge only if it still has to be evaluated this time.
        return EvalCount[Dst] > 0;
      }
      // We can follow the exiting edges.
      EvalCount[Dst] = 1U;
      return true;
    }
    // If the loop has to be evaluated more times, we do not follow the exiting edges.
    return EvalCount[Dst] > 0 && SrcLoop->contains(Dst);
  }
  if (!SrcLoop && !DstLoop) {
    // There's no loop, just evaluate Dst once.
    EvalCount[Dst] = 1U;
  }
  return true;
}

void CodeInterpreter::updateSuccessorAnalyzer(std::shared_ptr<CodeAnalyzer> CurrentAnalyzer,
                                              llvm::Instruction *TermInstr,
                                              unsigned SuccIdx)
{
  llvm::DenseMap<llvm::BasicBlock *, std::shared_ptr<CodeAnalyzer>> &BBAnalyzers =
      Scopes.back().BBAnalyzers;
  llvm::BasicBlock *SuccBB = TermInstr->getSuccessor(SuccIdx);

  std::shared_ptr<CodeAnalyzer> SuccAnalyzer;
  auto SAIt = BBAnalyzers.find(SuccBB);
  if (SAIt == BBAnalyzers.end()) {
    SuccAnalyzer = CurrentAnalyzer->clone();
    BBAnalyzers[SuccBB] = SuccAnalyzer;
  } else {
    SuccAnalyzer = SAIt->second;
    SuccAnalyzer->convexMerge(*CurrentAnalyzer);
  }

  CurrentAnalyzer->setPathLocalInfo(SuccAnalyzer, TermInstr, SuccIdx);
}

void CodeInterpreter::interpretCall(std::shared_ptr<CodeAnalyzer> CurAnalyzer,
                                    llvm::Instruction *I)
{
  llvm::CallBase *CB = llvm::cast<llvm::CallBase>(I);
  llvm::Function *F = CB->getCalledFunction();
  if (!F || F->empty())
    return;

  if (!updateRecursionCount(F))
    return;

  std::shared_ptr<AnalysisStore> FunctionStore = GlobalStore->newFunctionStore(*this);

  CurAnalyzer->prepareForCall(I, FunctionStore);
  interpretFunction(F, FunctionStore);
  CurAnalyzer->returnFromCall(I, FunctionStore);

  updateLoopInfo(I->getFunction());
}

void CodeInterpreter::updateLoopInfo(llvm::Function *F)
{
  assert(F);
  auto &FAM = getMAM().getResult<llvm::FunctionAnalysisManagerModuleProxy>(*F->getParent()).getManager();
  LoopInfo = &(FAM.getResult<llvm::LoopAnalysis>(*F));
}

/// Get the latch condition instruction.
static llvm::Value *getLatchConditionInst(llvm::BasicBlock *BB)
{
  if (llvm::BranchInst *BI = llvm::dyn_cast_or_null<llvm::BranchInst>(BB->getTerminator()))
    if (BI->isConditional())
      return BI->getCondition();

  return nullptr;
}

static unsigned retrieveOMPExternalCallBoundaryTripCount(llvm::Function *F, llvm::Loop *L)
{
  auto branch = L->getLoopLatch();
  // Get the true latch because the default is not correct
  llvm::Value *cmpvalue = getLatchConditionInst(branch);
  while (cmpvalue == nullptr) {
    llvm::dbgs() << "branch = " << *branch << "\n";
    auto end_block = llvm::dyn_cast_or_null<llvm::BranchInst>(branch->getTerminator());
    if (end_block == nullptr) {
      // block terminated by something that is not a branch (e.g. invoke), bail out
      return 0;
    }
    branch = end_block->getSuccessor(0);
    cmpvalue = getLatchConditionInst(branch);
  }

  // loop condition should always be an ICmpInst
  llvm::ICmpInst *cmpinst = llvm::dyn_cast<llvm::ICmpInst>(cmpvalue);
  if (cmpinst == nullptr)
    return 0;

  auto second_operand = cmpinst->getOperand(1);
  // The second operand of the comparison should always be a load
  auto load = llvm::dyn_cast_or_null<llvm::LoadInst>(second_operand);
  if (load == nullptr)
    return 0;

  for (auto i = load->getOperand(0)->use_begin(); i != load->getOperand(0)->use_end(); i++) {
    // Search if it is used in one of the omp static init
    if (auto call = llvm::dyn_cast_or_null<llvm::CallInst>(i->getUser())) {
      if (call->getCalledFunction()->getName().find("__kmpc_for_static_init") == 0) {
        // If it is a omp loop search the constant value loaded in the ub
        for (auto s = load->getOperand(0)->use_begin(); s != load->getOperand(0)->use_end(); s++) {
          if (auto store = llvm::dyn_cast_or_null<llvm::StoreInst>(s->getUser())) {
            LLVM_DEBUG(store->dump());
            if (auto constant_value = llvm::dyn_cast_or_null<llvm::ConstantInt>(store->getOperand(0))) {
              return constant_value->getSExtValue() + 1;
            }
          }
        }
      }
    }
  }
  return 0;
}

void CodeInterpreter::retrieveLoopTripCount(llvm::Function *F)
{
  assert(LoopInfo && F);
  llvm::ScalarEvolution *SE = nullptr;
  for (llvm::Loop *L : LoopInfo->getLoopsInPreorder()) {
    if (llvm::BasicBlock *Latch = L->getLoopLatch()) {
      if (DefaultTripCount > 0U && MaxTripCount > 0U) {
        unsigned TripCount = 0U;
        // Get user supplied unroll count
        std::optional<unsigned> OUC =
            mdutils::MetadataManager::retrieveLoopUnrollCount(*L, LoopInfo);
        if (OUC.has_value()) {
          TripCount = OUC.value();
        } else {
          // Compute loop trip count
          if (!SE) {
            auto &FAM = getMAM().getResult<llvm::FunctionAnalysisManagerModuleProxy>(*F->getParent()).getManager();
            SE = &(FAM.getResult<llvm::ScalarEvolutionAnalysis>(*F));
          }
          TripCount = SE->getSmallConstantTripCount(L);

          // Handle OMP load of boundary with external call
          if (TripCount == 0)
            TripCount = retrieveOMPExternalCallBoundaryTripCount(F, L);
        }
        TripCount = (TripCount > 0U) ? TripCount : DefaultTripCount;
        LoopTripCount[Latch] = (TripCount > MaxTripCount) ? MaxTripCount : TripCount;
      } else {
        // Loop unrolling disabled
        LoopTripCount[Latch] = 1U;
      }
    }
  }
}

bool CodeInterpreter::updateRecursionCount(llvm::Function *F)
{
  auto RCIt = RecursionCount.find(F);
  if (RCIt == RecursionCount.end()) {
    unsigned FromMD = mdutils::MetadataManager::retrieveMaxRecursionCount(*F);
    if (FromMD > 0)
      --FromMD;

    RecursionCount[F] = FromMD;
    return true;
  }
  unsigned &Remaining = RCIt->second;
  if (Remaining > 0) {
    --Remaining;
    return true;
  }
  return false;
}

} // end namespace taffo
