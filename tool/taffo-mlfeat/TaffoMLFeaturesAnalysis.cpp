#include "Metadata.h"
#include "TaffoMLFeaturesAnalysis.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace llvm;

cl::opt<bool> CountAll("countall",
                       cl::value_desc("countall"),
                       cl::desc("Perform analysis of the entire code (not just the instrumented parts)"),
                       cl::init(false));

#define MD_COUNT_INSTR "tmlfa.count"

struct MLFeatureBlock {
  BasicBlock* entry;
  std::set<BasicBlock*> contents;

  int depth = 0;
  int tripCount = 0;

  InstructionMix imix;
  int maxAllocSize = 0;
  int numAnnotatedInstr = 0;
  // 12 is a completely random guess on the average length
  // of a modern cpu's pipeline
  int minDist_mul = 12;
  int minDist_div = 12;
  int minDist_callBase = 12;

  // used for sorting
  bool operator<(const MLFeatureBlock& other) const {
    return depth != other.depth ? depth < other.depth : tripCount < other.tripCount;
  }
};

struct MLFeatureBlockComputationState {
  int lastDist_mul = std::numeric_limits<int>::max();
  int lastDist_div = std::numeric_limits<int>::max();
  int lastDist_callBase = std::numeric_limits<int>::max();
};

char TaffoMLFeatureAnalysisPass::ID = 0;

void setCountEnabledForInstruction(Instruction* instr, bool enabled) {
  MDNode* existing = instr->getMetadata(MD_COUNT_INSTR);
  if (existing && !enabled) {
    instr->setMetadata(MD_COUNT_INSTR, nullptr);
  }
  else if (!existing && enabled) {
    ConstantInt* booltrue = ConstantInt::get(Type::getInt1Ty(instr->getContext()), 1);
    ConstantAsMetadata* cmd = ConstantAsMetadata::get(booltrue);
    MDNode* newmd = MDTuple::get(instr->getContext(), {cmd});
    instr->setMetadata(MD_COUNT_INSTR, newmd);
  }
}

bool getCountEnabledForInstruction(Instruction* instr) {
  MDNode* existing = instr->getMetadata(MD_COUNT_INSTR);
  return !!existing;
}

void computeBasicBlockStats(MLFeatureBlock& b, BasicBlock* bb, MLFeatureBlockComputationState& state) {
  for (Instruction& i : *bb) {
    if (isSkippableInstruction(&i))
      continue;

    if (AllocaInst* alloca = dyn_cast<AllocaInst>(&i)) {
      const DataLayout& dl = alloca->getModule()->getDataLayout();
      std::optional<llvm::TypeSize> size = alloca->getAllocationSizeInBits(dl);
      if (size.has_value())
        b.maxAllocSize = std::max(b.maxAllocSize, (int) ((*size) / 8));
    }
    else if (CallBase* call = dyn_cast<CallBase>(&i)) {
      if (Function* f = call->getCalledFunction()) {
        if (f->getName().equals("malloc")) {
          Value* nalloc = call->getArgOperand(0);
          if (ConstantInt* ci = dyn_cast<ConstantInt>(nalloc))
            b.maxAllocSize = std::max(b.maxAllocSize, (int) (ci->getSExtValue()));
        }
        else if (f->getName().equals("calloc")) {
          ConstantInt* count = dyn_cast<ConstantInt>(call->getArgOperand(0));
          ConstantInt* size = dyn_cast<ConstantInt>(call->getArgOperand(0));
          if (count && size)
            b.maxAllocSize = std::max(b.maxAllocSize, (int) (count->getSExtValue() * size->getSExtValue()));
        }
      }
    }

    if (!CountAll && !getCountEnabledForInstruction(&i))
      continue;

    b.imix.updateWithInstruction(&i);

    if (isa<CallBase>(&i)) {
      b.minDist_callBase = std::min(b.minDist_callBase, state.lastDist_callBase);
      state.lastDist_callBase = 0;
    }
    else
      state.lastDist_callBase = std::max(state.lastDist_callBase, state.lastDist_callBase + 1);

    if (i.getOpcode() == Instruction::Mul) {
      b.minDist_mul = std::min(b.minDist_mul, state.lastDist_mul);
      state.lastDist_mul = 0;
    }
    else
      state.lastDist_mul = std::max(state.lastDist_mul, state.lastDist_mul + 1);

    if (i.getOpcode() == Instruction::SDiv || i.getOpcode() == Instruction::UDiv) {
      b.minDist_div = std::min(b.minDist_div, state.lastDist_div);
      state.lastDist_div = 0;
    }
    else
      state.lastDist_div = std::max(state.lastDist_div, state.lastDist_div + 1);

    mdutils::MetadataManager& mm = mdutils::MetadataManager::getMetadataManager();
    mdutils::MDInfo* mdi = mm.retrieveMDInfo(&i);
    if (mdi)
      b.numAnnotatedInstr += !!(mdi->getEnableConversion());
  }
}

void computeBlockStats(MLFeatureBlock& b) {
  /* TODO:
   * (1) clique of b.contents & loop on cliques;
   * (2) handle branches with `state` sharing */
  for (BasicBlock* bb : b.contents) {
    MLFeatureBlockComputationState state;
    computeBasicBlockStats(b, bb, state);
  }
}

void computeEnabledInstructions(Function* f, DominatorTree& dom) {
  struct state {
    BasicBlock* bb;
    int nestingLevel;
  };
  std::deque<state> queue;
  queue.push_back({dom.getRoot(), 0});

  while (!queue.empty()) {
    state self = *(queue.begin());
    queue.pop_front();

    for (Instruction& inst : *self.bb) {
      int delim = isDelimiterInstruction(&inst);
      if (!delim) {
        setCountEnabledForInstruction(&inst, self.nestingLevel > 0);
      }
      else {
        self.nestingLevel += delim;
        setCountEnabledForInstruction(&inst, false);
      }
    }

    for (DomTreeNode* nexti : dom[self.bb]->children())
      if (nexti->getBlock() != self.bb)
        queue.push_back({nexti->getBlock(), self.nestingLevel});
  }
}

void TaffoMLFeatureAnalysisPass::getAnalysisUsage(AnalysisUsage& AU) const {
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<LoopInfoWrapperPass>();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
}

bool TaffoMLFeatureAnalysisPass::runOnFunction(Function& F) {
  if (!CountAll) {
    DominatorTreeWrapperPass& dtwp = Pass::getAnalysis<DominatorTreeWrapperPass>();
    computeEnabledInstructions(&F, dtwp.getDomTree());
  }

  LoopInfo& li = this->getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  /* compute the list of basic blocks that are outside any loop */
  std::set<BasicBlock*> allbbs;
  for (BasicBlock& bb : F)
    allbbs.insert(&bb);
  for (Loop* l : li.getLoopsInPreorder())
    for (BasicBlock* bb : l->blocks())
      allbbs.erase(bb);

  assert(allbbs.find(&F.getEntryBlock()) != allbbs.end() && "entry block of function is in a loop??");

  ScalarEvolution& SE = Pass::getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  SmallVector<Loop*, 4> loops = li.getLoopsInPreorder();
  int nfeat = loops.size() + 1;

  std::vector<MLFeatureBlock> features(nfeat);
  features[0].tripCount = 1;
  features[0].contents = allbbs;
  computeBlockStats(features[0]);

  int i = 1;
  for (Loop* l : loops) {
    for (BasicBlock* bb : l->blocks())
      features[i].contents.insert(bb);
    features[i].entry = l->getHeader();
    const SCEV* tripcnt = SE.getConstantMaxBackedgeTakenCount(l);
    if (const SCEVConstant* realtripcnt = dyn_cast<SCEVConstant>(tripcnt))
      features[i].tripCount = realtripcnt->getAPInt().getZExtValue();
    else
      features[i].tripCount = -1;
    features[i].depth = l->getLoopDepth();
    computeBlockStats(features[i]);
    i++;
  }

  // sort blocks by depth first, trip count later
  std::sort(features.begin(), features.end());

  /* loopNestMtx[i][j] == true  ==>>  L_j inside L_i */
  std::vector<std::vector<bool>> loopNestMtx;
  loopNestMtx.resize(nfeat);
  for (int i = 0; i < nfeat; i++) {
    loopNestMtx[i].resize(nfeat);
    loopNestMtx[0][i] = true;
    loopNestMtx[i][i] = true;
  }

  for (int i = 1; i < nfeat; i++) {
    for (int j = 1; j < nfeat; j++) {
      if (loopNestMtx[j][i] == true)
        continue;
      loopNestMtx[i][j] = loops[i - 1]->contains(loops[j - 1]);
    }
  }

  int blockIdx[nfeat];
  blockIdx[0] = 0;
  for (int i = 1, k = 1; i < nfeat; i++)
    if (features[i].imix.ninstr > 0)
      blockIdx[i] = k++;
    else
      blockIdx[i] = -1;

  /* print */
  for (int ri = 0; ri < nfeat; ri++) {
    int i = blockIdx[ri];
    if (i < 0)
      continue;
    for (int rj = 1; rj < nfeat; rj++) {
      int j = blockIdx[rj];
      if (j < 0)
        continue;
      std::cout << "B" << i << "_contain_B" << j << " " << loopNestMtx[ri][rj] << std::endl;
    }
    std::cout << "B" << i << "_depth " << features[ri].depth << std::endl;
    std::cout << "B" << i << "_tripCount " << features[ri].tripCount << std::endl;
    std::cout << "B" << i << "_maxAllocSize " << features[ri].maxAllocSize << std::endl;
    std::cout << "B" << i << "_numAnnotatedInstr " << features[ri].numAnnotatedInstr << std::endl;
    std::cout << "B" << i << "_minDist_mul " << features[ri].minDist_mul << std::endl;
    std::cout << "B" << i << "_minDist_div " << features[ri].minDist_div << std::endl;
    std::cout << "B" << i << "_minDist_call " << features[ri].minDist_callBase << std::endl;
    std::cout << "B" << i << "_n_* " << features[ri].imix.ninstr << std::endl;
    for (auto it = features[ri].imix.stat.begin(); it != features[ri].imix.stat.end(); it++)
      std::cout << "B" << i << "_n_" << it->first << " " << it->second << std::endl;
  }

  return false;
}
