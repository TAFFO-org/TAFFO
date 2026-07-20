#pragma once
#include "CodeInterpreter.hpp"
#include "VRALogger.hpp"
#include "RangedRecurrences.hpp"
#include <llvm/Analysis/ValueTracking.h>

#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/MemorySSA.h>

#include <cstdint>
#include <memory>
#include <map>
#include <string>
#define DEBUG_TYPE "taffo-vra"
namespace llvm { class raw_ostream; }
namespace taffo {

class Range;
class VRAGlobalStore;

enum FollowingPathResponse {
    ENQUE_BLOCK, NO_ENQUE, LOOP_JOIN, LOOP_FORK
};

enum class TripCountReason {
    Unknown = 0,
    DeducedBySCEV,
    HeuristicFallback
    // (future slots) UserHint, BoundedByGuard, ecc.
};

struct VRALoopInfo {
    llvm::Loop* L;
    u_int64_t TripCount = 0;
    TripCountReason Reason = TripCountReason::Unknown;
    llvm::Value* InductionVariable;
    llvm::CmpInst* exitCmp = nullptr;

    llvm::SmallVector<llvm::BasicBlock*> bbFlow;

    /// @brief check if this value is invariant w.r.t. this loop
    /// @param V llvm::Value to analyze
    /// @return true when it is invariant
    bool isInvariant(const llvm::Value* V) { 

        if (!V) return true;
        auto *I = dyn_cast<llvm::Instruction>(V);

        if (I && !L->contains(I)) return true;
        if (L->isLoopInvariant(V)) return true;

        if (!I || !isSafeToSpeculativelyExecute(I)) return false;

        for (const llvm::Value *Op : I->operands())
            if (!isInvariant(Op)) return false;
        return true;
    }

    /// @brief check if all blocks of the loop are visited
    /// @return true is whole visited (all latches are visited), false otherwise
    bool isEntirelyVisited() {
        llvm::SmallVector<llvm::BasicBlock*, 4> latches;
        L->getLoopLatches(latches);
        
        for (auto latch : latches) {
            if (std::find(bbFlow.begin(), bbFlow.end(), latch) == bbFlow.end()) return false;
        }
        return true;
    }

    VRALoopInfo() : L(nullptr) {}
    VRALoopInfo(llvm::Loop* L) : L(L) {}
};

enum VRAInspectionKind {
    REC,        // known recurrence
    ASSIGN,       // assignation
    UNKNOWN     // unhandled
};

struct RecurrenceSummary {
    std::map<std::string, std::uint64_t> counts;
    std::uint64_t unsolved = 0;
};

struct VRARecurrenceInfo {
    const llvm::Value* root;
    llvm::SmallVector<const llvm::Value*> chain;
    llvm::SmallVector<const llvm::LoadInst*> loads;
    VRAInspectionKind kind;

    // reference to build specialized affine/geo recurrence
    const llvm::LoadInst* loadJunction = nullptr;           // load which correspond the same basemem of the store
    const llvm::LoadInst* loadHigherDim = nullptr;          // load from other base mem but with higher dimension (for flattened)
    const llvm::Value* innerRR = nullptr;                   // used for collect nested RR (delta)

    llvm::SmallVector<llvm::Function*> depsOnFn;
    llvm::SmallVector<llvm::Value*> depsOnRR;

    std::shared_ptr<RangedRecurrence> RR = nullptr;
    std::shared_ptr<Range> lastRange = nullptr;             // cache to avoid new computation, exspecially when lastRangeComputedAt is not increased (common case)
    u_int64_t lastRangeComputedAt = 0;

    VRARecurrenceInfo(): root(nullptr) {}
    VRARecurrenceInfo(const llvm::Value* root): root(root) {}
    bool isValid() { return chain.size() > 0; }
    std::string chainToString();
};

/// @brief Assignations are useful for composed recurrence (like crossing in this work)
struct VRAAssignationInfo : public VRARecurrenceInfo {
    VRAAssignationInfo() : VRARecurrenceInfo() { kind = VRAInspectionKind::ASSIGN; }
    VRAAssignationInfo(const llvm::Value* root) : VRARecurrenceInfo(root) { kind = VRAInspectionKind::ASSIGN; }
    VRAAssignationInfo(const llvm::Value* root, llvm::SmallVector<const llvm::Value*> Chain, llvm::SmallVector<const llvm::LoadInst*> Loads) 
        : VRARecurrenceInfo(root) {
        kind = VRAInspectionKind::ASSIGN;
        chain = std::move(Chain);
        loads = std::move(Loads);
    }
};

// Utility used by both VRAnalyzer and ModuleInterpreter to peel off pointer casts and find the originating memory object for a given pointer.
const llvm::Value* getBaseMemoryObject(const llvm::Value* Ptr);

struct VRAFunctionInfo {
    llvm::Function* F;
    llvm::SmallVector<llvm::BasicBlock*> bbFlow;
    llvm::DenseMap<const llvm::Loop*, VRALoopInfo> loops;
    llvm::DenseMap<const llvm::Value*, VRARecurrenceInfo> RRs;
    llvm::DenseMap<const llvm::Value*, VRAAssignationInfo> ASs;
    FunctionScope scope;

    llvm::DominatorTree* DT = nullptr;
    llvm::LoopInfo* LI = nullptr;
    llvm::ScalarEvolution* SE = nullptr;
    llvm::MemorySSA* MSSA = nullptr;

    VRAFunctionInfo(): F(nullptr) {}
    VRAFunctionInfo(llvm::Function* F, llvm::ModuleAnalysisManager& MAM): F(F) {
        auto& FAM = MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(*F->getParent()).getManager();

        if(F->isDeclaration())
            return;

        LI = &(FAM.getResult<llvm::LoopAnalysis>(*F));
        DT = &(FAM.getResult<llvm::DominatorTreeAnalysis>(*F));
        SE = &(FAM.getResult<llvm::ScalarEvolutionAnalysis>(*F));

        if(auto* MSSAResult = FAM.getCachedResult<llvm::MemorySSAAnalysis>(*F))
            MSSA = &(MSSAResult->getMSSA());
        else {
            auto*mssaRes = &FAM.getResult<llvm::MemorySSAAnalysis>(*F);
            if(mssaRes)
                MSSA = &mssaRes->getMSSA();
        }
    }

    void addRecurrenceInfo(VRARecurrenceInfo RI) {
        RRs.try_emplace(RI.root, RI);
    }

    void addAssignmentInfo(VRAAssignationInfo AI) {
        ASs.try_emplace(AI.root, AI);
    }

    size_t countLoops() { return loops.size(); }
    bool existAtLeastOneLoopWithoutTripCount() {
        for (auto [_, loop] : loops) {
            if (loop.TripCount == 0) return true;
        }
        return false;
    }
};

class ModuleInterpreter {
public:

    std::shared_ptr<AnalysisStore> getStoreForValue(const llvm::Value* V) const;
    std::shared_ptr<VRAGlobalStore> getGlobalStore() const { return GlobalStore; }
    std::shared_ptr<AnalysisStore> getFunctionStore() const {
        if (curFn.empty())
            return nullptr;
        auto It = FNs.find(curFn.back());
        if (It == FNs.end())
            return nullptr;
        return It->second.scope.FunctionStore;
    }
    llvm::ModuleAnalysisManager& getMAM() const { return MAM; }
    const RecurrenceSummary& getRecurrenceSummary() const { return RecSummary; }
    void printRecurrenceSummary(llvm::raw_ostream &OS) const;

    /// @brief get specific recurrence info (if any) from its root
    /// @param root llvm::Value* root of the recurrence
    /// @return VRARecurrenceInfo obj if exists or nullptr
    VRARecurrenceInfo* getVRARecurrenceInfo(const llvm::Value* root) {
        for (auto &FEntry : FNs) {
            auto &VFI = FEntry.second;
            if (auto It = VFI.RRs.find(root); It != VFI.RRs.end())
                return &It->second;
        }
        return nullptr;
    }

    /// @brief get specific function info (if any) from its llvm::Function*
    /// @param F llvm::Function* pointer
    /// @return VRAFunctionInfo obj if exists or nullptr
    VRAFunctionInfo* getVRAFunctionInfo(llvm::Function* F) {
        if (FNs.count(F)) {
            return &FNs[F];
        }
        return nullptr;
    }

    /// @brief retrieve last range stored for specific base memory
    /// @param BaseStore base memory
    /// @return range stored if any or nullptr
    std::shared_ptr<Range> getLastStoredRange(const llvm::Value* BaseStore);

    /// @brief run algorithm for this module
    void interpret();

    // method to embed fixed-point loop and avoid recall preseed and inspect
    void resolve();

    ModuleInterpreter(llvm::Module& M, llvm::ModuleAnalysisManager& MAM);

protected:

    // 1) PRESEED METHODS
    void preSeed();
    void interpretFunction(llvm::Function* F, std::shared_ptr<AnalysisStore> FunctionStore = nullptr);
    FollowingPathResponse followPath(VRAFunctionInfo info, llvm::BasicBlock* src, llvm::BasicBlock* dst, llvm::SmallVector<llvm::Loop*> nesting) const;
    void interpretCall(std::shared_ptr<CodeAnalyzer> CurAnalyzer, llvm::Instruction* I);
    void updateSuccessorAnalyzer(std::shared_ptr<CodeAnalyzer> CurrentAnalyzer, llvm::Instruction* TermInstr, unsigned SuccIdx);

    // 2) INSPECTION PHASE METHODS
    void inspect();
    bool isInductionVariable(llvm::Function *F, llvm::Loop* L, const llvm::PHINode* PHI);
    void handlePHIChain(VRAFunctionInfo VFI, llvm::Loop* L, const llvm::PHINode* PHI, VRARecurrenceInfo& VRI);
    void handleStoreChain(VRAFunctionInfo VFI, llvm::Loop* L, const llvm::StoreInst* Store, VRARecurrenceInfo& VRI);

    // LATTICE - RESOLUTION METHODS FOR CALLS
    void resolveFunction(llvm::Function* F, std::shared_ptr<AnalysisStore> FunctionStore = nullptr);
    void resolveCall(std::shared_ptr<CodeAnalyzer> CurAnalyzer, llvm::Instruction* I);
    
    // 3) ASSEMBLING METHODS
    void assemble();

    bool analyzeSolvability(const llvm::Value* cur, VRAFunctionInfo& VFI, VRARecurrenceInfo& VRI, VRALoopInfo& VLI);
    bool isSolvableDependenceTreeBackwark(const llvm::Value *V, llvm::Loop* L, VRARecurrenceInfo& VRI);
    void updateKnownSuccessorAnalyzer(std::shared_ptr<CodeAnalyzer> CurrentAnalyzer, llvm::BasicBlock* nextBlock, const llvm::BasicBlock* curBlock);

    bool isFakeRecurrence(VRARecurrenceInfo& VRI);

    bool isAffineRecurrence(VRARecurrenceInfo& VRI);            // basic, flattened (T,S)
    bool isDeltaAffineRecurrence(VRARecurrenceInfo& VRI);
    bool isCrossingAffineRecurrence(VRAAssignationInfo& VRI);

    bool isGeometricRecurrence(VRARecurrenceInfo& VRI);         // basic, flattened (T,S)   
    bool isDeltaGeometricRecurrence(VRARecurrenceInfo& VRI);
    bool isCrossingGeometricRecurrence(VRAAssignationInfo& VRI);

    bool isLinearRecurrence(VRARecurrenceInfo& VRI);
    // add here new recurrences...

    // check offset of the GEP: used to compute idx delta between store and load
    const llvm::Value* matchIVOffset(VRAFunctionInfo VFI, const llvm::Value *Idx, int64_t &Offset, llvm::Loop *L);

    // 4) TRIP COUNT METHODS
    void tripCount();
    bool existAtLeastOneLoopWithoutTripCount() {
        for (auto [_, FN] : FNs) {
            if (FN.existAtLeastOneLoopWithoutTripCount()) return true;
        }
        return false;
    }

    // 5) PROPAGATION METHODS
    void propagate();
    void propagateFunction(llvm::Function* F, std::shared_ptr<AnalysisStore> FunctionStore = nullptr);
    void walk(llvm::Loop* L = nullptr);

    // resolve all locked loops and RR after last iteration
    void fallback();

    // Statistic methods
    size_t countLoops() {
        size_t num_loops = 0;
        for (auto [F, VFI] : FNs) {
            num_loops += VFI.countLoops();
        }
        return num_loops;
    }

    size_t countPotentialRecurrences() {
        size_t num_rr = 0;
        for (auto [F, VFI] : FNs) {
            num_rr += VFI.RRs.size();
        }
        return num_rr;
    }

    /// @brief Exploits instruction position got from dominance order found by preseed
    /// @param V1 first value to compare
    /// @param V2 second value to compare
    /// @return true when the first value is before the second, false otherwise
    bool isBefore(const llvm::Value *V1, const llvm::Value *V2) const {
        auto I1 = InstrPos.find(V1);
        auto I2 = InstrPos.find(V2);

        if (I1 == InstrPos.end() || I2 == InstrPos.end())
            return false;

        return I1->second < I2->second;
    }

private:

    llvm::Module& M;
    std::shared_ptr<VRAGlobalStore> GlobalStore;
    llvm::SmallVector<llvm::Function*, 4U> curFn;    //current function scope
    llvm::ModuleAnalysisManager& MAM;

    llvm::Function* EntryFn = nullptr;
    llvm::DenseMap<llvm::Function*, VRAFunctionInfo> FNs;

    llvm::SmallVector<const llvm::Value*> solvedRR;
    u_int16_t solvedTC;

    llvm::DenseMap<const llvm::Value*, unsigned> InstrPos;
    u_int16_t remainingUnsolvedRR = 0;
    RecurrenceSummary RecSummary;

    llvm::SmallVector<std::pair<const llvm::Function*, const llvm::Value*>> phiFalledBack;
    bool isFallback = false;

    /// @brief statistic method
    void computeRecurrenceSummary();
};

}
