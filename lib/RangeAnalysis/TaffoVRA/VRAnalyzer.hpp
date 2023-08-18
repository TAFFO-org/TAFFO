#ifndef TAFFO_VRANALIZER_HPP
#define TAFFO_VRANALIZER_HPP

#include "CodeInterpreter.hpp"
#include "VRAFunctionStore.hpp"
#include "VRAGlobalStore.hpp"
#include "VRALogger.hpp"
#include "VRAStore.hpp"

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

class VRAnalyzer : protected VRAStore, public CodeAnalyzer
{
public:
  VRAnalyzer(std::shared_ptr<VRALogger> VRAL, CodeInterpreter& CI)
      : VRAStore(VRASK_VRAnalyzer, VRAL),
        CodeAnalyzer(ASK_VRAnalyzer),
        CodeInt(CI) {}

  void convexMerge(const AnalysisStore &Other) override;
  std::shared_ptr<CodeAnalyzer> newCodeAnalyzer(CodeInterpreter &CI) override;
  std::shared_ptr<AnalysisStore> newFunctionStore(CodeInterpreter &CI) override;

  bool hasValue(const llvm::Value *V) const override
  {
    auto It = DerivedRanges.find(V);
    return It != DerivedRanges.end() && It->second;
  }

  std::shared_ptr<CILogger> getLogger() const override { return Logger; }
  std::shared_ptr<CodeAnalyzer> clone() override;
  void analyzeInstruction(llvm::Instruction *I) override;
  void setPathLocalInfo(std::shared_ptr<CodeAnalyzer> SuccAnalyzer,
                        llvm::Instruction *TermInstr, unsigned SuccIdx) override;
  bool requiresInterpretation(llvm::Instruction *I) const override;
  void prepareForCall(llvm::Instruction *I,
                      std::shared_ptr<AnalysisStore> FunctionStore) override;
  void returnFromCall(llvm::Instruction *I,
                      std::shared_ptr<AnalysisStore> FunctionStore) override;

  static bool classof(const AnalysisStore *AS)
  {
    return AS->getKind() == ASK_VRAnalyzer;
  }

  static bool classof(const VRAStore *VS)
  {
    return VS->getKind() == VRASK_VRAnalyzer;
  }

#ifdef UNITTESTS
public:
  NodePtrT getNode(const llvm::Value *v) override;
  void setNode(const llvm::Value *V, NodePtrT Node) override;
#else
private:
  NodePtrT getNode(const llvm::Value *v) override;
  void setNode(const llvm::Value *V, NodePtrT Node) override;
#endif

private:
  // Instruction Handlers
  void handleSpecialCall(const llvm::Instruction *I);
  void handleMemCpyIntrinsics(const llvm::Instruction *memcpy);
  bool isMallocLike(const llvm::Function *F) const;
  bool isCallocLike(const llvm::Function *F) const;
  void handleMallocCall(const llvm::CallBase *CB);
  bool detectAndHandleLibOMPCall(const llvm::CallBase *CB);

  void handleReturn(const llvm::Instruction *ret);

  void handleAllocaInstr(const llvm::Instruction *I);
  void handleStoreInstr(const llvm::Instruction *store);
  void handleLoadInstr(llvm::Instruction *load);
  void handleGEPInstr(const llvm::Instruction *gep);
  void handleBitCastInstr(const llvm::Instruction *I);

  void handleCmpInstr(const llvm::Instruction *cmp);
  void handlePhiNode(const llvm::Instruction *phi);
  void handleSelect(const llvm::Instruction *i);

  // Data handling
  using VRAStore::fetchRange;
  const range_ptr_t fetchRange(const llvm::Value *V) override;
  const RangeNodePtrT fetchRangeNode(const llvm::Value *V) override;


  // Interface with CodeInterpreter
  std::shared_ptr<VRAGlobalStore> getGlobalStore() const
  {
    return std::static_ptr_cast<VRAGlobalStore>(CodeInt.getGlobalStore());
  }

  std::shared_ptr<VRAStore> getAnalysisStoreForValue(const llvm::Value *V) const
  {
    std::shared_ptr<AnalysisStore> AStore = CodeInt.getStoreForValue(V);
    if (!AStore) {
      return nullptr;
    }

    // Since llvm::dyn_cast<T>() does not do cross-casting, we must do this:
    if (std::shared_ptr<VRAnalyzer> VRA =
            std::dynamic_ptr_cast<VRAnalyzer>(AStore)) {
      return std::static_ptr_cast<VRAStore>(VRA);
    } else if (std::shared_ptr<VRAGlobalStore> VRAGS =
                   std::dynamic_ptr_cast<VRAGlobalStore>(AStore)) {
      return std::static_ptr_cast<VRAStore>(VRAGS);
    } else if (std::shared_ptr<VRAFunctionStore> VRAFS =
                   std::dynamic_ptr_cast<VRAFunctionStore>(AStore)) {
      return std::static_ptr_cast<VRAStore>(VRAFS);
    }
    return nullptr;
  }

  // Logging
  void logRangeln(const llvm::Value *v);

  CodeInterpreter &CodeInt;
};

} // end namespace taffo

#undef DEBUG_TYPE

#endif
