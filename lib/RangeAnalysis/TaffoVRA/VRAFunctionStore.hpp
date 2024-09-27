#ifndef TAFFO_VRA_FUNCTION_STORE_HPP
#define TAFFO_VRA_FUNCTION_STORE_HPP

#include "CodeInterpreter.hpp"
#include "VRALogger.hpp"
#include "VRAStore.hpp"

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

class VRAFunctionStore : protected VRAStore, public AnalysisStore
{
public:
  VRAFunctionStore(std::shared_ptr<VRALogger> VRAL, const llvm::DataLayout &DL)
      : VRAStore(VRASK_VRAFunctionStore, VRAL, DL),
        AnalysisStore(ASK_VRAFunctionStore),
        ReturnValue() {}

  void convexMerge(const AnalysisStore &Other) override;
  std::shared_ptr<CodeAnalyzer> newCodeAnalyzer(CodeInterpreter &CI) override;
  std::shared_ptr<AnalysisStore> newFunctionStore(CodeInterpreter &CI) override;
  bool hasValue(const llvm::Value *V) const override { return DerivedRanges.count(V); }
  std::shared_ptr<CILogger> getLogger() const override { return Logger; }

  void setNode(const llvm::Value *V, NodePtrT Node) override
  {
    VRAStore::setNode(V, Node);
  }

  NodePtrT getNode(const llvm::Value *V) override
  {
    return VRAStore::getNode(V);
  }

  // Function handling stuff
  NodePtrT getRetVal() const { return ReturnValue; }
  void setRetVal(NodePtrT RetVal);
  void setArgumentRanges(const llvm::Function &F,
                         const std::list<NodePtrT> &AARanges);

  static bool classof(const AnalysisStore *AS)
  {
    return AS->getKind() == ASK_VRAFunctionStore;
  }

  static bool classof(const VRAStore *VS)
  {
    return VS->getKind() == VRASK_VRAFunctionStore;
  }

protected:
  NodePtrT ReturnValue;
};

} // end namespace taffo

#undef DEBUG_TYPE

#endif
