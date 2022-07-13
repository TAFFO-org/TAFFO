#include "VRAFunctionStore.hpp"

#include "RangeOperations.hpp"
#include "VRAGlobalStore.hpp"
#include "VRAnalyzer.hpp"

#define DEBUG_TYPE "taffo-vra"

using namespace taffo;

void VRAFunctionStore::convexMerge(const AnalysisStore &Other)
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
VRAFunctionStore::newCodeAnalyzer(CodeInterpreter &CI)
{
  return std::make_shared<VRAnalyzer>(CI);
}

std::shared_ptr<AnalysisStore>
VRAFunctionStore::newFunctionStore(CodeInterpreter &CI)
{
  return std::make_shared<VRAFunctionStore>(CI);
}

void VRAFunctionStore::setRetVal(NodePtrT RetVal)
{
  if (!RetVal)
    return;

  if (RangeNodePtrT RetRange = std::dynamic_ptr_cast<VRARangeNode>(RetVal)) {
    RangeNodePtrT ReturnRange = std::dynamic_ptr_cast_or_null<VRARangeNode>(ReturnValue);
    ReturnValue = getUnionRange(ReturnRange, RetRange);
  } else {
    ReturnValue = RetVal;
  }
}

void VRAFunctionStore::setArgumentRanges(const llvm::Function &F,
                                         const std::list<NodePtrT> &AARanges)
{
  assert(AARanges.size() == F.arg_size() && "Mismatch between number of actual and formal parameters.");
  auto derived_info_it = AARanges.begin();
  auto derived_info_end = AARanges.end();

  for (const llvm::Argument &formal_arg : F.args()) {
    assert(derived_info_it != derived_info_end);
    if (*derived_info_it) {
      setNode(&formal_arg, *derived_info_it);
    }
    ++derived_info_it;
  }
}
