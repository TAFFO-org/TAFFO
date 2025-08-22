#include "RangeOperations.hpp"
#include "VRAFunctionStore.hpp"
#include "VRAGlobalStore.hpp"
#include "VRAnalyzer.hpp"

#define DEBUG_TYPE "taffo-vra"

using namespace taffo;

void VRAFunctionStore::convexMerge(const AnalysisStore& Other) {
  // Since llvm::dyn_cast<T>() does not do cross-casting, we must do this:
  if (llvm::isa<VRAnalyzer>(Other))
    VRAStore::convexMerge(llvm::cast<VRAStore>(llvm::cast<VRAnalyzer>(Other)));
  else if (llvm::isa<VRAGlobalStore>(Other))
    VRAStore::convexMerge(llvm::cast<VRAStore>(llvm::cast<VRAGlobalStore>(Other)));
  else
    VRAStore::convexMerge(llvm::cast<VRAStore>(llvm::cast<VRAFunctionStore>(Other)));
}

std::shared_ptr<CodeAnalyzer> VRAFunctionStore::newCodeAnalyzer(CodeInterpreter& CI) {
  return std::make_shared<VRAnalyzer>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()), CI);
}

std::shared_ptr<AnalysisStore> VRAFunctionStore::newFunctionStore(CodeInterpreter& CI) {
  return std::make_shared<VRAFunctionStore>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()));
}

void VRAFunctionStore::setRetVal(std::shared_ptr<ValueInfo> RetVal) {
  if (!RetVal)
    return;

  if (std::shared_ptr<ValueInfoWithRange> RetRange = std::dynamic_ptr_cast<ValueInfoWithRange>(RetVal)) {
    std::shared_ptr<ValueInfoWithRange> ReturnRange = std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(ReturnValue);
    ReturnValue = getUnionRange(ReturnRange, RetRange);
  }
  else {
    ReturnValue = RetVal;
  }
}

void VRAFunctionStore::setArgumentRanges(const llvm::Function& F,
                                         const std::list<std::shared_ptr<ValueInfo>>& AARanges) {
  assert(AARanges.size() == F.arg_size() && "Mismatch between number of actual and formal parameters.");
  auto derived_info_it = AARanges.begin();
  auto derived_info_end = AARanges.end();

  for (const llvm::Argument& formal_arg : F.args()) {
    assert(derived_info_it != derived_info_end);
    if (*derived_info_it)
      setNode(&formal_arg, *derived_info_it);
    ++derived_info_it;
  }
}
