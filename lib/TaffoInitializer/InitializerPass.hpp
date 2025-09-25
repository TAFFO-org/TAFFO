#pragma once

#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/TaffoInitInfo.hpp"

#include <llvm/ADT/Statistic.h>
#include <llvm/IR/PassManager.h>

#include <list>

#define DEBUG_TYPE "taffo-init"

STATISTIC(AnnotationCount, "Number of valid annotations found");
STATISTIC(FunctionCloned, "Number of fixed point function inserted");

namespace taffo {

class InitializerPass : public llvm::PassInfoMixin<InitializerPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);

private:
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  TaffoInitInfo taffoInitInfo;
  std::list<llvm::Value*> infoPropagationQueue;
  llvm::SmallPtrSet<llvm::Function*, 2> annotatedFunctions;
  llvm::DenseMap<llvm::CallBase*, llvm::Function*> handledCalls;

  /**
   * Check for indirect calls the module, adding dedicated direct trampoline calls.
   * The trampolines enable subsequent passes to better analyze the indirect calls.
   */
  void manageIndirectCalls(llvm::Module& m);
  void handleCallIfIndirect(const llvm::Module& m, llvm::CallBase* curCall, llvm::Function* indirectFunction);
  void handleKmpcFork(const llvm::Module& m, llvm::CallBase* call, llvm::Function* indirectFunction);

  llvm::Function* findStartingPointFunctionGlobal(llvm::Module& m);

  void readAndRemoveGlobalAnnotations(llvm::Module& m);
  void readAndRemoveLocalAnnotations(llvm::Function& f);
  void readAndRemoveLocalAnnotations(llvm::Module& m);
  void parseAnnotation(llvm::Value* annotatedValue, llvm::Value* annotationValue, bool* isStartingPoint = nullptr);

  void propagateInfo();
  void propagateInfo(llvm::Value* value, llvm::Value* user);
  void propagateInfo(const ValueInfo* srcInfo,
                     const ValueInitInfo& srcInitInfo,
                     ValueInfo* dstInfo,
                     ValueInitInfo& dstInitInfo);
  void cloneFunctionForCall(llvm::CallBase* call);
  void saveValueWeights();
  void logInfoPropagationQueue();

  bool isSpecialFunction(const llvm::Function* f) {
    llvm::StringRef fName = f->getName();
    return fName.starts_with("llvm.") || f->empty();
  }
};

} // namespace taffo

#undef DEBUG_TYPE
