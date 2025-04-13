#ifndef TAFFO_INITIALIZER_PASS_HPP
#define TAFFO_INITIALIZER_PASS_HPP

#include "Initializer/TaffoInitializer/TaffoInfo/TaffoInitInfo.hpp"
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/AbstractCallSite.h>
#include <list>

#define DEBUG_TYPE "taffo-init"

STATISTIC(AnnotationCount, "Number of valid annotations found");
STATISTIC(FunctionCloned, "Number of fixed point function inserted");

namespace taffo {

class TaffoInitializerPass : public llvm::PassInfoMixin<TaffoInitializerPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &m, llvm::ModuleAnalysisManager&);

#ifndef UNITTESTS
private:
#endif
  TaffoInitInfo taffoInitInfo;
  std::list<llvm::Value*> infoPropagationQueue;
  llvm::SmallPtrSet<llvm::Function*, 2> annotatedFunctions;

  llvm::Function *findStartingPointFunctionGlobal(llvm::Module &m);

  void readAndRemoveGlobalAnnotations(llvm::Module &m);
  void readAndRemoveLocalAnnotations(llvm::Function &f);
  void readAndRemoveLocalAnnotations(llvm::Module &m);
  void parseAnnotation(llvm::Value *annotatedValue, llvm::Value *annotationValue, bool *isStartingPoint = nullptr);
  void removeNotFloats();

  void propagateInfo();
  void propagateInfo(llvm::Value *src, llvm::Value *dst);
  //std::shared_ptr<ValueInfo> extractGEPValueInfo(const llvm::Value *value, ValueInfo *valueInfo, const llvm::Value *user);
  void generateFunctionClones();
  llvm::Function *cloneFunction(llvm::CallBase *call);
  void saveValueWeights();
  void logInfoPropagationQueue();

  bool isSpecialFunction(const llvm::Function *f) {
    llvm::StringRef fName = f->getName();
    return fName.starts_with("llvm.") || f->empty();
  }
};

} // namespace taffo

#undef DEBUG_TYPE
#endif // TAFFO_INITIALIZER_PASS_HPP
