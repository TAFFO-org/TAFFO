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
  llvm::PreservedAnalyses run(llvm::Module &m, llvm::ModuleAnalysisManager &);

#ifndef UNITTESTS
private:
#endif
  std::list<llvm::Value*> infoPropagationQueue;
  TaffoInitInfo taffoInitInfo;

  llvm::Function *findStartingPointFunctionGlobal(llvm::Module &m);

  void readGlobalAnnotations(llvm::Module &m, bool functionAnnotation = false);
  void readAndRemoveLocalAnnotations(llvm::Function &f);
  void readAndRemoveLocalAnnotations(llvm::Module &m);

  void parseAnnotation(llvm::Value *annotationValue, llvm::Value *annotatedValue, bool *isTarget = nullptr);
  void removeNoFloatTy();
  void printAnnotatedObj(llvm::Module &m);

  void buildConversionQueueForRootValues(const ConvQueueType &roots, ConvQueueType &queue);
  void createUserInfo(llvm::Value *value, ConvQueueInfo& valueConvQueueInfo, llvm::Value *user, ConvQueueInfo &userConvQueueInfo);
  std::shared_ptr<ValueInfo> extractGEPValueInfo(const llvm::Value *value, std::shared_ptr<ValueInfo> valueInfo, const llvm::Value *user);
  void generateFunctionSpace(ConvQueueType &convQueue, ConvQueueType &global, llvm::SmallPtrSet<llvm::Function *, 10> &callTrace);
  llvm::Function *createFunctionAndQueue(llvm::CallBase *call, ConvQueueType &vals, ConvQueueType &global, std::vector<llvm::Value *> &convQueue);
  void printConversionQueue(ConvQueueType &queue);
  void removeAnnotationCalls(ConvQueueType &queue);

  void setValueInfo(llvm::Value *v, const ConvQueueInfo &valueConvQueueInfo);
  void setFunctionArgsInfo(llvm::Module &m, ConvQueueType &queue);

  bool isSpecialFunction(const llvm::Function *f) {
    llvm::StringRef fName = f->getName();
    return fName.starts_with("llvm.") || f->empty();
  }
};

} // namespace taffo

#undef DEBUG_TYPE
#endif // TAFFO_INITIALIZER_PASS_HPP
