#ifndef TAFFO_INITIALIZER_PASS_HPP
#define TAFFO_INITIALIZER_PASS_HPP

#include "TaffoInfo/ValueInfo.hpp"
#include "InsertionOrderedMap.hpp"

#include <llvm/ADT/Statistic.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/AbstractCallSite.h>
#include <limits>

#define DEBUG_TYPE "taffo-init"

STATISTIC(AnnotationCount, "Number of valid annotations found");
STATISTIC(FunctionCloned, "Number of fixed point function inserted");

namespace taffo {

struct ConvQueueInfo {
  unsigned int backtrackingDepthLeft = 0;
  unsigned int rootDistance = UINT_MAX;
  std::shared_ptr<ValueInfo> valueInfo;
  std::string dump() {
    std::string ret;
    ret += "backtrackingDepthLeft: " + std::to_string(backtrackingDepthLeft) + "\n";
    ret += "rootDistance: " + std::to_string(rootDistance) + "\n";
    ret += "valueInfo: " + ( valueInfo ? valueInfo->toString() : "" ) + "\n";
    return ret;
  }
};

class TaffoInitializer : public llvm::PassInfoMixin<TaffoInitializer> {
public:
  llvm::PreservedAnalyses run(llvm::Module &m, llvm::ModuleAnalysisManager &);

#ifndef UNITTESTS
private:
#endif
  using ConvQueueType = InsertionOrderedMap<llvm::Value*, ConvQueueInfo>;

  llvm::SmallPtrSet<llvm::Function*, 32> enabledFunctions;

  llvm::Function *findStartingPointFunctionGlobal(llvm::Module &m);
  void readGlobalAnnotations(llvm::Module &m, ConvQueueType &res, bool functionAnnotation = false);
  void readLocalAnnotations(llvm::Function &f, ConvQueueType &res);
  void readAllLocalAnnotations(llvm::Module &m, ConvQueueType &res);
  bool parseAnnotation(ConvQueueType &annotatedValues, llvm::Value *annotationValue, llvm::Value *annotatedValue, bool *isTarget = nullptr);
  void removeNoFloatTy(ConvQueueType &res);
  void printAnnotatedObj(llvm::Module &m);

  void buildConversionQueueForRootValues(const ConvQueueType &roots, ConvQueueType &queue);
  void createUserInfo(const llvm::Value *value, const ConvQueueInfo& valueConvQueueInfo, llvm::Value *user, ConvQueueInfo &userConvQueueInfo);
  std::shared_ptr<ValueInfo> extractGEPValueInfo(const llvm::Value *value, std::shared_ptr<ValueInfo> valueInfo, const llvm::Value *user);
  void generateFunctionSpace(ConvQueueType &vals, ConvQueueType &global, llvm::SmallPtrSet<llvm::Function *, 10> &callTrace);
  llvm::Function *createFunctionAndQueue(llvm::CallBase *call, ConvQueueType &vals, ConvQueueType &global, std::vector<llvm::Value *> &convQueue);
  void printConversionQueue(ConvQueueType &queue);
  void removeAnnotationCalls(ConvQueueType &queue);

  void setValueInfo(llvm::Value *v, const ConvQueueInfo& valueConvQueueInfo);
  void setFunctionArgsInfo(llvm::Module &m, ConvQueueType &queue);

  bool isSpecialFunction(const llvm::Function *f) {
    llvm::StringRef fName = f->getName();
    return fName.starts_with("llvm.") || f->empty();
  }
};

} // namespace taffo

#undef DEBUG_TYPE
#endif // TAFFO_INITIALIZER_PASS_HPP
