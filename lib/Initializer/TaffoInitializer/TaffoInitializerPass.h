#ifndef __TAFFO_INITIALIZER_PASS_H__
#define __TAFFO_INITIALIZER_PASS_H__


#include "InputInfo.h"
#include "MultiValueMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <limits>


#define DEBUG_TYPE "taffo-init"


STATISTIC(AnnotationCount, "Number of valid annotations found");
STATISTIC(FunctionCloned, "Number of fixed point function inserted");


namespace taffo
{

struct ValueInfo {
  unsigned int backtrackingDepthLeft = 0;
  unsigned int fixpTypeRootDistance = UINT_MAX;

  std::shared_ptr<mdutils::MDInfo> metadata;
  llvm::Optional<std::string> target;
};


struct TaffoInitializer : public llvm::ModulePass {
  static char ID;

  using ConvQueueT = MultiValueMap<llvm::Value *, ValueInfo>;

  llvm::SmallPtrSet<llvm::Function *, 32> enabledFunctions;

  TaffoInitializer() : ModulePass(ID) {}
  bool runOnModule(llvm::Module &M) override;

  llvm::Function *findStartingPointFunctionGlobal(llvm::Module &M);
  void readGlobalAnnotations(llvm::Module &m, ConvQueueT &res, bool functionAnnotation = false);
  void readLocalAnnotations(llvm::Function &f, ConvQueueT &res);
  void readAllLocalAnnotations(llvm::Module &m, ConvQueueT &res);
  bool parseAnnotation(ConvQueueT &res, llvm::ConstantExpr *annoPtrInst, llvm::Value *instr, bool *isTarget = nullptr);
  void removeNoFloatTy(ConvQueueT &res);
  void printAnnotatedObj(llvm::Module &m);

  void buildConversionQueueForRootValues(const ConvQueueT &val, ConvQueueT &res);
  void createInfoOfUser(llvm::Value *used, const ValueInfo &VIUsed, llvm::Value *user, ValueInfo &VIUser);
  std::shared_ptr<mdutils::MDInfo> extractGEPIMetadata(const llvm::Value *user,
                                                       const llvm::Value *used,
                                                       std::shared_ptr<mdutils::MDInfo> user_mdi,
                                                       std::shared_ptr<mdutils::MDInfo> used_mdi);
  void generateFunctionSpace(ConvQueueT &vals, ConvQueueT &global, llvm::SmallPtrSet<llvm::Function *, 10> &callTrace);
  llvm::Function *createFunctionAndQueue(llvm::CallBase *call, ConvQueueT &vals, ConvQueueT &global, std::vector<llvm::Value *> &convQueue);
  void printConversionQueue(ConvQueueT &vals);
  void removeAnnotationCalls(ConvQueueT &vals);

  void setMetadataOfValue(llvm::Value *v, ValueInfo &VI);
  void setFunctionArgsMetadata(llvm::Module &m, ConvQueueT &Q);

  bool isSpecialFunction(const llvm::Function *f)
  {
    llvm::StringRef fName = f->getName();
    return fName.startswith("llvm.") || f->getBasicBlockList().empty();
  };
};


} // namespace taffo


#undef DEBUG_TYPE

#endif
