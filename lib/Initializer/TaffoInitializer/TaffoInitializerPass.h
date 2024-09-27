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
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <limits>
#include "llvm/Demangle/Demangle.h"

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


class TaffoInitializer : public llvm::PassInfoMixin<TaffoInitializer> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

#ifndef UNITTESTS
private:
#endif
  using ConvQueueT = MultiValueMap<llvm::Value *, ValueInfo>;

  llvm::SmallPtrSet<llvm::Function *, 32> enabledFunctions;

  llvm::Function *findStartingPointFunctionGlobal(llvm::Module &M);
  void readGlobalAnnotations(llvm::Module &m, const llvm::DataLayout &DL, ConvQueueT &res, bool functionAnnotation = false);
  void readLocalAnnotations(llvm::Function &f, const llvm::DataLayout &DL, ConvQueueT &res);
  void readAllLocalAnnotations(llvm::Module &m, const llvm::DataLayout &DL, ConvQueueT &res);
  bool parseAnnotation(const llvm::DataLayout &DL, ConvQueueT &res, llvm::ConstantExpr *annoPtrInst, llvm::Value *instr, bool *isTarget = nullptr);
  void removeNoFloatTy(ConvQueueT &res);
  void printAnnotatedObj(llvm::Module &m, const llvm::DataLayout &DL);

  void buildConversionQueueForRootValues(const ConvQueueT &val, ConvQueueT &res, const llvm::DataLayout &DL);
  void createInfoOfUser(llvm::Value *used, const ValueInfo &VIUsed, llvm::Value *user, ValueInfo &VIUser, const llvm::DataLayout &DL);
  std::shared_ptr<mdutils::MDInfo> extractGEPIMetadata(const llvm::Value *user,
                                                       const llvm::Value *used,
                                                       std::shared_ptr<mdutils::MDInfo> user_mdi,
                                                       std::shared_ptr<mdutils::MDInfo> used_mdi);
  void generateFunctionSpace(const llvm::DataLayout &DL, ConvQueueT &vals, ConvQueueT &global, llvm::SmallPtrSet<llvm::Function *, 10> &callTrace);
  llvm::Function *createFunctionAndQueue(const llvm::DataLayout &DL, llvm::CallBase *call, ConvQueueT &vals, ConvQueueT &global, std::vector<llvm::Value *> &convQueue);
  void printConversionQueue(ConvQueueT &vals);
  void removeAnnotationCalls(ConvQueueT &vals);

  void setMetadataOfValue(llvm::Value *v, ValueInfo &VI);
  void setFunctionArgsMetadata(llvm::Module &m, ConvQueueT &Q);

  //Special functions are ignored by Taffo
  bool isSpecialFunction(const llvm::Function *f)
  {
    llvm::StringRef fName = f->getName();
    std::basic_string<char> demangledFName = llvm::demangle(fName.str());
    bool isSpecial = fName.startswith("llvm.")
                     || f->getBasicBlockList().empty()
                     || demangledFName.find("std::_Sp_counted_base") != std::string::npos;
    LLVM_DEBUG(if (isSpecial) llvm::dbgs() << "Special function name: " << demangledFName << "\n";
               else llvm::dbgs() << "Non-special function name: " << demangledFName << "\n";);

    return isSpecial;
  };
};


} // namespace taffo


#undef DEBUG_TYPE

#endif
