#pragma once

#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <string>
#include <map>

#define DEBUG_TYPE "taffo-dta"

namespace tuner {

bool isMergeable(const std::shared_ptr<taffo::FixedPointInfo> &fpv,
                 const std::shared_ptr<taffo::FixedPointInfo> &fpu);
std::shared_ptr<taffo::FixedPointInfo> merge(const std::shared_ptr<taffo::FixedPointInfo> &fpv,
                                       const std::shared_ptr<taffo::FixedPointInfo> &fpu);
std::shared_ptr<taffo::NumericTypeInfo> merge(const std::shared_ptr<taffo::NumericTypeInfo> &fpv,
                                          const std::shared_ptr<taffo::NumericTypeInfo> &fpu);

struct TunerInfo {
  std::shared_ptr<taffo::ValueInfo> metadata;
  std::shared_ptr<taffo::NumericTypeInfo> initialType;
  std::optional<std::string> bufferID;
};

struct FunInfo {
  llvm::Function *newFun;
  /* {function argument index, type of argument}
   * argument idx is -1 for return value */
  std::vector<std::pair<int, std::shared_ptr<taffo::ValueInfo>>> fixArgs;
};

class DataTypeAllocationPass : public llvm::PassInfoMixin<DataTypeAllocationPass> {
public:
  /* to not be accessed directly, use valueInfo() */
  llvm::DenseMap<llvm::Value *, std::shared_ptr<TunerInfo>> info;
  /* original function -> cloned function map */
  llvm::DenseMap<llvm::Function *, std::vector<FunInfo>> functionPool;
  /* buffer ID sets */
  std::map<std::string, llvm::SmallPtrSet<llvm::Value *, 2>> bufferIDSets;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  void retrieveAllMetadata(llvm::Module &m, std::vector<llvm::Value *> &vals,
                           llvm::SmallPtrSetImpl<llvm::Value *> &valset);
  
  void retrieveBufferID(llvm::Value *V);

  bool processMetadataOfValue(llvm::Value *v, const std::shared_ptr<taffo::ValueInfo>& valueInfo);

  bool associateFixFormat(std::shared_ptr<taffo::ScalarInfo> &scalarInfo, llvm::Value *value);

  void sortQueue(std::vector<llvm::Value *> &vals,
                 llvm::SmallPtrSetImpl<llvm::Value *> &valset);

  void mergeFixFormat(const std::vector<llvm::Value *> &vals,
                      const llvm::SmallPtrSetImpl<llvm::Value *> &valset);

#ifdef TAFFO_BUILD_ILP_DTA
  void buildModelAndOptimze(llvm::Module &m,
                            const std::vector<llvm::Value *> &vals,
                            const llvm::SmallPtrSetImpl<llvm::Value *> &valset);
#endif // TAFFO_BUILD_ILP_DTA

  bool mergeFixFormat(llvm::Value *v, llvm::Value *u);

  bool mergeFixFormatIterative(llvm::Value *v, llvm::Value *u);

  void mergeBufferIDSets();

  void restoreTypesAcrossFunctionCall(llvm::Value *arg_or_call_param);
  void setTypesOnFunctionArgumentFromCallArgument(llvm::Value *call_param, std::shared_ptr<taffo::ValueInfo> finalMd);
  void setTypesOnCallArgumentFromFunctionArgument(llvm::Argument *arg, std::shared_ptr<taffo::ValueInfo> finalMd);

  std::vector<llvm::Function *> collapseFunction(llvm::Module &m);

  llvm::Function *findEqFunction(llvm::Function *fun, llvm::Function *origin);

  void attachFPMetaData(std::vector<llvm::Value *> &vals);

  void attachFunctionMetaData(llvm::Module &m);

  std::shared_ptr<TunerInfo> getTunerInfo(llvm::Value *val)
  {
    auto vi = info.find(val);
    if (vi == info.end()) {
      LLVM_DEBUG(llvm::dbgs() << "new valueinfo for " << *val << "\n");
      info[val] = std::make_shared<TunerInfo>(TunerInfo());
      return info[val];
    } else {
      return vi->getSecond();
    }
  }

  bool hasTunerInfo(llvm::Value *val) { return info.find(val) != info.end(); }

  bool conversionDisabled(llvm::Value *val) {
    if (llvm::isa<llvm::Constant>(val))
      return false;
    if (llvm::isa<llvm::Argument>(val)) {
      if (!hasTunerInfo(val))
        return true;
      return !(getTunerInfo(val)->metadata && getTunerInfo(val)->metadata->isConversionEnabled());
    }
    std::shared_ptr<taffo::ValueInfo> valueInfo = taffo::TaffoInfo::getInstance().getValueInfo(*val);
    return !(valueInfo && valueInfo->isConversionEnabled()) && incomingValuesDisabled(val);
  }

  bool incomingValuesDisabled(llvm::Value *v) {
    using namespace llvm;
    if (!taffo::getUnwrappedType(v)->isFloatTy())
      return true;

    if (auto *phi = dyn_cast<PHINode>(v)) {
      bool disabled = false;
      for (Value *inc : phi->incoming_values()) {
        if (!isa<PHINode>(inc) && conversionDisabled(inc)) {
          disabled = true;
          break;
        }
      }
      return disabled;
    } else {
      return true;
    }
  }

#ifdef TAFFO_BUILD_ILP_DTA
  bool overwriteType(std::shared_ptr<taffo::ValueInfo> old, std::shared_ptr<taffo::ValueInfo> model);
#endif // TAFFO_BUILD_ILP_DTA

  template <typename AnalysisT>
  typename AnalysisT::Result& getFunctionAnalysisResult(llvm::Function& F)
  {
    auto &FAM = MAM->getResult<llvm::FunctionAnalysisManagerModuleProxy>(*(F.getParent())).getManager();
    return FAM.getResult<AnalysisT>(F);
  }

  llvm::ModuleAnalysisManager& getMAM() { assert(MAM); return *MAM; }

  llvm::FunctionAnalysisManager& getFAM(llvm::Module& M)
  {
    assert(MAM);
    return MAM->getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
  }

private:
  llvm::ModuleAnalysisManager *MAM = nullptr;
};

} // namespace tuner
 
#undef DEBUG_TYPE
