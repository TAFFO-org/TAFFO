#pragma once

#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

#include <map>
#include <string>

#define DEBUG_TYPE "taffo-dta"

namespace tuner {

/* this is the core of the strategy pattern for each new strategy
 * you should create a new class that inherits from dataTypeAllocationStrategy
 * and implement the apply, merge and isMergeable methods.
 * (the apply method is the actual strategy that will be applied to each value
 * the isMergeable method decides if two values are mergeable and the merge methods
 * decides how to merge them)
 *
 * When implementing a new strategy you should:
 * - create a new class that inherits from dataTypeAllocationStrategy
 * - implement the apply, merge and isMergeable methods
 * - add a new entry in the strategyMap in DataTypeAllocationPass.cpp
 * - add a new entry in the DtaStrategyType enum in DTAConfig.hpp and a new entry in the DtaStrategy in DTAConfig.cpp */

class dataTypeAllocationStrategy {
public:
  virtual ~dataTypeAllocationStrategy() {}
  virtual bool apply(std::shared_ptr<taffo::ScalarInfo>& scalarInfo, llvm::Value* value) = 0;
  virtual bool isMergeable(std::shared_ptr<taffo::NumericTypeInfo> valueNumericType,
                           std::shared_ptr<taffo::NumericTypeInfo> userNumericType) = 0;
  virtual std::shared_ptr<taffo::NumericTypeInfo> merge(const std::shared_ptr<taffo::NumericTypeInfo>& fpv,
                                                        const std::shared_ptr<taffo::NumericTypeInfo>& fpu) = 0;
};

// *** STRATEGIES DECLARATIONS ***
class fixedPointOnlyStrategy : public dataTypeAllocationStrategy {
public:
  bool apply(std::shared_ptr<taffo::ScalarInfo>& scalarInfo, llvm::Value* value) override;
  bool isMergeable(std::shared_ptr<taffo::NumericTypeInfo> valueNumericType,
                   std::shared_ptr<taffo::NumericTypeInfo> userNumericType) override;
  std::shared_ptr<taffo::NumericTypeInfo> merge(const std::shared_ptr<taffo::NumericTypeInfo>& fpv,
                                                const std::shared_ptr<taffo::NumericTypeInfo>& fpu) override;
};

class floatingPointOnlyStrategy : public dataTypeAllocationStrategy {
public:
  bool apply(std::shared_ptr<taffo::ScalarInfo>& scalarInfo, llvm::Value* value) override;
  bool isMergeable(std::shared_ptr<taffo::NumericTypeInfo> valueNumericType,
                   std::shared_ptr<taffo::NumericTypeInfo> userNumericType) override;
  std::shared_ptr<taffo::NumericTypeInfo> merge(const std::shared_ptr<taffo::NumericTypeInfo>& fpv,
                                                const std::shared_ptr<taffo::NumericTypeInfo>& fpu) override;
};

class fixedFloatingPointStrategy : public dataTypeAllocationStrategy {
public:
  bool apply(std::shared_ptr<taffo::ScalarInfo>& scalarInfo, llvm::Value* value) override;
  bool isMergeable(std::shared_ptr<taffo::NumericTypeInfo> valueNumericType,
                   std::shared_ptr<taffo::NumericTypeInfo> userNumericType) override;
  std::shared_ptr<taffo::NumericTypeInfo> merge(const std::shared_ptr<taffo::NumericTypeInfo>& fpv,
                                                const std::shared_ptr<taffo::NumericTypeInfo>& fpu) override;
};

// *** END OF STRATEGIES DECLARATIONS ***

struct DtaValueInfo {
  std::shared_ptr<taffo::NumericTypeInfo> initialType;
  std::optional<std::string> bufferID;
};

struct FunInfo {
  llvm::Function* newFun;
  /* {function argument index, type of argument}
   * argument idx is -1 for return value */
  std::vector<std::pair<int, std::shared_ptr<taffo::ValueInfo>>> fixArgs;
};

class DataTypeAllocationPass : public llvm::PassInfoMixin<DataTypeAllocationPass> {
public:
  /* to not be accessed directly, use valueInfo() */
  llvm::DenseMap<llvm::Value*, std::shared_ptr<DtaValueInfo>> info;
  /* original function -> cloned function map */
  llvm::DenseMap<llvm::Function*, std::vector<FunInfo>> functionPool;
  /* buffer ID sets */
  std::map<std::string, llvm::SmallPtrSet<llvm::Value*, 2>> bufferIDSets;

  llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);

  void setStrategy(dataTypeAllocationStrategy* strategy) { this->strategy = strategy; }

  void
  dataTypeAllocation(llvm::Module& m, std::vector<llvm::Value*>& values, llvm::SmallPtrSetImpl<llvm::Value*>& valueSet);

  void dataTypeAllocationOfValue(llvm::Value& value, std::vector<llvm::Value*>& values);

  void dataTypeAllocationOfFunctions(llvm::Module& m, std::vector<llvm::Value*>& values);

  void dataTypeAllocationOfGlobals(llvm::Module& m, std::vector<llvm::Value*>& values);

  void dataTypeAllocationOfArguments(llvm::Function& m, std::vector<llvm::Value*>& values);

  void dataTypeAllocationOfInstructions(llvm::Function& m, std::vector<llvm::Value*>& values);

  void retrieveBufferID(llvm::Value* V);

  bool processScalarInfo(std::shared_ptr<taffo::ScalarInfo>& scalarInfo,
                         llvm::Value* value,
                         const tda::TransparentType* transparentType,
                         bool forceEnable);

  void
  processStructInfo(std::shared_ptr<taffo::StructInfo>& structInfo,
                    llvm::Value* value,
                    const tda::TransparentType* transparentType,
                    llvm::SmallVector<std::pair<std::shared_ptr<taffo::ValueInfo>, tda::TransparentType*>, 8> queue);

  bool processValueInfo(llvm::Value* value);

  void sortQueue(std::vector<llvm::Value*>& vals, llvm::SmallPtrSetImpl<llvm::Value*>& valset);

  void mergeFixFormat(const std::vector<llvm::Value*>& vals, const llvm::SmallPtrSetImpl<llvm::Value*>& valset);

  double static getGreatest(std::shared_ptr<taffo::ScalarInfo>& scalarInfo, llvm::Value* value, taffo::Range* range);

#ifdef TAFFO_BUILD_ILP_DTA
  void buildModelAndOptimze(llvm::Module& m,
                            const std::vector<llvm::Value*>& vals,
                            const llvm::SmallPtrSetImpl<llvm::Value*>& valset);
#endif // TAFFO_BUILD_ILP_DTA

  bool mergeFixFormat(llvm::Value* v, llvm::Value* u);

  void mergeBufferIDSets();

  void restoreTypesAcrossFunctionCall(llvm::Value* arg_or_call_param);
  void setTypesOnFunctionArgumentFromCallArgument(llvm::Value* call_param, std::shared_ptr<taffo::ValueInfo> finalMd);
  void setTypesOnCallArgumentFromFunctionArgument(llvm::Argument* arg, std::shared_ptr<taffo::ValueInfo> finalMd);

  std::vector<llvm::Function*> collapseFunction(llvm::Module& m);

  llvm::Function* findEqFunction(llvm::Function* fun, llvm::Function* origin);

  void attachFPMetaData(std::vector<llvm::Value*>& vals);

  void attachFunctionMetaData(llvm::Module& m);

  std::shared_ptr<DtaValueInfo> createDtaValueInfo(llvm::Value* val) {
    LLVM_DEBUG(tda::log() << "new dtaValueInfo for " << *val << "\n");
    info[val] = std::make_shared<DtaValueInfo>(DtaValueInfo());
    return info[val];
  }

  std::shared_ptr<DtaValueInfo> getDtaValueInfo(llvm::Value* val) {
    auto dtaValueInfo = info.find(val);
    assert(dtaValueInfo != info.end() && "DtaValueInfo not present");
    return dtaValueInfo->getSecond();
  }

  std::shared_ptr<DtaValueInfo> getOrCreateDtaValueInfo(llvm::Value* val) {
    if (info.contains(val))
      return info[val];
    return createDtaValueInfo(val);
  }

  bool hasDtaInfo(llvm::Value* val) { return info.find(val) != info.end(); }

  bool isConversionDisabled(llvm::Value* val) {
    if (llvm::isa<llvm::Constant>(val))
      return false;
    if (!taffoInfo.hasValueInfo(*val))
      return true;
    if (llvm::isa<llvm::Argument>(val))
      return !taffoInfo.getValueInfo(*val)->isConversionEnabled();
    return !taffoInfo.getValueInfo(*val)->isConversionEnabled() && incomingValuesDisabled(val);
  }

  bool incomingValuesDisabled(llvm::Value* v) {
    using namespace llvm;
    if (!taffo::getFullyUnwrappedType(v)->isFloatingPointTy())
      return true;

    if (auto* phi = dyn_cast<PHINode>(v)) {
      bool disabled = false;
      for (Value* inc : phi->incoming_values()) {
        if (!isa<PHINode>(inc) && isConversionDisabled(inc)) {
          disabled = true;
          break;
        }
      }
      return disabled;
    }
    return true;
  }

#ifdef TAFFO_BUILD_ILP_DTA
  bool overwriteType(std::shared_ptr<taffo::ValueInfo> old, std::shared_ptr<taffo::ValueInfo> model);
#endif // TAFFO_BUILD_ILP_DTA

  template <typename AnalysisT>
  typename AnalysisT::Result& getFunctionAnalysisResult(llvm::Function& F) {
    auto& FAM = MAM->getResult<llvm::FunctionAnalysisManagerModuleProxy>(*(F.getParent())).getManager();
    return FAM.getResult<AnalysisT>(F);
  }

  llvm::ModuleAnalysisManager& getMAM() {
    assert(MAM);
    return *MAM;
  }

  llvm::FunctionAnalysisManager& getFAM(llvm::Module& M) {
    assert(MAM);
    return MAM->getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
  }

private:
  taffo::TaffoInfo& taffoInfo = taffo::TaffoInfo::getInstance();
  llvm::ModuleAnalysisManager* MAM = nullptr;
  dataTypeAllocationStrategy* strategy = nullptr;

  std::shared_ptr<taffo::ValueInfo> getStructFieldValueInfo(std::shared_ptr<taffo::StructInfo> structInfo,
                                                            const llvm::iterator_range<const llvm::Use*> gepIndices);

  void attachStructFieldType(llvm::GetElementPtrInst* gep, const taffo::NumericTypeInfo& numericType);

  void propagateStructFieldTypes(const std::vector<llvm::Value*>& queue);
};

} // namespace tuner

#undef DEBUG_TYPE
