#pragma once

#include "AllocationStrategy.hpp"
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

namespace taffo {

struct DtaValueInfo {
  std::shared_ptr<NumericTypeInfo> initialType;
  std::optional<std::string> bufferID;
};

struct FunInfo {
  llvm::Function* newFun;
  /* {function argument index, type of argument}
   * argument idx is -1 for return value */
  std::vector<std::pair<int, std::shared_ptr<ValueInfo>>> fixArgs;
};

class DataTypeAllocationPass : public llvm::PassInfoMixin<DataTypeAllocationPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);

  double static getGreatest(std::shared_ptr<ScalarInfo>& scalarInfo, llvm::Value* value, Range* range);

private:
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  AllocationStrategy* strategy = nullptr;

  /* to not be accessed directly, use valueInfo() */
  llvm::DenseMap<llvm::Value*, std::shared_ptr<DtaValueInfo>> info;
  /* original function -> cloned function map */
  llvm::DenseMap<llvm::Function*, std::vector<FunInfo>> functionPool;
  /* buffer ID sets */
  std::map<std::string, llvm::SmallPtrSet<llvm::Value*, 2>> bufferIDSets;

  void setStrategy(AllocationStrategy* strategy) { this->strategy = strategy; }

  void allocateTypes(llvm::Module& m, std::vector<llvm::Value*>& values, llvm::SmallPtrSetImpl<llvm::Value*>& valueSet);
  void allocateLocalTypes(llvm::Module& m, std::vector<llvm::Value*>& values);
  void allocateGlobalTypes(llvm::Module& m, std::vector<llvm::Value*>& values);
  void allocateValueType(llvm::Value& value, std::vector<llvm::Value*>& values);

  bool allocateType(llvm::Value* value);
  bool allocateScalarType(std::shared_ptr<ScalarInfo>& scalarInfo,
                          llvm::Value* value,
                          const tda::TransparentType* transparentType,
                          bool forceEnable);
  void allocateStructType(std::shared_ptr<StructInfo>& structInfo,
                          const llvm::Value* value,
                          const tda::TransparentType* transparentType,
                          llvm::SmallVector<std::pair<std::shared_ptr<ValueInfo>, tda::TransparentType*>, 8>& queue);

  void retrieveBufferID(llvm::Value* V);

  void sortQueue(std::vector<llvm::Value*>& vals, llvm::SmallPtrSetImpl<llvm::Value*>& valset);

  void mergeTypes(const std::vector<llvm::Value*>& vals, const llvm::SmallPtrSetImpl<llvm::Value*>& valset);
  bool mergeTypes(llvm::Value* value1, llvm::Value* value2);
  bool mergeTypes(std::shared_ptr<ValueInfo> valueInfo1,
                  tda::TransparentType* type1,
                  std::shared_ptr<ValueInfo> valueInfo2,
                  tda::TransparentType* type2);
  bool mergeTypes(std::shared_ptr<ScalarInfo> scalarInfo1, std::shared_ptr<ScalarInfo> scalarInfo2);

  void mergeBufferIDSets();

  bool propagateTypeAcrossCalls(llvm::Value* value);
  bool propagateArgType(llvm::Argument* arg, const std::shared_ptr<ValueInfo>& valueInfo);
  bool propagateCallType(llvm::CallBase* call);
  bool propagateGepType(llvm::GetElementPtrInst* gep);

  bool mergeTypeWithGepPtrOperand(llvm::GetElementPtrInst* gep, const std::shared_ptr<ScalarInfo>& gepInfo);

  std::vector<llvm::Function*> collapseFunction(llvm::Module& m);

  llvm::Function* findEqFunction(llvm::Function* fun, llvm::Function* origin);

  std::shared_ptr<DtaValueInfo> createDtaValueInfo(llvm::Value* value) {
    LLVM_DEBUG(tda::log() << "new dtaValueInfo for " << *value << "\n");
    info[value] = std::make_shared<DtaValueInfo>(DtaValueInfo());
    return info[value];
  }

  std::shared_ptr<DtaValueInfo> getDtaValueInfo(llvm::Value* value) {
    auto dtaValueInfo = info.find(value);
    assert(dtaValueInfo != info.end() && "dtaValueInfo not present");
    return dtaValueInfo->getSecond();
  }

  std::shared_ptr<DtaValueInfo> getOrCreateDtaValueInfo(llvm::Value* value) {
    if (info.contains(value))
      return info[value];
    return createDtaValueInfo(value);
  }

  bool hasDtaInfo(llvm::Value* value) { return info.find(value) != info.end(); }

  bool isConversionDisabled(llvm::Value* value) {
    if (llvm::isa<llvm::Constant>(value))
      return false;
    if (!taffoInfo.hasValueInfo(*value))
      return true;
    if (llvm::isa<llvm::Argument>(value))
      return !taffoInfo.getValueInfo(*value)->isConversionEnabled();
    return !taffoInfo.getValueInfo(*value)->isConversionEnabled() && incomingValuesDisabled(value);
  }

  bool incomingValuesDisabled(llvm::Value* value) {
    using namespace llvm;
    if (!getFullyUnwrappedType(value)->isFloatingPointTy())
      return true;

    if (auto* phi = dyn_cast<PHINode>(value)) {
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
};

} // namespace taffo

#undef DEBUG_TYPE
