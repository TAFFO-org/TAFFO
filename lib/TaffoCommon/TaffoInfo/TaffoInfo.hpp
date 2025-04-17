#pragma once

#include "Types/TransparentType.hpp"
#include "ValueInfo.hpp"
#include "../Containers/BiMap.hpp"
#include "llvm/IR/GlobalValue.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/Analysis/LoopInfo.h>

namespace taffo {

/**
 * Singleton class to maintain Taffo analysis information.
 * Supports serialization/deserialization to/from JSON for debug purposes.
 */
class TaffoInfo : public Serializable {
public:
  TaffoInfo(const TaffoInfo&) = delete;
  TaffoInfo &operator=(const TaffoInfo&) = delete;

  static TaffoInfo &getInstance();

  void setTransparentType(llvm::Value &v, const std::shared_ptr<TransparentType> &t);
  std::shared_ptr<TransparentType> getOrCreateTransparentType(llvm::Value &v);
  std::shared_ptr<TransparentType> getTransparentType(llvm::Value &v) const;
  bool hasTransparentType(const llvm::Value &v);

  void addStartingPoint(llvm::Function &f);
  void addDefaultStartingPoint(llvm::Module &m);
  bool isStartingPoint(llvm::Function &f) const;
  bool hasStartingPoint(llvm::Module &m) const;

  void setIndirectFunction(llvm::CallInst &call, llvm::Function &f);
  llvm::Function *getIndirectFunction(const llvm::CallInst &call) const;
  bool isIndirectFunction(const llvm::CallInst &call) const;

  void setOpenCLTrampoline(llvm::Function &f, llvm::Function &kernF);
  llvm::Function *getOpenCLTrampoline(const llvm::Function &f) const;
  bool isOpenCLTrampoline(const llvm::Function &f) const;

  void disableConversion(llvm::Instruction &i);
  bool isConversionDisabled(llvm::Instruction &i) const;

  void createValueInfo(llvm::Value &v);
  void setValueInfo(llvm::Value &v, const std::shared_ptr<ValueInfo> &vi);
  void setValueInfo(llvm::Value &v, std::shared_ptr<ValueInfo> &&vi);

  std::shared_ptr<ValueInfo> getValueInfo(const llvm::Value &v) const;
  bool hasValueInfo(const llvm::Value &v) const;

  void setValueWeight(llvm::Value &v, int weight);
  int getValueWeight(const llvm::Value &v) const;

  void setTaffoFunction(llvm::Function &originalF, llvm::Function &taffoF);
  void getTaffoCloneFunctions(const llvm::Function &originalF, llvm::SmallPtrSetImpl<llvm::Function*> &taffoFunctions) const; 
  bool isOriginalFunction(const llvm::Function &originalF) const;
  bool isTaffoCloneFunction(llvm::Function &f) const;
  void setOriginalFunctionLinkage(llvm::Function& originalF, llvm::GlobalValue::LinkageTypes linkage);
  llvm::GlobalValue::LinkageTypes getOriginalFunctionLinkage(const llvm::Function& originalF) const;

  void setMaxRecursionCount(llvm::Function &f, unsigned int maxRecursion);
  unsigned int getMaxRecursionCount(const llvm::Function &f) const;

  void setLoopUnrollCount(llvm::Loop &l, unsigned int unrollCount);
  unsigned int getLoopUnrollCount(const llvm::Loop &l) const;

  void setError(llvm::Instruction &i, double error);
  double getError(llvm::Instruction &i) const;

  void setCmpErrorMetadata(llvm::Instruction &i, CmpErrorInfo &compErrorInfo);
  std::shared_ptr<CmpErrorInfo> getCmpError(const llvm::Instruction &i) const;

  llvm::Type *getType(const std::string &typeId) const;

  void eraseValue(llvm::Value &v);
  void eraseLoop(llvm::Loop &l);

  void dumpToFile(const std::string &filePath, llvm::Module &m);
  void initializeFromFile(const std::string &filePath, llvm::Module &m);

private:
  llvm::DenseMap<llvm::Value*, std::shared_ptr<TransparentType>> transparentTypes;

  llvm::SmallVector<llvm::Function*> startingPoints;
  llvm::SmallDenseMap<llvm::CallInst*, llvm::Function*> indirectFunctions;
  llvm::SmallDenseMap<llvm::Function*, llvm::Function*> oclTrampolines;
  llvm::SmallVector<llvm::Instruction*> disabledConversion;

  BiMap<llvm::Function*, llvm::Function*> taffoCloneToOriginalFunction;
  llvm::DenseMap<llvm::Function*, llvm::SmallPtrSet<llvm::Function*, 2>> originalToTaffoCloneFunctions;
  llvm::DenseMap<llvm::Function*, llvm::GlobalValue::LinkageTypes> originalFunctionLinkage;

  llvm::DenseMap<llvm::Value*, std::shared_ptr<ValueInfo>> valueInfo;
  llvm::DenseMap<llvm::Value*, int> valueWeights;

  llvm::DenseMap<llvm::Function*, unsigned int> maxRecursionCount;
  llvm::DenseMap<llvm::Loop*, unsigned int> loopUnrollCount;
  llvm::DenseMap<llvm::Instruction*, double> error;
  llvm::DenseMap<llvm::Instruction*, std::shared_ptr<CmpErrorInfo>> cmpError;

  llvm::SmallPtrSet<llvm::Value*, 4> erasedValues;
  llvm::SmallPtrSet<llvm::Loop*, 4> erasedLoops;

  BiMap<std::string, llvm::Value*> idValueMapping;
  BiMap<std::string, llvm::Loop*> idLoopMapping;
  BiMap<std::string, llvm::Type*> idTypeMapping;
  unsigned int idCounter;
  unsigned int idDigits;
  json jsonRepresentation;
  std::string logContextTag = "TaffoInfo";

  TaffoInfo() : idCounter(0), idDigits(1) {}

  void generateTaffoIds();
  void generateTaffoId(llvm::Value *v);
  void generateTaffoId(llvm::Loop *l);
  std::string generateValueId(const llvm::Value *v);
  std::string generateLoopId(const llvm::Loop *l);
  void updateIdDigits();

  json serialize() const override;
  void deserialize(const json &j) override;
};

} // namespace taffo
