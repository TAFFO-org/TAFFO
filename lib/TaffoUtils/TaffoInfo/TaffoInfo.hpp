#ifndef TAFFO_TAFFOINFO_HPP
#define TAFFO_TAFFOINFO_HPP

#include "TypeDeducer/DeducedPointerType.hpp"
#include "ValueInfo.hpp"
#include "../BiMap.hpp"

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

  void setDeducedPointerTypes(const llvm::DenseMap<llvm::Value*, DeducedPointerType> &deducedPointerTypes);
  void setDeducedPointerType(llvm::Value &v, const DeducedPointerType &t);
  std::optional<DeducedPointerType> getDeducedPointerType(const llvm::Value &v) const;

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

  void setValueInfo(llvm::Value &v, const std::shared_ptr<ValueInfo> &vi);
  void setConstantInfo(llvm::Constant &constant, llvm::Instruction &userInst, const std::shared_ptr<ScalarInfo> &constInfo);
  std::shared_ptr<ValueInfo> getValueInfo(const llvm::Value &v) const;
  bool hasValueInfo(const llvm::Value &v) const;

  void setValueWeight(llvm::Value &v, int weight);
  int getValueWeight(const llvm::Value &v) const;

  void setTaffoFunction(llvm::Function &originalF, llvm::Function &taffoF);
  llvm::Function *getOriginalFunction(llvm::Function &taffoF) const;
  bool hasTaffoFunctions(const llvm::Function &originalF) const;
  bool isTaffoFunction(llvm::Function &f) const;
  void getTaffoFunctions(const llvm::Function &originalF, llvm::SmallPtrSetImpl<llvm::Function*> &taffoFunctions) const;

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
  void eraseValue(llvm::Function &f);

  void dumpToFile(const std::string &filePath, llvm::Module &m);
  void initializeFromFile(const std::string &filePath, llvm::Module &m);

private:
  llvm::DenseMap<llvm::Value*, DeducedPointerType> deducedPointerTypes;

  llvm::SmallVector<llvm::Function*> startingPoints;
  llvm::SmallDenseMap<llvm::CallInst*, llvm::Function*> indirectFunctions;
  llvm::SmallDenseMap<llvm::Function*, llvm::Function*> oclTrampolines;
  llvm::SmallVector<llvm::Instruction*> disabledConversion;

  BiMap<llvm::Function*, llvm::Function*> originalFunctions;
  llvm::DenseMap<llvm::Function*, llvm::SmallPtrSet<llvm::Function*, 2>> taffoFunctions;

  llvm::DenseMap<llvm::Value*, std::shared_ptr<ValueInfo>> valueInfo;
  llvm::DenseMap<llvm::Value*, int> valueWeights;
  llvm::DenseMap<llvm::Constant*, llvm::SmallPtrSet<llvm::Instruction*, 2>> constantUsers;

  llvm::DenseMap<llvm::Function*, unsigned int> maxRecursionCount;
  llvm::DenseMap<llvm::Loop*, unsigned int> loopUnrollCount;
  llvm::DenseMap<llvm::Instruction*, double> error;
  llvm::DenseMap<llvm::Instruction*, std::shared_ptr<CmpErrorInfo>> cmpError;

  llvm::SmallPtrSet<llvm::Value*, 4> erasedValues;

  BiMap<std::string, llvm::Value*> idValueMapping;
  BiMap<std::string, llvm::Loop*> idLoopMapping;
  BiMap<std::string, llvm::Type*> idTypeMapping;
  unsigned int idCounter;
  unsigned int idDigits;
  json jsonRepresentation;

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

#endif // TAFFO_TAFFOINFO_HPP
