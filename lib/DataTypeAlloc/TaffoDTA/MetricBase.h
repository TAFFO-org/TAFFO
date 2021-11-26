#pragma once
#include "CPUCosts.h"
#include "InputInfo.h"
#include "MemSSAUtils.hpp"
#include "Model.h"
#include "OptimizerInfo.h"
#include "TaffoDTA.h"
#include "unordered_map"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Support/Casting.h"
#include "stdio.h"

namespace tuner {
class Optimizer;
class PhiWatcher;
class MemWatcher;
} // namespace tuner

class MetricBase {

protected:
  enum MetricKind { MK_Perf, MK_Size };


public:
  void setOpt(tuner::Optimizer *O) { opt = O; }
  void handleAlloca(llvm::Instruction *instruction,
                    shared_ptr<tuner::ValueInfo> valueInfo);
  shared_ptr<tuner::OptimizerInfo> processConstant(llvm::Constant *constant);
  shared_ptr<tuner::OptimizerInfo>
  handleGEPConstant(const llvm::ConstantExpr *cexp_i);

  void handleGEPInstr(llvm::Instruction *gep,
                      shared_ptr<tuner::ValueInfo> valueInfo);
  bool extractGEPOffset(
      const llvm::Type *source_element_type,
      const llvm::iterator_range<llvm::User::const_op_iterator> indices,
      std::vector<unsigned> &offset);
  void handleFCmp(llvm::Instruction *instr,
                  shared_ptr<tuner::ValueInfo> valueInfo);
  void handleReturn(llvm::Instruction *instr,
                    shared_ptr<tuner::ValueInfo> valueInfo);

  void saveInfoForPointer(llvm::Value *value,
                          shared_ptr<tuner::OptimizerPointerInfo> pointerInfo);
  void handleCall(llvm::Instruction *instruction,
                  shared_ptr<tuner::ValueInfo> valueInfo);
  void handleUnknownFunction(llvm::Instruction *instruction,
                             shared_ptr<tuner::ValueInfo> valueInfo);
    shared_ptr<tuner::OptimizerScalarInfo>
  handleBinOpCommon(llvm::Instruction *instr, llvm::Value *op1,
                    llvm::Value *op2, bool forceFixEquality,
                    shared_ptr<tuner::ValueInfo> valueInfos);
  MetricKind getKind() const noexcept { return Kind; }
  shared_ptr<tuner::OptimizerStructInfo>
  loadStructInfo(llvm::Value *glob, shared_ptr<mdutils::StructInfo> pInfo,
                 string name);
  virtual void handleDisabled(std::shared_ptr<tuner::OptimizerScalarInfo> res, const tuner::CPUCosts & cpuCosts, const char* start) = 0;
  virtual void handleFAdd(llvm::BinaryOperator *instr, const unsigned OpCode,
                          const shared_ptr<tuner::ValueInfo> &valueInfos)=0;
  virtual void handleFSub(llvm::BinaryOperator *instr, const unsigned OpCode,
                          const shared_ptr<tuner::ValueInfo> &valueInfos)=0;
  virtual void handleFMul(llvm::BinaryOperator *instr, const unsigned OpCode,
                          const shared_ptr<tuner::ValueInfo> &valueInfos)=0;
  virtual void handleFDiv(llvm::BinaryOperator *instr, const unsigned OpCode,
                          const shared_ptr<tuner::ValueInfo> &valueInfos)=0;
  virtual void handleFRem(llvm::BinaryOperator *instr, const unsigned OpCode,
                          const shared_ptr<tuner::ValueInfo> &valueInfos)=0;
  virtual shared_ptr<tuner::OptimizerScalarInfo> allocateNewVariableForValue(
      llvm::Value *value, shared_ptr<mdutils::FPType> fpInfo,
      shared_ptr<mdutils::Range> rangeInfo,
      shared_ptr<double> suggestedMinError, string functionName,
      bool insertInList = true, string nameAppendix = "",
      bool insertENOBinMin = true, bool respectFloatingPointConstraint = true) = 0;
  virtual void saveInfoForValue(llvm::Value *value,
                                shared_ptr<tuner::OptimizerInfo> optInfo)=0;
  virtual void closePhiLoop(llvm::PHINode *phiNode,
                            llvm::Value *requestedValue)=0;
  virtual void closeMemLoop(llvm::LoadInst *load, llvm::Value *requestedValue)=0;
  virtual string getEnobActivationVariable(llvm::Value *value, int cardinal)=0;
  virtual void openPhiLoop(llvm::PHINode *phiNode, llvm::Value *value)=0;
  virtual void openMemLoop(llvm::LoadInst *load, llvm::Value *value)=0;
  virtual void handleLoad(llvm::Instruction *instruction,
                          const shared_ptr<tuner::ValueInfo> &valueInfo)=0;
  virtual shared_ptr<tuner::OptimizerScalarInfo>
  allocateNewVariableWithCastCost(llvm::Value *toUse, llvm::Value *whereToUse)=0;
  virtual void handleStore(llvm::Instruction *instruction,
                           const shared_ptr<tuner::ValueInfo> &valueInfo)=0;
  virtual void handleFPPrecisionShift(llvm::Instruction *instruction,
                                      shared_ptr<tuner::ValueInfo> valueInfo)=0;
  virtual void handlePhi(llvm::Instruction *instruction,
                         shared_ptr<tuner::ValueInfo> valueInfo)=0;
  virtual void handleCastInstruction(llvm::Instruction *instruction,
                                     shared_ptr<tuner::ValueInfo> valueInfo)=0;
  virtual int getMaxIntBitOfValue(llvm::Value *pValue)=0;
  virtual int getMinIntBitOfValue(llvm::Value *pValue)=0;
  virtual void handleSelect(llvm::Instruction *instruction,
                            shared_ptr<tuner::ValueInfo> valueInfo)=0;
  virtual ~MetricBase(){};

protected:
  MetricBase(MetricKind k): Kind(k) {}
  tuner::Optimizer *opt;
  const MetricKind Kind;

  bool valueHasInfo(llvm::Value *value);
  tuner::Model &getModel();
  std::unordered_map<std::string, llvm::Function *> &
  getFunctions_still_to_visit();
  std::vector<llvm::Function *> &getCall_stack();
  llvm::DenseMap<llvm::Value *, std::shared_ptr<tuner::OptimizerInfo>> &
  getValueToVariableName();
  std::stack<shared_ptr<tuner::OptimizerInfo>> &getRetStack();
  void addDisabledSkipped();
  shared_ptr<tuner::OptimizerInfo> getInfoOfValue(llvm::Value *value);
  tuner::TaffoTuner *getTuner();
  tuner::PhiWatcher &getPhiWatcher();
  std::unordered_map<std::string, llvm::Function *> &getKnown_functions();
  tuner::CPUCosts &getCpuCosts();
  tuner::MemWatcher &getMemWatcher();

};

class MetricPerf : public MetricBase {
public:
  MetricPerf() : MetricBase(MetricKind::MK_Perf) {}

  static bool classof(const MetricBase *M) noexcept {
    return M->getKind() == MK_Perf;
  }

  void saveInfoForValue(llvm::Value *value,
                        shared_ptr<tuner::OptimizerInfo> optInfo) override;
  void closePhiLoop(llvm::PHINode *phiNode, llvm::Value *requestedValue) override;
  void closeMemLoop(llvm::LoadInst *load, llvm::Value *requestedValue) override;
  void openPhiLoop(llvm::PHINode *phiNode, llvm::Value *value) override;
  void openMemLoop(llvm::LoadInst *load, llvm::Value *value) override;

  void handleLoad(llvm::Instruction *instruction,
                  const shared_ptr<tuner::ValueInfo> &valueInfo) override;
  void handleStore(llvm::Instruction *instruction,
                   const shared_ptr<tuner::ValueInfo> &valueInfo) override;
  void handleFPPrecisionShift(llvm::Instruction *instruction,
                              shared_ptr<tuner::ValueInfo> valueInfo) override;
  void handlePhi(llvm::Instruction *instruction,
                 shared_ptr<tuner::ValueInfo> valueInfo);
  void handleCastInstruction(llvm::Instruction *instruction,
                             shared_ptr<tuner::ValueInfo> valueInfo) override;
  int getMaxIntBitOfValue(llvm::Value *pValue) override;
  int getMinIntBitOfValue(llvm::Value *pValue) override;
  void handleDisabled(std::shared_ptr<tuner::OptimizerScalarInfo> res,
                      const tuner::CPUCosts &cpuCosts,
                      const char *start) override;
  void handleFAdd(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;
  void handleFSub(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;

  void handleFMul(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;
  void handleFDiv(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;
  void handleFRem(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;

  shared_ptr<tuner::OptimizerScalarInfo> allocateNewVariableForValue(
      llvm::Value *value, shared_ptr<mdutils::FPType> fpInfo,
      shared_ptr<mdutils::Range> rangeInfo,
      shared_ptr<double> suggestedMinError, string functionName,
      bool insertInList = true, string nameAppendix = "",
      bool insertENOBinMin = true,
      bool respectFloatingPointConstraint = true) override;

  shared_ptr<tuner::OptimizerScalarInfo>
  allocateNewVariableWithCastCost(llvm::Value *toUse, llvm::Value *whereToUse) override;

protected:
  MetricPerf(MetricKind k): MetricBase(k) {}
  int getENOBFromError(double error);
  static int getENOBFromRange(const shared_ptr<mdutils::Range>& range,
                       mdutils::FloatType::FloatStandard standard);
  void handleSelect(llvm::Instruction *instruction,
                    shared_ptr<tuner::ValueInfo> valueInfo) override;
  std::string getEnobActivationVariable(llvm::Value *value, int cardinal) override;

};

/*

class MetricSize : public MetricPerf {
public:
  MetricSize() : MetricPerf(MetricKind::MK_Size) {}

  static bool classof(const MetricBase *M) noexcept {
    return M->getKind() == MK_Size;
  }

  void saveInfoForValue(llvm::Value *value,
                        shared_ptr<tuner::OptimizerInfo> optInfo) override;
  void closePhiLoop(llvm::PHINode *phiNode, llvm::Value *requestedValue) override;
  void closeMemLoop(llvm::LoadInst *load, llvm::Value *requestedValue) override;
  void openPhiLoop(llvm::PHINode *phiNode, llvm::Value *value) override;
  void openMemLoop(llvm::LoadInst *load, llvm::Value *value) override;

  void handleLoad(llvm::Instruction *instruction,
                  const shared_ptr<tuner::ValueInfo> &valueInfo) override;
  void handleStore(llvm::Instruction *instruction,
                   const shared_ptr<tuner::ValueInfo> &valueInfo) override;
  void handleFPPrecisionShift(llvm::Instruction *instruction,
                              shared_ptr<tuner::ValueInfo> valueInfo) override;
  void handlePhi(llvm::Instruction *instruction,
                 shared_ptr<tuner::ValueInfo> valueInfo);
  void handleCastInstruction(llvm::Instruction *instruction,
                             shared_ptr<tuner::ValueInfo> valueInfo) override;
  int getMaxIntBitOfValue(llvm::Value *pValue) override;
  int getMinIntBitOfValue(llvm::Value *pValue) override;
  void handleFAdd(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;
  void handleFSub(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;

  void handleFMul(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;
  void handleFDiv(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;
  void handleFRem(llvm::BinaryOperator *instr, const unsigned OpCode,
                  const shared_ptr<tuner::ValueInfo> &valueInfos) override;

  shared_ptr<tuner::OptimizerScalarInfo> allocateNewVariableForValue(
      llvm::Value *value, shared_ptr<mdutils::FPType> fpInfo,
      shared_ptr<mdutils::Range> rangeInfo,
      shared_ptr<double> suggestedMinError, string functionName,
      bool insertInList = true, string nameAppendix = "",
      bool insertENOBinMin = true,
      bool respectFloatingPointConstraint = true) override;

  shared_ptr<tuner::OptimizerScalarInfo>
  allocateNewVariableWithCastCost(llvm::Value *toUse, llvm::Value *whereToUse) override;

protected:
  int getENOBFromError(double error);
  int getENOBFromRange(shared_ptr<mdutils::Range> range,
                       mdutils::FloatType::FloatStandard standard);
  void handleSelect(llvm::Instruction *instruction,
                    shared_ptr<tuner::ValueInfo> valueInfo) override;
  std::string getEnobActivationVariable(llvm::Value *value, int cardinal) override;

};
*/
