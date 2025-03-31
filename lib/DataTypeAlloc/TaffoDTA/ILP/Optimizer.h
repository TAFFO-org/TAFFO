#ifndef __TAFFO_DTA_OPTIMIZER_H__
#define __TAFFO_DTA_OPTIMIZER_H__

#include "TaffoInfo/ValueInfo.hpp"
#include "TaffoDTA.h"
#include "CPUCosts.h"
#include "Model.h"
#include "OptimizerInfo.h"
#include "TypeUtils.h"
#include "PhiWatcher.h"
#include "MemWatcher.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Support/CommandLine.h>
#include <unordered_map>
#include <stack>

#define DEBUG_TYPE "taffo-dta"

extern bool hasDouble;
extern bool hasHalf;
extern bool hasQuad;
extern bool hasPPC128;
extern bool hasFP80;
extern bool hasBF16;

// This means how much the casting cost will be relevant for the computation
extern llvm::cl::opt<double> MixedTuningTime;
extern llvm::cl::opt<double> MixedTuningENOB;
extern llvm::cl::opt<double> MixedTuningCastingTime;
extern llvm::cl::opt<bool> MixedDoubleEnabled;
extern llvm::cl::opt<bool> MixedTripCount;
#define TUNING_CASTING (MixedTuningCastingTime)
#define TUNING_MATH (MixedTuningTime)
#define TUNING_ENOB (MixedTuningENOB)

#define FIX_DELTA_MAX 1

#define BIG_NUMBER 10000

class MetricBase;
class MetricPerf;

namespace tuner {

class Optimizer {
public:
  /// Data related to function call
  std::unordered_map<std::string, llvm::Function *> known_functions;
  std::unordered_map<std::string, llvm::Function *> functions_still_to_visit;
  std::vector<llvm::Function *> call_stack;
  std::stack<shared_ptr<OptimizerInfo>> retStack;
  std::unique_ptr<MetricBase> metric;


  llvm::DenseMap<llvm::Value *, std::shared_ptr<OptimizerInfo>> valueToVariableName;
  Model model;
  llvm::Module &module;
  TaffoTuner *tuner;

  CPUCosts cpuCosts;
  PhiWatcher phiWatcher;
  MemWatcher memWatcher;

  int DisabledSkipped;
  int StatSelectedFixed = 0;
  int StatSelectedDouble = 0;
  int StatSelectedFloat = 0;
  int StatSelectedHalf = 0;
  int StatSelectedQuad = 0;
  int StatSelectedPPC128 = 0;
  int StatSelectedFP80 = 0;
  int StatSelectedBF16 = 0;

  /*
  bool hasHalf;
  bool hasQuad;
  bool hasFP80 ;
  bool hasPPC128;
  bool hasBF16;*/

private:
  llvm::Instruction *currentInstruction;
  unsigned int currentInstructionTripCount = 1;

public:
  void handleGlobal(llvm::GlobalObject *glob, shared_ptr<TunerInfo> tunerInfo);

  bool finish();


  explicit Optimizer(llvm::Module &mm, TaffoTuner *tuner, MetricBase *met, std::string modelFile, CPUCosts::CostType cType);

  ~Optimizer();

  void initialize();

  void handleCallFromRoot(llvm::Function *f);

  std::shared_ptr<taffo::ValueInfo> getAssociatedMetadata(llvm::Value *pValue);

  void printStatInfos();

public:
  void handleInstruction(llvm::Instruction *instruction, shared_ptr<TunerInfo> valueInfo);

  /** Returns the cost of the instruction currently being processed by handleInstruction. */
  int getCurrentInstructionCost();

  void emitError(const string &stringhina);


  shared_ptr<OptimizerInfo> getInfoOfValue(llvm::Value *value);

  void
  handleBinaryInstruction(llvm::Instruction *instr, const unsigned int OpCode, const shared_ptr<TunerInfo> &valueInfos);
  void handleUnaryInstruction(llvm::Instruction *instr, const shared_ptr<TunerInfo> &valueInfos);

  void insertTypeEqualityConstraint(shared_ptr<OptimizerScalarInfo> op1, shared_ptr<OptimizerScalarInfo> op2,
                                    bool forceFixBitsConstraint);

  shared_ptr<OptimizerInfo> handleGEPConstant(const llvm::ConstantExpr *cexp_i);


  bool valueHasInfo(llvm::Value *value);

  list<shared_ptr<OptimizerInfo>> fetchFunctionCallArgumentInfo(const llvm::CallBase *call_i);
  void processFunction(llvm::Function &function, list<shared_ptr<OptimizerInfo>> argInfo, shared_ptr<OptimizerInfo> retInfo);

  void handleTerminators(llvm::Instruction *term, shared_ptr<TunerInfo> valueInfo);

  shared_ptr<OptimizerScalarInfo>
  handleBinOpCommon(llvm::Instruction *instr, llvm::Value *op1, llvm::Value *op2, bool forceFixEquality,
                    shared_ptr<TunerInfo> valueInfos);

  void saveInfoForPointer(llvm::Value *value, shared_ptr<OptimizerPointerInfo> pointerInfo);


  shared_ptr<taffo::NumericType> modelvarToTType(shared_ptr<OptimizerScalarInfo> sharedPtr);

  shared_ptr<taffo::ValueInfo> buildDataHierarchy(shared_ptr<OptimizerInfo> info);

  void handleUnknownFunction(llvm::Instruction *call_i, shared_ptr<TunerInfo> valueInfo);


  friend class MetricBase;
};

} // namespace tuner

#undef DEBUG_TYPE

#endif
