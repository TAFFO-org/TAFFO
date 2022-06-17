#ifndef __TAFFO_DTA_OPTIMIZER_H__
#define __TAFFO_DTA_OPTIMIZER_H__

#include "CPUCosts.h"
#include "InputInfo.h"
#include "Metadata.h"
#include "Model.h"
#include "OptimizerInfo.h"
#include "TaffoDTA.h"
#include "TypeUtils.h"
#include "PhiWatcher.h"
#include "MemWatcher.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include <PtrCasts.h>
#include <stack>
#include <unordered_map>

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


using namespace llvm;

class MetricBase;
class MetricPerf;

namespace tuner
{

class Optimizer
{
public:
  /// Data related to function call
  std::unordered_map<std::string, llvm::Function *> known_functions;
  std::unordered_map<std::string, llvm::Function *> functions_still_to_visit;
  std::vector<llvm::Function *> call_stack;
  std::stack<shared_ptr<OptimizerInfo>> retStack;
  std::unique_ptr<MetricBase> metric;


  DenseMap<llvm::Value *, std::shared_ptr<OptimizerInfo>> valueToVariableName;
  DenseMap<std::pair<const llvm::User *, llvm::Value *>, std::shared_ptr<OptimizerInfo>> constantInfo;
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
  Instruction *currentInstruction;
  unsigned int currentInstructionTripCount = 1;

public:
  void handleGlobal(GlobalObject *glob, shared_ptr<ValueInfo> valueInfo);

  bool finish();


  explicit Optimizer(llvm::Module &mm, TaffoTuner *tuner, MetricBase *met, std::string modelFile, CPUCosts::CostType cType);

  ~Optimizer();

  void initialize();

  void handleCallFromRoot(Function *f);

  std::shared_ptr<mdutils::MDInfo> getAssociatedMetadata(Value *pValue);

  void printStatInfos();

public:
  void handleInstruction(Instruction *instruction, shared_ptr<ValueInfo> valueInfo);

  /** Returns the cost of the instruction currently being processed by handleInstruction. */
  unsigned getCurrentInstructionCost();

  void emitError(const string &stringhina);


  shared_ptr<OptimizerInfo> getInfoOfValue(Value *value, const User *user);

  void
  handleBinaryInstruction(Instruction *instr, const unsigned int OpCode, const shared_ptr<ValueInfo> &valueInfos);
  void handleUnaryInstruction(Instruction *instr, const shared_ptr<ValueInfo> &valueInfos);

  void insertTypeEqualityConstraint(shared_ptr<OptimizerScalarInfo> op1, shared_ptr<OptimizerScalarInfo> op2,
                                    bool forceFixBitsConstraint);

  shared_ptr<OptimizerInfo> handleGEPConstant(const ConstantExpr *cexp_i);


  bool valueHasInfo(Value *value);

  list<shared_ptr<OptimizerInfo>> fetchFunctionCallArgumentInfo(const CallBase *call_i);
  void processFunction(Function &function, list<shared_ptr<OptimizerInfo>> argInfo, shared_ptr<OptimizerInfo> retInfo);

  void handleTerminators(Instruction *term, shared_ptr<ValueInfo> valueInfo);

  shared_ptr<OptimizerScalarInfo>
  handleBinOpCommon(Instruction *instr, Value *op1, Value *op2, bool forceFixEquality,
                    shared_ptr<ValueInfo> valueInfos);

  void saveInfoForPointer(Value *value, shared_ptr<OptimizerPointerInfo> pointerInfo);


  shared_ptr<mdutils::TType> modelvarToTType(shared_ptr<OptimizerScalarInfo> sharedPtr);

  shared_ptr<mdutils::MDInfo> buildDataHierarchy(shared_ptr<OptimizerInfo> info);

  void handleUnknownFunction(Instruction *call_i, shared_ptr<ValueInfo> valueInfo);


  friend class MetricBase;
};

} // namespace tuner

#endif
