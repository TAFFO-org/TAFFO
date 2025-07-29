#include "BufferIDFiles.h"
#include "DTAConfig.hpp"
#include "DataTypeAllocationPass.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/NumericInfo.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueInfo.hpp"
#include "Utils/PtrCasts.hpp"
#ifdef TAFFO_BUILD_ILP_DTA
#include "ILP/MetricBase.h"
#include "ILP/Optimizer.h"
#endif // TAFFO_BUILD_ILP_DTA

#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/AbstractCallSite.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Debug.h>

using namespace llvm;
using namespace tda;
using namespace taffo;
using namespace tuner;

#define DEBUG_TYPE "taffo-dta"

STATISTIC(FixCast, "Number of fixed point format cast");

/* the strategy map that associates the global variable values (set through the command line arguments)
 * to the strategy object that held the methods for the chosen strategy. When adding a new strategy
 * you should add a new entry in the map */
std::map<DtaStrategyType, std::function<dataTypeAllocationStrategy*()>> strategyMap = {
  {fixedPointOnly,    []() -> fixedPointOnlyStrategy* { return new fixedPointOnlyStrategy(); }        },
  {floatingPointOnly, []() -> floatingPointOnlyStrategy* { return new floatingPointOnlyStrategy(); }  },
  {fixedFloating,     []() -> fixedFloatingPointStrategy* { return new fixedFloatingPointStrategy(); }}
};

PreservedAnalyses DataTypeAllocationPass::run(Module& m, ModuleAnalysisManager& AM) {
  LLVM_DEBUG(log().logln("[DataTypeAllocationPass]", Logger::Magenta));
  taffoInfo.initializeFromFile(VRA_TAFFO_INFO, m);

  MAM = &AM;

  // method that allocate the strategy object, whose methods will be used to apply the strategy
  setStrategy(strategyMap[DtaStrategy]());

  std::vector<Value*> values;
  SmallPtrSet<Value*, 8U> valueSet;
  dataTypeAllocation(m, values, valueSet);

#ifdef TAFFO_BUILD_ILP_DTA
  if (MixedMode) {
    LLVM_DEBUG(
    log() << "Model " << CostModelFilename << "\n");
    LLVM_DEBUG(
    log() << "Inst " << InstructionSet << "\n");
    buildModelAndOptimze(m, vals, valset);
  }
  else {
    mergeFixFormat(vals, valset);
  }
#else
  mergeFixFormat(values, valueSet);
#endif

  mergeBufferIDSets();

  std::vector<Function*> toDelete = collapseFunction(m);

  LLVM_DEBUG(log() << "attaching metadata\n");
  attachFPMetaData(values);
  attachFunctionMetaData(m);

  for (Function* f : toDelete)
    taffoInfo.eraseValue(f);

  taffoInfo.dumpToFile(DTA_TAFFO_INFO, m);
  LLVM_DEBUG(log().logln("[End of DataTypeAllocationPass]", Logger::Magenta));
  return PreservedAnalyses::all();
}

// *** STRATEGIES IMPLEMENTATIONS ***
// (for each strategy, implement the apply, merge and isMergeable methods)

bool fixedPointOnlyStrategy::apply(std::shared_ptr<ScalarInfo>& scalarInfo, Value* value) {
  if (!scalarInfo->isConversionEnabled()) {
    LLVM_DEBUG(log() << "conversion disabled: skipping\n");
    return false;
  }

  if (scalarInfo->numericType) {
    LLVM_DEBUG(log() << "numeric type already assigned\n");
    return true;
  }

  Range* range = scalarInfo->range.get();
  if (!range) {
    LLVM_DEBUG(log() << "no range: skipping\n");
    return false;
  }

  double greatest = DataTypeAllocationPass::getGreatest(scalarInfo, value, range);
  LLVM_DEBUG(log() << "maximum value involved: " << greatest << "\n");

  FixedPointTypeGenError fpgerr;

  /* Testing maximum type for operands, not deciding type yet */
  fixedPointTypeFromRange(Range(0, greatest), &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
  if (fpgerr == FixedPointTypeGenError::NoError) {
    FixedPointInfo res = fixedPointTypeFromRange(*range, &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
    if (fpgerr == FixedPointTypeGenError::NoError) {
      LLVM_DEBUG(log().log("converting to ").logln(res, Logger::Green));
      scalarInfo->numericType = res.clone();
      return true;
    }

    LLVM_DEBUG(
      Logger& logger = log();
      logger << Logger::Red << "Error generating fixed point type: ";
      switch (fpgerr) {
      case FixedPointTypeGenError::InvalidRange:            logger << "invalid range\n"; break;
      case FixedPointTypeGenError::UnboundedRange:          logger << "unbounded range\n"; break;
      case FixedPointTypeGenError::NotEnoughIntAndFracBits:
      case FixedPointTypeGenError::NotEnoughFracBits:       logger << "result not representable\n"; break;
      default:                                              logger << "error code unknown\n";
      }
      logger << Logger::Reset;);
  }
  else
    LLVM_DEBUG(log() << "operands not representable as fixed point with specified constraints\n");

  /* We failed, try to keep original type */
  Type* type = getFullyUnwrappedType(value);
  if (type->isFloatingPointTy()) {
    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(type->getTypeID(), greatest));
    scalarInfo->numericType = res;
    LLVM_DEBUG(log() << "keeping original type: " << *res << "\n");
    return true;
  }

  LLVM_DEBUG(log() << "original type was not floating point: skipping\n");
  return false;
}

std::shared_ptr<NumericTypeInfo> fixedPointOnlyStrategy::merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                                               const std::shared_ptr<NumericTypeInfo>& fpu) {

  std::shared_ptr<FixedPointInfo> fpv_fixed = dynamic_ptr_cast<FixedPointInfo>(fpv);
  std::shared_ptr<FixedPointInfo> fpu_fixed = dynamic_ptr_cast<FixedPointInfo>(fpu);

  int sign_v = fpv_fixed->isSigned() ? 1 : 0;
  int int_v = int(fpv_fixed->getBits()) - fpv_fixed->getFractionalBits() - sign_v;
  int sign_u = fpu_fixed->isSigned() ? 1 : 0;
  int int_u = int(fpu_fixed->getBits()) - fpu_fixed->getFractionalBits() - sign_u;

  int sign_res = std::max(sign_u, sign_v);
  int int_res = std::max(int_u, int_v);
  int size_res = std::max(fpv_fixed->getBits(), fpu_fixed->getBits());
  int frac_res = size_res - int_res - sign_res;
  if (sign_res + int_res + frac_res != size_res || frac_res < 0)
    return nullptr; // Invalid format.
  else
    return std::make_shared<FixedPointInfo>(sign_res, size_res, frac_res);
}

bool fixedPointOnlyStrategy::isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                                         std::shared_ptr<NumericTypeInfo> userNumericType) {

  std::shared_ptr<FixedPointInfo> fpv = dynamic_ptr_cast<FixedPointInfo>(valueNumericType);
  std::shared_ptr<FixedPointInfo> fpu = dynamic_ptr_cast<FixedPointInfo>(userNumericType);
  if (!fpv || !fpu) {
    LLVM_DEBUG(log() << "not attempting merge of " << valueNumericType->toString() << ", "
                     << valueNumericType->toString() << " because one is not a FPType\n");
    return false;
  }

  if (fpv != fpu)
    return false;

  return fpv->getBits() == fpu->getBits()
      && (std::abs(int(fpv->getFractionalBits()) - int(fpu->getFractionalBits()))
          + (fpv->isSigned() == fpu->isSigned() ? 0 : 1))
           <= SimilarBits;
}

bool floatingPointOnlyStrategy::apply(std::shared_ptr<ScalarInfo>& scalarInfo, Value* value) {
  if (!scalarInfo->isConversionEnabled()) {
    LLVM_DEBUG(log() << "conversion disabled: skipping\n");
    return false;
  }

  if (scalarInfo->numericType) {
    LLVM_DEBUG(log() << "numeric type already assigned: skipping\n");
    return true;
  }

  Range* rng = scalarInfo->range.get();
  if (rng == nullptr) {
    LLVM_DEBUG(log() << "no range: skipping\n");
    return false;
  }

  double greatest = DataTypeAllocationPass::getGreatest(scalarInfo, value, rng);

  FloatingPointInfo::FloatStandard standard;
  if (UseFloat == "f16")
    standard = FloatingPointInfo::Float_half;
  else if (UseFloat == "f32")
    standard = FloatingPointInfo::Float_float;
  else if (UseFloat == "f64")
    standard = FloatingPointInfo::Float_double;
  else if (UseFloat == "bf16")
    standard = FloatingPointInfo::Float_bfloat;
  else {
    errs() << "Invalid format " << UseFloat << " specified to the -usefloat argument\n";
    abort();
  }
  // // auto standard = static_cast<mdutils::FloatType::FloatStandard>(ForceFloat.getValue());

  // standard = FloatingPointInfo::Float_double;

  auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(standard, greatest));
  double maxRep =
    std::max(std::abs(res->getMaxValueBound().convertToDouble()), std::abs(res->getMinValueBound().convertToDouble()));
  LLVM_DEBUG(log() << "maximum value representable: " << maxRep << "\n");

  if (greatest >= maxRep) {
    LLVM_DEBUG(log() << "cannot force conversion to float " << res << " because max value is not representable\n");
  }
  else {
    LLVM_DEBUG(log() << "forcing conversion to float\n");
    scalarInfo->numericType = res;
    return true;
  }

  /* We failed, try to keep original type */
  Type* type = getFullyUnwrappedType(value);
  if (type->isFloatingPointTy()) {
    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(type->getTypeID(), greatest));
    scalarInfo->numericType = res;
    LLVM_DEBUG(log() << "keeping original type " << *res << "\n");
    return true;
  }

  LLVM_DEBUG(log() << "the original type was not floating point: skipping\n");
  return false;
}

std::shared_ptr<NumericTypeInfo> floatingPointOnlyStrategy::merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                                                  const std::shared_ptr<NumericTypeInfo>& fpu) {
  if (isa<FloatingPointInfo>(fpu.get()) && isa<FloatingPointInfo>(fpv.get())) {
    std::shared_ptr<FloatingPointInfo> a = dynamic_ptr_cast<FloatingPointInfo>(fpu);
    std::shared_ptr<FloatingPointInfo> b = dynamic_ptr_cast<FloatingPointInfo>(fpv);
    FloatingPointInfo::FloatStandard maxStd = std::max(a->getStandard(), b->getStandard());
    double maxMax = std::max(a->getGreatestNumber(), b->getGreatestNumber());
    return std::make_shared<FloatingPointInfo>(maxStd, maxMax);
  }
  llvm_unreachable("unknown numericType subclass");
}

// dunmmy strategy, always return true
bool floatingPointOnlyStrategy::isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                                            std::shared_ptr<NumericTypeInfo> userNumericType) {

  std::shared_ptr<FloatingPointInfo> fpv = dynamic_ptr_cast<FloatingPointInfo>(valueNumericType);
  std::shared_ptr<FloatingPointInfo> fpu = dynamic_ptr_cast<FloatingPointInfo>(userNumericType);
  if (!fpv || !fpu) {
    LLVM_DEBUG(log() << "not attempting merge of " << valueNumericType->toString() << ", "
                     << valueNumericType->toString() << " because one is not a FixedPointType\n");
    return false;
  }

  return true;
}

// dummy strategy, use fixed if integer part of rng-> max is even use floating otherwise
bool fixedFloatingPointStrategy::apply(std::shared_ptr<ScalarInfo>& scalarInfo, Value* value) {
  if (!scalarInfo->isConversionEnabled()) {
    LLVM_DEBUG(log() << "conversion disabled: skipping\n");
    return false;
  }

  if (scalarInfo->numericType) {
    LLVM_DEBUG(log() << "numeric type already assigned: skipping\n");
    return true;
  }

  Range* rng = scalarInfo->range.get();
  if (rng == nullptr) {
    LLVM_DEBUG(log() << "no range: skipping\n");
    return false;
  }

  double greatest = DataTypeAllocationPass::getGreatest(scalarInfo, value, rng);

  if ((int) rng->max % 2 == 0) {
    FixedPointTypeGenError fpgerr;

    /* Testing maximum type for operands, not deciding type yet */
    fixedPointTypeFromRange(Range(0, greatest), &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
    if (fpgerr == FixedPointTypeGenError::NoError) {
      FixedPointInfo res = fixedPointTypeFromRange(*rng, &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
      if (fpgerr == FixedPointTypeGenError::NoError) {
        LLVM_DEBUG(log().log("converting to ").logln(res, Logger::Green));
        scalarInfo->numericType = res.clone();
        return true;
      }

      LLVM_DEBUG(
        Logger& logger = log();
        logger << Logger::Red << "error generating fixed point type: \n";
        switch (fpgerr) {
        case FixedPointTypeGenError::InvalidRange:            logger << "invalid range\n"; break;
        case FixedPointTypeGenError::UnboundedRange:          logger << "unbounded range\n"; break;
        case FixedPointTypeGenError::NotEnoughIntAndFracBits:
        case FixedPointTypeGenError::NotEnoughFracBits:       logger << "result not representable\n"; break;
        default:                                              logger << "error code unknown\n";
        }
        logger << Logger::Reset;);
    }
    else
      LLVM_DEBUG(log() << "operands not representable as fixed point with specified constraints\n");
  }
  else {

    FloatingPointInfo::FloatStandard standard;
    if (UseFloat == "f16")
      standard = FloatingPointInfo::Float_half;
    else if (UseFloat == "f32")
      standard = FloatingPointInfo::Float_float;
    else if (UseFloat == "f64")
      standard = FloatingPointInfo::Float_double;
    else if (UseFloat == "bf16")
      standard = FloatingPointInfo::Float_bfloat;
    else {
      errs() << "Invalid format " << UseFloat << " specified to the -usefloat argument\n";
      abort();
    }

    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(standard, greatest));
    double maxRep = std::max(std::abs(res->getMaxValueBound().convertToDouble()),
                             std::abs(res->getMinValueBound().convertToDouble()));
    LLVM_DEBUG(log() << "maximum value representable: " << maxRep << "\n");

    if (greatest >= maxRep) {
      LLVM_DEBUG(log() << "cannot force conversion to float " << res << " because max value is not representable\n");
    }
    else {
      LLVM_DEBUG(log() << "forcing conversion to float " << res << "\n");
      scalarInfo->numericType = res;
      return true;
    }
  }

  /* We failed, try to keep original type */
  Type* Ty = getFullyUnwrappedType(value);
  if (Ty->isFloatingPointTy()) {
    auto res = std::make_shared<FloatingPointInfo>(FloatingPointInfo(Ty->getTypeID(), greatest));
    scalarInfo->numericType = res;
    LLVM_DEBUG(log() << "keeping original type " << res << "\n");
    return true;
  }

  LLVM_DEBUG(log() << "original type was not floating point: skipping\n");
  return false;
}

// dunmmy strategy, always return true
bool fixedFloatingPointStrategy::isMergeable(std::shared_ptr<NumericTypeInfo> valueNumericType,
                                             std::shared_ptr<NumericTypeInfo> userNumericType) {

  return true;
}

// dummy strategy, always return the fpu type
std::shared_ptr<NumericTypeInfo> fixedFloatingPointStrategy::merge(const std::shared_ptr<NumericTypeInfo>& fpv,
                                                                   const std::shared_ptr<NumericTypeInfo>& fpu) {
  if (isa<FloatingPointInfo>(fpu.get()))
    return dynamic_ptr_cast<FloatingPointInfo>(fpu)->clone();
  else
    return dynamic_ptr_cast<FixedPointInfo>(fpu)->clone();
}

// *** END OF STRATEGIES IMPLEMENTATIONS ***

double DataTypeAllocationPass::getGreatest(std::shared_ptr<ScalarInfo>& scalarInfo, Value* value, Range* range) {

  double greatest = std::max(std::abs(range->min), std::abs(range->max));
  auto* I = dyn_cast<Instruction>(value);
  if (I) {
    TaffoInfo& taffoInfo = TaffoInfo::getInstance();
    auto getRange = [&taffoInfo](Value* v) -> std::shared_ptr<Range> {
      if (!taffoInfo.hasValueInfo(*v))
        return nullptr;
      std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*v));
      if (!scalarInfo)
        return nullptr;
      return scalarInfo->range;
    };

    if (I->isBinaryOp() || I->isUnaryOp()) {
      Value* firstOperand = I->getOperand(0U);
      if (std::shared_ptr<Range> range = getRange(firstOperand))
        greatest = std::max(greatest, std::max(std::abs(range->max), std::abs(range->min)));
      else
        LLVM_DEBUG(log() << "[Warning] No range metadata found on first operand of " << *I << "\n");
    }
    if (I->isBinaryOp()) {
      Value* secondOperand = I->getOperand(1U);
      if (std::shared_ptr<Range> range = getRange(secondOperand))
        greatest = std::max(greatest, std::max(std::abs(range->max), std::abs(range->min)));
      else
        LLVM_DEBUG(log() << "[Warning] No range metadata found on second operand of " << *I << "\n");
    }
  }

  return greatest;
}

void DataTypeAllocationPass::dataTypeAllocationOfValue(Value& value, std::vector<Value*>& values) {
  if (processValueInfo(&value)) {
    values.push_back(&value);
    retrieveBufferID(&value);
  }
}

void DataTypeAllocationPass::dataTypeAllocationOfGlobals(Module& m, std::vector<Value*>& values) {
  for (GlobalObject& globObj : m.globals())
    dataTypeAllocationOfValue(globObj, values);
  LLVM_DEBUG(log() << "\n");
}

void DataTypeAllocationPass::dataTypeAllocationOfArguments(Function& f, std::vector<Value*>& values) {
  for (Argument& arg : f.args())
    dataTypeAllocationOfValue(arg, values);
}

void DataTypeAllocationPass::dataTypeAllocationOfInstructions(Function& f, std::vector<Value*>& values) {
  for (Instruction& inst : instructions(f))
    dataTypeAllocationOfValue(inst, values);
}

void DataTypeAllocationPass::dataTypeAllocationOfFunctions(Module& m, std::vector<Value*>& values) {
  for (Function& f : m.functions()) {
    if (f.isIntrinsic())
      continue;
    if (!taffoInfo.isTaffoCloneFunction(f) && !taffoInfo.isStartingPoint(f)) {
      LLVM_DEBUG(log() << " Skip function " << f.getName() << " as it is not a cloned function\n";);
      continue;
    }
    LLVM_DEBUG(log().log("[DataTypeAllocation of function] ", Logger::Blue).logValueln(&f));
    dataTypeAllocationOfArguments(f, values);
    dataTypeAllocationOfInstructions(f, values);
    LLVM_DEBUG(log() << "\n");
  }
}

void DataTypeAllocationPass::dataTypeAllocation(Module& m,
                                                std::vector<Value*>& values,
                                                SmallPtrSetImpl<Value*>& valueSet) {
  LLVM_DEBUG(log().logln("[DataTypeAllocation of globals]", Logger::Blue));
  dataTypeAllocationOfGlobals(m, values);
  dataTypeAllocationOfFunctions(m, values);

  LLVM_DEBUG(log().logln("[Sorting queue]", Logger::Blue));
  sortQueue(values, valueSet);
}

/**
 * Reads metadata for a value and DOES THE ACTUAL DATA TYPE ALLOCATION.
 * Yes you read that right.
 */
void DataTypeAllocationPass::retrieveBufferID(Value* V) {
  LLVM_DEBUG(log() << "Looking up buffer id of " << *V << "\n");
  auto MaybeBID = taffoInfo.getValueInfo(*V)->getBufferId();
  if (MaybeBID.has_value()) {
    std::string Tag = *MaybeBID;
    auto& Set = bufferIDSets[Tag];
    Set.insert(V);
    LLVM_DEBUG(log() << "Found buffer ID '" << Tag << "' for " << *V << "\n");
    if (hasDtaInfo(V))
      getDtaValueInfo(V)->bufferID = Tag;
  }
  else {
    LLVM_DEBUG(log() << "No buffer ID for " << *V << "\n");
  }
}

bool DataTypeAllocationPass::processScalarInfo(std::shared_ptr<ScalarInfo>& scalarInfo,
                                               Value* value,
                                               const std::shared_ptr<TransparentType>& transparentType,
                                               bool forceEnable) {
  if (forceEnable)
    scalarInfo->conversionEnabled = true;

  // FIXME: hack to propagate itofp metadata
  if (/*MixedMode && */ isa<UIToFPInst>(value) || isa<SIToFPInst>(value)) {
    LLVM_DEBUG(log() << "FORCING CONVERSION OF A ITOFP!\n";);
    scalarInfo->conversionEnabled = true;
  }

  if (!transparentType->containsFloatingPointType()) {
    LLVM_DEBUG(log() << "value is not a float: skipping\n");
    return false;
  }

  // TODO: insert logic here to associate different types in a clever way
  return strategy->apply(scalarInfo, value);
}

void DataTypeAllocationPass::processStructInfo(
  std::shared_ptr<StructInfo>& structInfo,
  Value* value,
  const std::shared_ptr<TransparentType>& transparentType,
  SmallVector<std::pair<std::shared_ptr<ValueInfo>, std::shared_ptr<TransparentType>>, 8> queue) {
  if (!transparentType->isStructType()) {
    LLVM_DEBUG(log() << "[ERROR] found non conforming structInfo " << structInfo->toString() << " on value " << *value
                     << "\n");
    LLVM_DEBUG(log() << "contained type " << *transparentType << " is not a struct type\n");
    LLVM_DEBUG(log() << "The top-level MDInfo was " << structInfo->toString() << "\n");
    llvm_unreachable("Non-conforming StructInfo.");
  }
  for (unsigned i = 0; i < structInfo->getNumFields(); i++)
    if (const std::shared_ptr<ValueInfo>& field = structInfo->getField(i))
      queue.push_back(
        std::make_pair(field, std::static_ptr_cast<TransparentStructType>(transparentType)->getFieldType(i)));
}

bool DataTypeAllocationPass::processValueInfo(Value* value) {
  std::shared_ptr<ValueInfo> valueInfo;
  if (taffoInfo.hasValueInfo(*value))
    valueInfo = taffoInfo.getValueInfo(*value);

  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger.log("[Value] ", Logger::Bold).logValueln(value);
    indenter.increaseIndent(););

  if (!valueInfo) {
    LLVM_DEBUG(log() << "no valueInfo: skipping\n");
    return false;
  }
  LLVM_DEBUG(logger << "valueInfo: " << *valueInfo << "\n");

  std::shared_ptr<ValueInfo> newValueInfo = valueInfo->clone();

  if (value->getType()->isVoidTy()) {
    LLVM_DEBUG(log() << "value has void type: skipping\n");
    return true;
  }

  /* HACK to set the enabled status on phis which compensates for a bug in vra.
   * Affects axbench/sobel. */
  bool forceEnableConv = false;
  if (isa<PHINode>(value) && !isConversionDisabled(value) && isa<ScalarInfo>(newValueInfo.get()))
    forceEnableConv = true;

  bool skippedAll = true;
  std::shared_ptr<TransparentType> transparentType = taffoInfo.getOrCreateTransparentType(*value);
  SmallVector<std::pair<std::shared_ptr<ValueInfo>, std::shared_ptr<TransparentType>>, 8> queue(
    {std::make_pair(newValueInfo, transparentType)});

  while (!queue.empty()) {
    const auto& [valueInfo, transparentType] = queue.pop_back_val();

    if (std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast<ScalarInfo>(valueInfo))
      skippedAll &= !processScalarInfo(scalarInfo, value, transparentType, forceEnableConv);
    else if (std::shared_ptr<StructInfo> structInfo = dynamic_ptr_cast<StructInfo>(valueInfo))
      processStructInfo(structInfo, value, transparentType, queue);
    else
      llvm_unreachable("unknown valueInfo kind");
  }

  if (!skippedAll) {
    std::shared_ptr<DtaValueInfo> dtaValueInfo = createDtaValueInfo(value);
    taffoInfo.setValueInfo(*value, newValueInfo);
    LLVM_DEBUG(log().log("new valueInfo: ").logln(*newValueInfo, Logger::Cyan));
    if (std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast<ScalarInfo>(newValueInfo))
      dtaValueInfo->initialType = scalarInfo->numericType;
  }
  return !skippedAll;
}

void DataTypeAllocationPass::sortQueue(std::vector<Value*>& vals, SmallPtrSetImpl<Value*>& valset) {
  // Topological sort by means of a reversed DFS.
  enum VState {
    Visited,
    Visiting
  };
  DenseMap<Value*, VState> vstates;
  std::vector<Value*> revQueue;
  std::vector<Value*> stack;
  revQueue.reserve(vals.size());
  stack.reserve(vals.size());

  for (Value* v : vals) {
    if (vstates.count(v))
      continue;

    stack.push_back(v);
    while (!stack.empty()) {
      Value* c = stack.back();
      auto cstate = vstates.find(c);
      if (cstate == vstates.end()) {
        vstates[c] = Visiting;
        for (Value* u : c->users()) {
          if (!isa<Instruction>(u) && !isa<GlobalObject>(u))
            continue;

          if (isConversionDisabled(u)) {
            LLVM_DEBUG(log() << "[WARNING] Skipping " << *u << " because its conversion is disabled!\n");
            continue;
          }

          stack.push_back(u);
        }
      }
      else if (cstate->second == Visiting) {
        revQueue.push_back(c);
        stack.pop_back();
        vstates[c] = Visited;
      }
      else {
        assert(cstate->second == Visited);
        stack.pop_back();
      }
    }
  }

  vals.clear();
  valset.clear();
  for (auto i = revQueue.rbegin(); i != revQueue.rend(); ++i) {
    vals.push_back(*i);
    valset.insert(*i);
    if (Argument* Arg = dyn_cast<Argument>(*i)) {
      LLVM_DEBUG(log() << "Restoring consistency of argument " << *Arg << " of function "
                       << Arg->getParent()->getNameOrAsOperand() << "\n");
      restoreTypesAcrossFunctionCall(Arg);
    }
  }
}

void DataTypeAllocationPass::mergeFixFormat(const std::vector<Value*>& vals, const SmallPtrSetImpl<Value*>& valset) {
  if (DisableTypeMerging)
    return;

  assert(vals.size() == valset.size() && "They must contain the same elements.");
  bool merged = false;
  for (Value* v : vals) {
    for (Value* u : v->users()) {
      if (valset.count(u)) {
        if (mergeFixFormat(v, u)) {
          restoreTypesAcrossFunctionCall(v);
          restoreTypesAcrossFunctionCall(u);

          merged = true;
        }
      }
    }
  }
  if (IterativeMerging && merged)
    mergeFixFormat(vals, valset);
}

bool DataTypeAllocationPass::mergeFixFormat(Value* v, Value* u) {
  std::shared_ptr<DtaValueInfo> valueTunerInfo = getDtaValueInfo(v);
  std::shared_ptr<DtaValueInfo> userTunerInfo = getOrCreateDtaValueInfo(u);
  if (!taffoInfo.hasValueInfo(*v) || !taffoInfo.hasValueInfo(*u)) {
    LLVM_DEBUG(log() << "not attempting merge of " << *v << ", " << *u << " because at least one is a struct\n");
    return false;
  }
  std::shared_ptr<ScalarInfo> valueInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*v));
  std::shared_ptr<ScalarInfo> userInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*u));
  if (!valueInfo->numericType || !valueTunerInfo->initialType || !userInfo->numericType
      || !userTunerInfo->initialType) {
    LLVM_DEBUG(log() << "not attempting merge of " << *v << ", " << *u
                     << " because at least one does not change to a fixed point type\n");
    return false;
  }
  if (v->getType()->isPointerTy() || u->getType()->isPointerTy()) {
    LLVM_DEBUG(log() << "not attempting merge of " << *v << ", " << *u << " because at least one is a pointer\n");
    return false;
  }

  std::shared_ptr<NumericTypeInfo> valueNumericType = dynamic_ptr_cast<NumericTypeInfo>(valueTunerInfo->initialType);
  std::shared_ptr<NumericTypeInfo> userNumericType = dynamic_ptr_cast<NumericTypeInfo>(userTunerInfo->initialType);

  if (strategy->isMergeable(valueNumericType, userNumericType)) {

    std::shared_ptr<NumericTypeInfo> fp = strategy->merge(valueNumericType, userNumericType);
    if (!fp) {
      LLVM_DEBUG(log() << "not attempting merge of " << *v << ", " << *u << " because resulting type is invalid\n");
      return false;
    }
    LLVM_DEBUG(log() << "Merged type : \n"
                     << "\t" << *v << " fix type " << valueNumericType->toString() << "\n"
                     << "\t" << *u << " fix type " << userNumericType->toString() << "\n"
                     << "Final format " << fp->toString() << "\n";);

    valueInfo->numericType = fp->clone();
    userInfo->numericType = fp->clone();
    return true;
  }
  else {
    FixCast++;
  }

  return false;
}

void DataTypeAllocationPass::mergeBufferIDSets() {
  LLVM_DEBUG(log() << "\n"
                   << __PRETTY_FUNCTION__ << " BEGIN\n\n");
  BufferIDTypeMap InMap, OutMap;
  if (!BufferIDImport.empty()) {
    LLVM_DEBUG(log() << "Importing Buffer ID sets from " << BufferIDImport << "\n\n");
    ReadBufferIDFile(BufferIDImport, InMap);
  }

  for (auto& Set : bufferIDSets) {
    LLVM_DEBUG(log() << "Merging Buffer ID set " << Set.first << "\n");

    std::shared_ptr<NumericTypeInfo> DestType;
    if (InMap.find(Set.first) != InMap.end()) {
      LLVM_DEBUG(log() << "Set has type specified in file\n");
      DestType = InMap.at(Set.first)->clone();
    }
    else {
      for (auto* V : Set.second) {
        std::shared_ptr<DtaValueInfo> dtaValueInfo = getDtaValueInfo(V);
        if (!taffoInfo.hasValueInfo(*V)) {
          LLVM_DEBUG(log() << "Metadata is null or struct, not handled, bailing out! Value='" << *V << "'\n");
          goto nextSet;
        }
        std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*V));
        std::shared_ptr<NumericTypeInfo> T = scalarInfo->numericType;
        if (T) {
          LLVM_DEBUG(log() << "Type=" << T->toString() << " Value='" << *V << "'\n");
        }
        else {
          LLVM_DEBUG(log() << "Type is null, not handled, bailing out! Value='" << *V << "'\n");
          continue;
        }

        if (!DestType)
          DestType = T->clone();
        else
          DestType = strategy->merge(DestType, T);
      }
    }
    LLVM_DEBUG(log() << "Computed merged type: " << DestType->toString() << "\n");

    for (auto* V : Set.second) {
      std::shared_ptr<DtaValueInfo> dtaValueInfo = getDtaValueInfo(V);
      std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*V));
      scalarInfo->numericType = DestType->clone();
      restoreTypesAcrossFunctionCall(V);
    }
    OutMap[Set.first] = DestType->clone();

nextSet:
    LLVM_DEBUG(log() << "Merging Buffer ID set " << Set.first << " DONE\n\n");
  }

  if (!BufferIDExport.empty()) {
    LLVM_DEBUG(log() << "Exporting Buffer ID sets to " << BufferIDExport << "\n\n");
    WriteBufferIDFile(BufferIDExport, OutMap);
  }

  LLVM_DEBUG(log() << __PRETTY_FUNCTION__ << " END\n\n");
}

void DataTypeAllocationPass::restoreTypesAcrossFunctionCall(Value* v) {
  LLVM_DEBUG(log() << "restoreTypesAcrossFunctionCall(" << *v << ")\n");
  if (!hasDtaInfo(v)) {
    LLVM_DEBUG(log() << " --> skipping restoring types because value is not converted\n");
    return;
  }

  std::shared_ptr<ValueInfo> finalValueInfo = taffoInfo.getValueInfo(*v);

  if (auto* arg = dyn_cast<Argument>(v)) {
    LLVM_DEBUG(log() << "Is a function argument, propagating to calls\n");
    setTypesOnCallArgumentFromFunctionArgument(arg, finalValueInfo);
  }
  else {
    LLVM_DEBUG(log() << "Not a function argument, propagating to function arguments\n");
    setTypesOnFunctionArgumentFromCallArgument(v, finalValueInfo);
  }

  LLVM_DEBUG(log() << "restoreTypesAcrossFunctionCall ended\n");
}

void DataTypeAllocationPass::setTypesOnFunctionArgumentFromCallArgument(Value* v, std::shared_ptr<ValueInfo> finalMd) {
  for (Use& use : v->uses()) {
    User* user = use.getUser();
    auto* call = dyn_cast<CallBase>(user);
    if (call == nullptr)
      continue;
    LLVM_DEBUG(log() << "restoreTypesAcrossFunctionCall: processing user " << *(user) << ")\n");

    auto* fun = dyn_cast<Function>(call->getCalledFunction());
    if (fun == nullptr) {
      LLVM_DEBUG(log() << " --> skipping restoring types from call site " << *user
                       << " because function reference cannot be resolved\n");
      continue;
    }
    if (fun->isVarArg()) {
      LLVM_DEBUG(log() << " --> skipping restoring types from call site " << *user << " because function is vararg\n");
      continue;
    }

    assert(fun->arg_size() > use.getOperandNo() && "invalid call to function; operandNo > numOperands");
    Argument* arg = fun->arg_begin() + use.getOperandNo();
    if (hasDtaInfo(arg)) {
      taffoInfo.setValueInfo(*arg, finalMd->clone());
      setTypesOnCallArgumentFromFunctionArgument(arg, finalMd);
      LLVM_DEBUG(log() << " --> set new metadata, now checking uses of the argument... (hope there's no recursion!)\n");
      setTypesOnFunctionArgumentFromCallArgument(arg, finalMd);
    }
    else {
      LLVM_DEBUG(log() << "Not looking good, formal arg #" << use.getOperandNo() << " (" << *arg
                       << ") has no valueInfo, but actual argument does...\n");
    }
  }
}

void DataTypeAllocationPass::setTypesOnCallArgumentFromFunctionArgument(Argument* arg,
                                                                        std::shared_ptr<ValueInfo> finalMd) {
  Function* fun = arg->getParent();
  int n = arg->getArgNo();
  LLVM_DEBUG(log() << " --> setting types to " << finalMd->toString() << " on call arguments from function "
                   << fun->getName() << " argument " << n << "\n");
  for (auto it = fun->user_begin(); it != fun->user_end(); it++) {
    if (isa<CallInst>(*it) || isa<InvokeInst>(*it)) {
      Value* callarg = it->getOperand(n);
      LLVM_DEBUG(log() << " --> target " << *callarg << ", CallBase " << **it << "\n");

      if (!hasDtaInfo(callarg)) {
        if (!isa<Argument>(callarg)) {
          LLVM_DEBUG(log() << " --> actual argument doesn't get converted; skipping\n");
          continue;
        }
        else {
          LLVM_DEBUG(
            log() << " --> actual argument IS AN ARGUMENT ITSELF! not skipping even if it doesn't get converted\n");
        }
      }
      taffoInfo.setValueInfo(*callarg, finalMd->clone());
      if (auto* Arg = dyn_cast<Argument>(callarg)) {
        LLVM_DEBUG(log() << " --> actual argument IS AN ARGUMENT ITSELF, recursing\n");
        setTypesOnCallArgumentFromFunctionArgument(Arg, finalMd);
      }
    }
  }
}

std::vector<Function*> DataTypeAllocationPass::collapseFunction(Module& m) {
  std::vector<Function*> toDel;
  for (Function& f : m.functions()) {
    if (std::ranges::find(toDel, &f) != toDel.end())
      continue;
    LLVM_DEBUG(log() << "Analyzing original function " << f.getName() << "\n");

    SmallPtrSet<Function*, 2> taffoFunctions;
    taffoInfo.getTaffoCloneFunctions(f, taffoFunctions);
    for (Function* cloneF : taffoFunctions) {
      LLVM_DEBUG(log() << "\t Clone: " << *cloneF << "\n");
      if (cloneF->user_empty()) {
        LLVM_DEBUG(log() << "\t Ignoring " << cloneF->getName() << " because it's not used anywhere\n");
      }
      else if (Function* eqFun = findEqFunction(cloneF, &f)) {
        LLVM_DEBUG(log() << "\t Replace function " << cloneF->getName() << " with " << eqFun->getName() << "\n";);
        cloneF->replaceAllUsesWith(eqFun);
        toDel.push_back(cloneF);
      }
    }
  }
  return toDel;
}

bool compareTypesOfMDInfo(const std::shared_ptr<ValueInfo>& mdi1, const std::shared_ptr<ValueInfo>& mdi2) {
  if (mdi1->getKind() != mdi2->getKind())
    return false;

  if (isa<ScalarInfo>(mdi1.get())) {
    std::shared_ptr<ScalarInfo> ii1 = static_ptr_cast<ScalarInfo>(mdi1);
    std::shared_ptr<ScalarInfo> ii2 = static_ptr_cast<ScalarInfo>(mdi2);
    if (ii1->numericType && ii2->numericType)
      return *ii1->numericType == *ii2->numericType;
    else
      return false;
  }
  else if (isa<StructInfo>(mdi1.get())) {
    std::shared_ptr<StructInfo> si1 = static_ptr_cast<StructInfo>(mdi1);
    std::shared_ptr<StructInfo> si2 = static_ptr_cast<StructInfo>(mdi2);
    if (si1->getNumFields() == si2->getNumFields()) {
      int numFields = si1->getNumFields();
      for (int i = 0; i < numFields; i++) {
        std::shared_ptr<ValueInfo> p1 = si1->getField(i);
        std::shared_ptr<ValueInfo> p2 = si1->getField(i);
        if ((p1.get() == nullptr) != (p2.get() == nullptr))
          return false;
        if (p1.get() != nullptr) {
          if (!compareTypesOfMDInfo(p1, p2))
            return false;
        }
      }
      return true;
    }
    else
      return false;
  }
  else {
    return false;
  }
}

Function* DataTypeAllocationPass::findEqFunction(Function* fun, Function* origin) {
  std::vector<std::pair<int, std::shared_ptr<ValueInfo>>> fixSign;

  LLVM_DEBUG(log() << "\t\t Search eq function for " << fun->getName() << " in " << origin->getName() << " pool\n";);

  if (getFullyUnwrappedType(fun)->isFloatingPointTy() && hasDtaInfo(*fun->user_begin())) {
    if (taffoInfo.hasValueInfo(**fun->user_begin())) {
      std::shared_ptr<ValueInfo> retval = taffoInfo.getValueInfo(**fun->user_begin());
      fixSign.push_back(std::pair(-1, retval)); // ret value in signature
      LLVM_DEBUG(log() << "\t\t Return type : " << *retval << "\n";);
    }
  }

  int i = 0;
  for (Argument& arg : fun->args()) {
    if (hasDtaInfo(&arg) && taffoInfo.hasValueInfo(arg)) {
      fixSign.push_back(std::pair(i, taffoInfo.getValueInfo(arg)));
      LLVM_DEBUG(log() << "\t\t Arg " << i << " type : " << *taffoInfo.getValueInfo(arg) << "\n";);
    }
    i++;
  }

  for (FunInfo fi : functionPool[origin]) {
    if (fi.fixArgs.size() == fixSign.size()) {
      auto fcheck = fi.fixArgs.begin();
      auto fthis = fixSign.begin();
      for (; fthis != fixSign.end(); fcheck++, fthis++) {
        if (fcheck->first != fthis->first)
          break;
        if (fcheck->second != fthis->second)
          if (!compareTypesOfMDInfo(fcheck->second, fthis->second))
            break;
      }
      if (fthis == fixSign.end())
        return fi.newFun;
    }
  }

  FunInfo funInfo;
  funInfo.newFun = fun;
  funInfo.fixArgs = fixSign;
  functionPool[origin].push_back(funInfo);
  LLVM_DEBUG(log() << "\t Function " << fun->getName() << " used\n";);
  return nullptr;
}

void DataTypeAllocationPass::attachFPMetaData(std::vector<Value*>& vals) {
  for (Value* v : vals) {
    // assert(info[v] && "Every value should have info");
    assert(taffoInfo.hasValueInfo(*v) && "every value should have metadata");

    // TODO remove this function
  }
}

void DataTypeAllocationPass::attachFunctionMetaData(Module& m) {
  for (Function& f : m.functions()) {
    if (f.isIntrinsic())
      continue;
    // for (Argument& arg : f.args())
    //   if (taffoInfo.hasValueInfo(arg) && hasTunerInfo(&arg))
    //     taffoInfo.setValueInfo(arg, getDtaValueInfo(&arg)->metadata);
    //  TODO remove this function
  }
}

#ifdef TAFFO_BUILD_ILP_DTA
void DataTypeAllocationPass::buildModelAndOptimze(Module& m,
                                                  const vector<Value*>& vals,
                                                  const SmallPtrSetImpl<Value*>& valset) {
  assert(vals.size() == valset.size() && "They must contain the same elements.");

  Optimizer optimizer(m, this, new MetricPerf(), CostModelFilename, CPUCosts::CostType::Performance);
  // Optimizer optimizer(m, this, new MetricPerf(),"", CPUCosts::CostType::Size);
  optimizer.initialize();

  LLVM_DEBUG(
  log() << "\n============ GLOBALS ============\n");

  for (GlobalObject& globObj : m.globals()) {
    LLVM_DEBUG(
    globObj.print(log()););
    LLVM_DEBUG(
    log() << "     -having-     ");
    if (!hasTunerInfo(&globObj)) {
      LLVM_DEBUG(
      log() << "No info available, skipping.");
    }
    else {
      LLVM_DEBUG(
      log() << getTunerInfo(&globObj)->metadata->toString() << "\n");

      optimizer.handleGlobal(&globObj, getTunerInfo(&globObj));
    }
    LLVM_DEBUG(
    log() << "\n\n";);
  }

  // FIXME: this is an hack to prevent multiple visit of the same function if it will be called somewhere from the
  // program
  for (Function& f : m.functions()) {
    // Skip compiler provided functions
    if (f.isIntrinsic() || f.isDeclaration())
      continue;

    if (!f.isIntrinsic() && !f.empty() && f.getName().equals("main")) {
      LLVM_DEBUG(
      log() << "========== GLOBAL ENTRY POINT main ==========";);

      optimizer.handleCallFromRoot(&f);
      break;
    }
  }

  // Looking for remaining functions
  for (Function& f : m.functions()) {
    // Skip compiler provided functions
    if (f.isIntrinsic()) {
      LLVM_DEBUG(
      log() << "Skipping intrinsic function " << f.getName() << "\n";);
      continue;
    }

    // Skip empty functions
    if (f.empty()) {
      LLVM_DEBUG(
      log() << "Skipping empty function " << f.getName() << "\n";);
      continue;
    }

    optimizer.handleCallFromRoot(&f);
  }

  bool result = optimizer.finish();
  assert(result && "Optimizer did not find a solution!");

  for (Value* v : vals) {
    LLVM_DEBUG(
    log() << "Processing " << *v << "...\n");

    if (!valset.count(v)) {
      LLVM_DEBUG(
      log() << "Not in the conversion queue! Skipping!\n\n";);
      continue;
    }

    std::shared_ptr<TunerInfo> viu = getTunerInfo(v);

    // Read from the model, search for the data type associated with that value and convert it!
    auto fp = optimizer.getAssociatedMetadata(v);
    if (!fp) {
      LLVM_DEBUG(
      log() << "Invalid datatype returned!\n";);
      continue;
    }
    LLVM_DEBUG(
    log() << "Datatype: " << fp->toString() << "\n");

    // Write the datatype
    bool result = overwriteType(viu->metadata, fp);
    if (result) {
      // Some datatype has changed, restore in function call
      LLVM_DEBUG(
      log() << "Restoring call type because of mergeDataTypes()...\n";);
      restoreTypesAcrossFunctionCall(v);
    }

    LLVM_DEBUG(
    log() << "done with [" << *v << "]\n\n");
    /*auto *iiv = dyn_cast<InputInfo>(viu->metadata.get());

    iiv->IType.reset(fp->clone());*/
  }

  optimizer.printStatInfos();
}

bool DataTypeAllocationPass::overwriteType(shared_ptr<ValueInfo> old, shared_ptr<ValueInfo> model) {
  if (!old || !model)
    return false;

  if (old->getKind() == ValueInfo::K_Scalar) {
    assert(model->getKind() == ValueInfo::K_Scalar && "Mismatching metadata infos!!!");

    std::shared_ptr<ScalarInfo> old1 = dynamic_ptr_cast<ScalarInfo>(old);
    std::shared_ptr<ScalarInfo> model1 = dynamic_ptr_cast<ScalarInfo>(model);

    if (!old1->numericType)
      return false;
    LLVM_DEBUG(
    log() << "model1: " << model1->numericType->toString() << "\n";);
    LLVM_DEBUG(
    log() << "old1: " << old1->numericType->toString() << "\n";);
    if (*old1->numericType == *model1->numericType)
      return false;

    old1->numericType = model1->numericType->clone();
    return true;
  }
  else if (old->getKind() == ValueInfo::K_Struct) {
    std::shared_ptr<StructInfo> old1 = dynamic_ptr_cast<StructInfo>(old);
    std::shared_ptr<StructInfo> model1 = dynamic_ptr_cast<StructInfo>(model);

    bool changed = false;
    for (unsigned i = 0; i < old1->getNumFields(); i++)
      changed |= overwriteType(old1->getField(i), model1->getField(i));
    return changed;
  }

  llvm_unreachable("unknown data type");
}
#endif // TAFFO_BUILD_ILP_DTA
