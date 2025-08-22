#include "BufferIDFiles.h"
#include "DTAConfig.hpp"
#include "DataTypeAllocationPass.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/NumericInfo.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueInfo.hpp"
#include "Utils/PtrCasts.hpp"

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

#define DEBUG_TYPE "taffo-dta"

STATISTIC(fixCast, "Number of fixed point format cast");

/* the strategy map that associates the global variable values (set through the command line arguments)
 * to the strategy object that held the methods for the chosen strategy. When adding a new strategy
 * you should add a new entry in the map */
std::map<DtaStrategyType, std::function<AllocationStrategy*()>> strategyMap = {
  {fixedPointOnly,    []() -> FixedPointOnlyStrategy* { return new FixedPointOnlyStrategy(); }        },
  {floatingPointOnly, []() -> FloatingPointOnlyStrategy* { return new FloatingPointOnlyStrategy(); }  },
  {fixedFloating,     []() -> FixedFloatingPointStrategy* { return new FixedFloatingPointStrategy(); }}
};

PreservedAnalyses DataTypeAllocationPass::run(Module& m, ModuleAnalysisManager&) {
  LLVM_DEBUG(log().logln("[DataTypeAllocationPass]", Logger::Magenta));
  taffoInfo.initializeFromFile(VRA_TAFFO_INFO, m);

  // method that allocate the strategy object, whose methods will be used to apply the strategy
  setStrategy(strategyMap[DtaStrategy]());

  std::vector<Value*> values;
  SmallPtrSet<Value*, 8U> valueSet;
  allocateTypes(m, values, valueSet);
  mergeTypes(values, valueSet);

  mergeBufferIDSets();

  std::vector<Function*> toDelete = collapseFunction(m);

  for (Function* f : toDelete)
    taffoInfo.eraseValue(f);

  taffoInfo.dumpToFile(DTA_TAFFO_INFO, m);
  LLVM_DEBUG(log().logln("[End of DataTypeAllocationPass]", Logger::Magenta));
  return PreservedAnalyses::all();
}

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

void DataTypeAllocationPass::allocateTypes(Module& m, std::vector<Value*>& values, SmallPtrSetImpl<Value*>& valueSet) {
  LLVM_DEBUG(log().logln("[DataTypeAllocation of globals]", Logger::Blue));
  allocateGlobalTypes(m, values);
  allocateLocalTypes(m, values);

  LLVM_DEBUG(log().logln("[Sorting queue]", Logger::Blue));
  sortQueue(values, valueSet);
}

void DataTypeAllocationPass::allocateGlobalTypes(Module& m, std::vector<Value*>& values) {
  for (GlobalObject& globObj : m.globals())
    allocateValueType(globObj, values);
  LLVM_DEBUG(log() << "\n");
}

void DataTypeAllocationPass::allocateLocalTypes(Module& m, std::vector<Value*>& values) {
  for (Function& f : m.functions()) {
    if (f.isIntrinsic())
      continue;
    if (!taffoInfo.isCloneFunction(f) && !taffoInfo.isStartingPoint(f)) {
      LLVM_DEBUG(log() << "Skip function " << f.getName() << " as it is not a cloned function\n";);
      continue;
    }
    LLVM_DEBUG(log().log("[DataTypeAllocation of function] ", Logger::Blue).logValueln(&f));
    for (Argument& arg : f.args())
      allocateValueType(arg, values);
    for (Instruction& inst : instructions(f))
      allocateValueType(inst, values);
  }
}

void DataTypeAllocationPass::allocateValueType(Value& value, std::vector<Value*>& values) {
  if (allocateType(&value)) {
    values.push_back(&value);
    retrieveBufferID(&value);
  }
}

bool DataTypeAllocationPass::allocateType(Value* value) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger.log("[Value] ", Logger::Bold).logValueln(value);
    indenter.increaseIndent(););

  if (!taffoInfo.hasValueInfo(*value)) {
    LLVM_DEBUG(log() << "no valueInfo: skipping\n");
    return false;
  }
  std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*value);
  LLVM_DEBUG(logger << "valueInfo: " << *valueInfo << "\n");

  std::shared_ptr<ValueInfo> newValueInfo = valueInfo->clone();

  if (value->getType()->isVoidTy()) {
    LLVM_DEBUG(log() << "value has void type: skipping\n");
    return true;
  }

  // HACK to set the enabled status on phis which compensates for a bug in vra.
  // Affects axbench/sobel.
  bool forceEnableConv = false;
  if (isa<PHINode>(value) && !isConversionDisabled(value) && isa<ScalarInfo>(newValueInfo.get()))
    forceEnableConv = true;

  bool skippedAll = true;
  TransparentType* transparentType = taffoInfo.getOrCreateTransparentType(*value);
  SmallVector<std::pair<std::shared_ptr<ValueInfo>, TransparentType*>, 8> queue(
    {std::make_pair(newValueInfo, transparentType)});

  while (!queue.empty()) {
    const auto& [valueInfo, transparentType] = queue.pop_back_val();

    if (std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast<ScalarInfo>(valueInfo))
      skippedAll &= !allocateScalarType(scalarInfo, value, transparentType, forceEnableConv);
    else if (std::shared_ptr<StructInfo> structInfo = dynamic_ptr_cast<StructInfo>(valueInfo))
      allocateStructType(structInfo, value, transparentType, queue);
    else
      llvm_unreachable("Unknown valueInfo kind");
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

bool DataTypeAllocationPass::allocateScalarType(std::shared_ptr<ScalarInfo>& scalarInfo,
                                                Value* value,
                                                const TransparentType* transparentType,
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

  return strategy->apply(scalarInfo, value);
}

void DataTypeAllocationPass::allocateStructType(
  std::shared_ptr<StructInfo>& structInfo,
  const Value* value,
  const TransparentType* transparentType,
  SmallVector<std::pair<std::shared_ptr<ValueInfo>, TransparentType*>, 8>& queue) {
  if (!transparentType->isStructTT()) {
    LLVM_DEBUG(log() << "[ERROR] found non conforming structInfo " << structInfo->toString() << " on value " << *value
                     << "\n");
    LLVM_DEBUG(log() << "contained type " << *transparentType << " is not a struct type\n");
    LLVM_DEBUG(log() << "The top-level MDInfo was " << structInfo->toString() << "\n");
    llvm_unreachable("Non-conforming StructInfo.");
  }
  for (unsigned i = 0; i < structInfo->getNumFields(); i++)
    if (const std::shared_ptr<ValueInfo>& field = structInfo->getField(i))
      queue.push_back(std::make_pair(field, cast<TransparentStructType>(transparentType)->getFieldType(i)));
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

void DataTypeAllocationPass::sortQueue(std::vector<Value*>& vals, SmallPtrSetImpl<Value*>& valset) {
  // Topological sort by means of a reversed DFS.
  enum VState {
    Visited,
    Visiting
  };
  DenseMap<Value*, VState> vStates;
  std::vector<Value*> revQueue;
  std::vector<Value*> stack;
  revQueue.reserve(vals.size());
  stack.reserve(vals.size());

  for (Value* value : vals) {
    if (vStates.count(value))
      continue;

    stack.push_back(value);
    while (!stack.empty()) {
      Value* c = stack.back();
      auto cState = vStates.find(c);
      if (cState == vStates.end()) {
        vStates[c] = Visiting;
        for (Value* user : c->users()) {
          if (!isa<Instruction>(user) && !isa<GlobalObject>(user))
            continue;
          if (isConversionDisabled(user)) {
            LLVM_DEBUG(log() << "conversion is disabled: skipping " << *user << "\n");
            continue;
          }
          stack.push_back(user);
        }
      }
      else if (cState->second == Visiting) {
        revQueue.push_back(c);
        stack.pop_back();
        vStates[c] = Visited;
      }
      else {
        assert(cState->second == Visited);
        stack.pop_back();
      }
    }
  }

  vals.clear();
  valset.clear();
  for (auto i = revQueue.rbegin(); i != revQueue.rend(); ++i) {
    vals.push_back(*i);
    valset.insert(*i);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Value* value : vals)
      changed |= propagateTypeAcrossCalls(value);
  }
}

void DataTypeAllocationPass::mergeTypes(const std::vector<Value*>& vals, const SmallPtrSetImpl<Value*>& valset) {
  if (disableTypeMerging)
    return;

  assert(vals.size() == valset.size() && "They must contain the same elements");
  bool merged = false;
  for (Value* value : vals) {
    for (Value* user : value->users()) {
      if (valset.count(user)) {
        if (mergeTypes(value, user)) {
          propagateTypeAcrossCalls(value);
          propagateTypeAcrossCalls(user);
          merged = true;
        }
      }
    }
  }
  if (iterativeMerging && merged)
    mergeTypes(vals, valset);
}

bool DataTypeAllocationPass::mergeTypes(Value* value1, Value* value2) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << Logger::Bold << "[" << __FUNCTION__ << "]\n" << Logger::Reset;
    indenter.increaseIndent();
    logger.log("value1: ").logValueln(value1);
    logger.log("value2: ").logValueln(value2););

  if (!taffoInfo.hasValueInfo(*value1) || !taffoInfo.hasValueInfo(*value2)) {
    LLVM_DEBUG(logger << "at least one has no valueInfo: skipping\n");
    return false;
  }

  auto valueInfo1 = taffoInfo.getValueInfo(*value1);
  auto valueInfo2 = taffoInfo.getValueInfo(*value2);
  auto* type1 = taffoInfo.getOrCreateTransparentType(*value1);
  auto* type2 = taffoInfo.getOrCreateTransparentType(*value2);

  return mergeTypes(valueInfo1, type1, valueInfo2, type2);
}
bool DataTypeAllocationPass::mergeTypes(std::shared_ptr<ValueInfo> valueInfo1,
                                        TransparentType* type1,
                                        std::shared_ptr<ValueInfo> valueInfo2,
                                        TransparentType* type2) {
  Logger& logger = log();

  // Scalar <-> Scalar
  if (auto scalarInfo1 = std::dynamic_ptr_cast<ScalarInfo>(valueInfo1)) {
    if (auto scalarInfo2 = std::dynamic_ptr_cast<ScalarInfo>(valueInfo2))
      return mergeTypes(scalarInfo1, scalarInfo2);
    // Struct-vs-Scalar is handled elsewhere (via GEP propagation). Skip here.
    LLVM_DEBUG(logger << "kinds mismatch (scalar vs non-scalar): skipping in mergeValueInfos\n");
    return false;
  }

  // Struct <-> Struct (recurse)
  if (auto structInfo1 = std::dynamic_ptr_cast<StructInfo>(valueInfo1)) {
    auto structInfo2 = std::dynamic_ptr_cast<StructInfo>(valueInfo2);
    if (!structInfo2) {
      LLVM_DEBUG(logger << "kinds mismatch (struct vs non-struct): skipping in mergeValueInfos\n");
      return false;
    }
    if (!type1 || !type2 || !type1->isStructTT() || !type2->isStructTT()) {
      LLVM_DEBUG(logger << "transparent types not struct: skipping\n");
      return false;
    }
    auto structType1 = llvm::cast<TransparentStructType>(type1);
    auto structType2 = llvm::cast<TransparentStructType>(type2);

    unsigned n = std::min(structInfo1->getNumFields(), structInfo2->getNumFields());
    bool changed = false;
    for (unsigned i = 0; i < n; ++i) {
      auto fieldInfo1 = structInfo1->getField(i);
      auto fieldInfo2 = structInfo2->getField(i);
      if (!fieldInfo1 || !fieldInfo2)
        continue; // nothing to merge for this field

      auto fieldType1 = structType1->getFieldType(i);
      auto fieldType2 = structType2->getFieldType(i);
      changed |= mergeTypes(fieldInfo1, fieldType1, fieldInfo2, fieldType2);
    }
    return changed;
  }
  llvm_unreachable("Unknown valueInfo kind");
}

bool DataTypeAllocationPass::mergeTypes(std::shared_ptr<ScalarInfo> scalarInfo1,
                                        std::shared_ptr<ScalarInfo> scalarInfo2) {
  Logger& logger = log();
  if (!scalarInfo1->numericType && !scalarInfo2->numericType) {
    LLVM_DEBUG(logger << "neither has a numeric type: skipping\n");
    return false;
  }

  if (!scalarInfo1->numericType) {
    scalarInfo1->numericType = scalarInfo2->numericType->clone();
    LLVM_DEBUG(
      logger << "scalarInfo1 has no numeric type: copying it from scalarInfo2\n";
      logger.log("new type: ").logln(*scalarInfo1->numericType, Logger::Green););
    return true;
  }
  if (!scalarInfo2->numericType) {
    scalarInfo2->numericType = scalarInfo1->numericType->clone();
    LLVM_DEBUG(
      logger << "scalarInfo2 has no numeric type: copying it from scalarInfo1\n";
      logger.log("new type: ").logln(*scalarInfo2->numericType, Logger::Green););
    return true;
  }

  LLVM_DEBUG(
    logger.log("type1: ").logln(*scalarInfo1->numericType, Logger::Cyan);
    logger.log("type2: ").logln(*scalarInfo2->numericType, Logger::Cyan););
  if (*scalarInfo1->numericType == *scalarInfo2->numericType) {
    LLVM_DEBUG(logger << "same type already\n");
    return false;
  }

  if (strategy->isMergeable(scalarInfo1->numericType, scalarInfo2->numericType)) {
    std::shared_ptr<NumericTypeInfo> mergedType = strategy->merge(scalarInfo1->numericType, scalarInfo2->numericType);
    if (!mergedType) {
      LLVM_DEBUG(logger << "merge failed: skipping\n");
      fixCast++;
      return false;
    }
    LLVM_DEBUG(logger.log("merged type: ").logln(*mergedType, Logger::Green););
    scalarInfo1->numericType = mergedType->clone();
    scalarInfo2->numericType = mergedType->clone();
    return true;
  }
  LLVM_DEBUG(logger << "types are not mergeable: skipping\n");
  fixCast++;
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

  for (auto& set : bufferIDSets) {
    LLVM_DEBUG(log() << "Merging Buffer ID set " << set.first << "\n");

    std::shared_ptr<NumericTypeInfo> DestType;
    if (InMap.find(set.first) != InMap.end()) {
      LLVM_DEBUG(log() << "Set has type specified in file\n");
      DestType = InMap.at(set.first)->clone();
    }
    else {
      for (auto* value : set.second) {
        std::shared_ptr<DtaValueInfo> dtaValueInfo = getDtaValueInfo(value);
        if (!taffoInfo.hasValueInfo(*value)) {
          LLVM_DEBUG(log() << "Metadata is null or struct, not handled, bailing out! Value='" << *value << "'\n");
          goto nextSet;
        }
        std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*value));
        std::shared_ptr<NumericTypeInfo> T = scalarInfo->numericType;
        if (T) {
          LLVM_DEBUG(log() << "Type=" << T->toString() << " Value='" << *value << "'\n");
        }
        else {
          LLVM_DEBUG(log() << "Type is null, not handled, bailing out! Value='" << *value << "'\n");
          continue;
        }

        if (!DestType)
          DestType = T->clone();
        else
          DestType = strategy->merge(DestType, T);
      }
    }
    LLVM_DEBUG(log() << "Computed merged type: " << DestType->toString() << "\n");

    for (auto* V : set.second) {
      std::shared_ptr<DtaValueInfo> dtaValueInfo = getDtaValueInfo(V);
      std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*V));
      scalarInfo->numericType = DestType->clone();
      propagateTypeAcrossCalls(V);
    }
    OutMap[set.first] = DestType->clone();

nextSet:
    LLVM_DEBUG(log() << "Merging Buffer ID set " << set.first << " DONE\n\n");
  }

  if (!BufferIDExport.empty()) {
    LLVM_DEBUG(log() << "Exporting Buffer ID sets to " << BufferIDExport << "\n\n");
    WriteBufferIDFile(BufferIDExport, OutMap);
  }

  LLVM_DEBUG(log() << __PRETTY_FUNCTION__ << " END\n\n");
}

bool DataTypeAllocationPass::propagateTypeAcrossCalls(Value* value) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << Logger::Bold << "[" << __FUNCTION__ << "] " << Logger::Reset;
    logger.logValueln(value);
    indenter.increaseIndent(););

  if (!taffoInfo.hasValueInfo(*value)) {
    LLVM_DEBUG(logger << "value has no valueInfo: skipping\n");
    return false;
  }
  std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*value);

  bool changed = false;
  if (auto* arg = dyn_cast<Argument>(value)) {
    LLVM_DEBUG(logger << "value is an argument: propagating backward to call arguments\n");
    changed |= propagateArgType(arg, valueInfo);
  }
  else if (auto* call = dyn_cast<CallBase>(value)) {
    LLVM_DEBUG(
      logger
      << "value is a call: propagating forward to function arguments, function return values and function itself\n");
    changed |= propagateCallType(call);
    for (Use& use : call->args()) {
      Value* callArg = use.get();
      changed |= propagateTypeAcrossCalls(callArg);
    }
  }
  else if (auto* gep = dyn_cast<GetElementPtrInst>(value)) {
    LLVM_DEBUG(logger << "value is a gep: propagating backward to pointer operand\n");
    changed |= propagateGepType(gep);
  }

  for (Use& use : value->uses()) {
    User* user = use.getUser();
    if (auto* call = dyn_cast<CallBase>(user)) {
      if (!is_contained(call->args(), value))
        continue; // Not a call argument
      LLVM_DEBUG(
        logger.log("value used as call argument in call: ").logValueln(call);
        logger << "propagating forward to function arguments\n";);
      auto* fun = dyn_cast<Function>(call->getCalledFunction());
      if (!fun) {
        LLVM_DEBUG(logger << "function reference cannot be resolved: skipping\n");
        continue;
      }
      if (fun->isVarArg()) {
        LLVM_DEBUG(logger << "vararg function: skipping\n");
        continue;
      }

      Argument* arg = fun->getArg(use.getOperandNo());
      changed |= mergeTypes(value, arg);
      changed |= propagateArgType(arg, valueInfo);
    }
    else if (auto* ret = dyn_cast<ReturnInst>(user)) {
      LLVM_DEBUG(
        logger.log("value used as return value in: ").logValueln(ret);
        logger << "propagating backward to function and call\n");
      Function* fun = ret->getFunction();
      changed |= mergeTypes(fun, value);
      for (User* funUser : fun->users())
        if (auto* funCall = dyn_cast<CallBase>(funUser)) {
          changed |= mergeTypes(funCall, value);
          changed |= propagateTypeAcrossCalls(funCall);
        }
    }
  }
  return changed;
}

bool DataTypeAllocationPass::propagateArgType(Argument* arg, const std::shared_ptr<ValueInfo>& valueInfo) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << Logger::Bold << "[" << __FUNCTION__ << "] " << Logger::Reset;
    logger.logValueln(arg);
    indenter.increaseIndent(););

  bool changed = false;
  Function* fun = arg->getParent();
  unsigned argIndex = arg->getArgNo();
  for (User* user : fun->users()) {
    if (auto* call = dyn_cast<CallBase>(user)) {
      Value* callArg = call->getOperand(argIndex);
      LLVM_DEBUG(
        logger.log("call: ").logValueln(call);
        logger.log("callArg: ").logValueln(callArg););

      changed |= mergeTypes(callArg, arg);
      if (auto* arg = dyn_cast<Argument>(callArg)) {
        LLVM_DEBUG(logger << "callArg is an argument itself: recursing\n");
        changed |= propagateArgType(arg, valueInfo);
      }
    }
  }
  return changed;
}

bool DataTypeAllocationPass::propagateCallType(CallBase* call) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << Logger::Bold << "[" << __FUNCTION__ << "] " << Logger::Reset;
    logger.logValueln(call);
    indenter.increaseIndent(););

  Function* fun = call->getCalledFunction();
  if (!fun) {
    LLVM_DEBUG(logger << "function reference cannot be resolved: skipping\n");
    return false;
  }
  if (!taffoInfo.isCloneFunction(*fun)) {
    LLVM_DEBUG(logger << "function is not a clone: skipping\n");
    return false;
  }
  bool changed = false;
  changed |= mergeTypes(fun, call);
  for (Instruction& inst : instructions(fun))
    if (auto* ret = dyn_cast<ReturnInst>(&inst))
      if (Value* retValue = ret->getReturnValue()) {
        changed |= mergeTypes(retValue, call);
        changed |= propagateTypeAcrossCalls(ret);
      }
  return changed;
}

bool DataTypeAllocationPass::propagateGepType(GetElementPtrInst* gep) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << Logger::Bold << "[" << __FUNCTION__ << "] " << Logger::Reset;
    logger.logValueln(gep);
    indenter.increaseIndent(););

  if (!taffoInfo.hasValueInfo(*gep))
    return false;
  Value* ptrOperand = gep->getPointerOperand();
  std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*gep);
  std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo);
  if (!scalarInfo)
    return false;
  bool changed = mergeTypeWithGepPtrOperand(gep, scalarInfo);
  if (changed)
    for (User* user : ptrOperand->users())
      if (auto* gepUser = dyn_cast<GetElementPtrInst>(user))
        propagateTypeAcrossCalls(gepUser);
  return changed;
}

bool DataTypeAllocationPass::mergeTypeWithGepPtrOperand(GetElementPtrInst* gep,
                                                        const std::shared_ptr<ScalarInfo>& gepInfo) {
  Logger& logger = log();
  Value* ptrOperand = gep->getPointerOperand();
  if (!taffoInfo.hasValueInfo(*ptrOperand))
    return false;

  std::shared_ptr<ValueInfo> ptrOperandInfo = taffoInfo.getValueInfo(*ptrOperand);
  ValueInfo* targetInfo = ptrOperandInfo.get();
  if (auto structInfo = std::dynamic_ptr_cast<StructInfo>(ptrOperandInfo)) {
    targetInfo = structInfo->getField(gep->indices());
    if (!targetInfo)
      return false;
  }

  ScalarInfo* targetScalarInfo = dyn_cast<ScalarInfo>(targetInfo);
  if (!targetScalarInfo || (!gepInfo->numericType && !targetScalarInfo->numericType))
    return false;

  if (!gepInfo->numericType) {
    gepInfo->numericType = targetScalarInfo->numericType->clone();
    LLVM_DEBUG(
      logger << "gep has no numeric type: copying it from target\n";
      logger.log("new value type: ").logln(*gepInfo->numericType, Logger::Green););
    return true;
  }
  if (!targetScalarInfo->numericType) {
    targetScalarInfo->numericType = gepInfo->numericType->clone();
    LLVM_DEBUG(
      logger << "target has no numeric type: copying it from gep\n";
      logger.log("new value type: ").logln(*targetScalarInfo->numericType, Logger::Green););
    return true;
  }
  LLVM_DEBUG(
    logger.log("gep type:    ").logln(*gepInfo->numericType, Logger::Cyan);
    logger.log("target type: ").logln(*targetScalarInfo->numericType, Logger::Cyan););
  if (*gepInfo->numericType == *targetScalarInfo->numericType) {
    LLVM_DEBUG(logger << "same type already\n";);
    return false;
  }

  std::shared_ptr<NumericTypeInfo> mergedType = strategy->merge(gepInfo->numericType, targetScalarInfo->numericType);
  gepInfo->numericType = mergedType->clone();
  targetScalarInfo->numericType = mergedType->clone();
  LLVM_DEBUG(logger.log("merged type: ").logln(*mergedType, Logger::Green););
  return true;
}

std::vector<Function*> DataTypeAllocationPass::collapseFunction(Module& m) {
  Logger& logger = log();
  std::vector<Function*> toDel;
  for (Function& f : m.functions()) {
    if (std::ranges::find(toDel, &f) != toDel.end())
      continue;
    LLVM_DEBUG(logger.log("Analyzing original function ").logValueln(&f));
    auto indenter = logger.getIndenter();
    indenter.increaseIndent();

    SmallPtrSet<Function*, 2> taffoFunctions;
    taffoInfo.getCloneFunctions(f, taffoFunctions);
    for (Function* cloneF : taffoFunctions) {
      LLVM_DEBUG(logger.log("clone: ").logValueln(cloneF));
      if (cloneF->user_empty())
        LLVM_DEBUG(logger << "clone not used anywhere: ignoring\n");
      else if (Function* eqFun = findEqFunction(cloneF, &f)) {
        LLVM_DEBUG(logger.log("replacing function clone with ").logValueln(eqFun););
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

  LLVM_DEBUG(log() << "Search eq function for " << fun->getName() << " in " << origin->getName() << " pool\n";);

  if (getFullyUnwrappedType(fun)->isFloatingPointTy() && hasDtaInfo(*fun->user_begin())) {
    if (taffoInfo.hasValueInfo(**fun->user_begin())) {
      std::shared_ptr<ValueInfo> retval = taffoInfo.getValueInfo(**fun->user_begin());
      fixSign.push_back(std::pair(-1, retval)); // ret value in signature
      LLVM_DEBUG(log() << "Return type : " << *retval << "\n";);
    }
  }

  int i = 0;
  for (Argument& arg : fun->args()) {
    if (taffoInfo.hasValueInfo(arg)) {
      fixSign.push_back(std::pair(i, taffoInfo.getValueInfo(arg)));
      LLVM_DEBUG(log() << "Arg " << i << " type : " << *taffoInfo.getValueInfo(arg) << "\n";);
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
  LLVM_DEBUG(log() << "Function " << fun->getName() << " used\n";);
  return nullptr;
}
