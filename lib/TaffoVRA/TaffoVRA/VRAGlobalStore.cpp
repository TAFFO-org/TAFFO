#include "Debug/Logger.hpp"
#include "RangeOperations.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"
#include "VRAGlobalStore.hpp"
#include "VRAnalyzer.hpp"

#include <llvm/IR/Operator.h>

using namespace llvm;
using namespace taffo;

#define DEBUG_TYPE "taffo-vra"

void VRAGlobalStore::convexMerge(const AnalysisStore& other) {
  // Since dyn_cast<T>() does not do cross-casting, we must do this:
  if (isa<VRAnalyzer>(other))
    VRAStore::convexMerge(cast<VRAStore>(cast<VRAnalyzer>(other)));
  else if (isa<VRAGlobalStore>(other))
    VRAStore::convexMerge(cast<VRAStore>(cast<VRAGlobalStore>(other)));
  else
    VRAStore::convexMerge(cast<VRAStore>(cast<VRAFunctionStore>(other)));
}

std::shared_ptr<CodeAnalyzer> VRAGlobalStore::newCodeAnalyzer(CodeInterpreter& CI) {
  return std::make_shared<VRAnalyzer>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()), CI);
}

std::shared_ptr<AnalysisStore> VRAGlobalStore::newFunctionStore(CodeInterpreter& CI) {
  return std::make_shared<VRAFunctionStore>(std::static_ptr_cast<VRALogger>(CI.getGlobalStore()->getLogger()));
}

////////////////////////////////////////////////////////////////////////////////
// Metadata Processing
////////////////////////////////////////////////////////////////////////////////

bool VRAGlobalStore::isValidRange(const Range* rng) const {
  return rng != nullptr && !std::isnan(rng->min) && !std::isnan(rng->max);
}

void VRAGlobalStore::harvestValueInfo(Module& m) {
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  for (GlobalVariable& v : m.globals()) {
    // retrieve info about global var v, if any
    if (taffoInfo.hasValueInfo(v)) {
      std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(v);
      auto scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo);
      if (scalarInfo && isValidRange(scalarInfo->range.get())) {
        UserInput[&v] = scalarInfo;
        DerivedRanges[&v] = std::make_shared<PointerInfo>(scalarInfo);
      }
      else if (auto structInfo = std::dynamic_ptr_cast<StructInfo>(valueInfo)) {
        UserInput[&v] = structInfo;
        DerivedRanges[&v] = structInfo;
      }
    }
    else if (auto structType = std::dynamic_ptr_cast<TransparentStructType>(taffoInfo.getOrCreateTransparentType(v))) {
      DerivedRanges[&v] = ValueInfoFactory::create(structType);
    }
    else {
      std::shared_ptr<ValueInfo> constInfo = fetchConstant(&v);
      if (constInfo && isa<ScalarInfo>(constInfo.get()))
        DerivedRanges[&v] = std::make_shared<PointerInfo>(constInfo);
      else if (constInfo && isa<PointerInfo>(constInfo.get()))
        DerivedRanges[&v] = constInfo;
      else
        DerivedRanges[&v] = std::make_shared<PointerInfo>(nullptr);
    }
  }

  for (Function& f : m.functions()) {
    if (f.empty())
      continue;
    // retrieve info about function parameters
    for (Argument& arg : f.args()) {
      int argWeight = taffoInfo.getValueWeight(arg);
      if (argWeight == 1) {
        if (std::shared_ptr<ValueInfo> argInfo = taffoInfo.getValueInfo(arg)) {
          if (std::shared_ptr<PointerInfo> argPointerInfo = std::dynamic_ptr_cast<PointerInfo>(argInfo))
            UserInput[&arg] = argPointerInfo->getUnwrappedInfo();
          else
            UserInput[&arg] = std::static_ptr_cast<ValueInfoWithRange>(argInfo);
        }
      }
    }

    // retrieve info about instructions, for each basic block bb
    for (BasicBlock& bb : f) {
      for (Instruction& inst : bb) {
        // fetch info about Instruction i
        if (!taffoInfo.hasValueInfo(inst))
          continue;
        std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(inst);
        // only retain info of instruction i if its weight is lesser than
        // the weight of all of its parents
        int weight = taffoInfo.getValueWeight(inst);
        bool root = true;
        if (weight > 0) {
          if (isa<AllocaInst>(inst)) {
            // Kludge for alloca not to be always roots
            root = false;
          }
          for (auto& u : inst.operands()) {
            Value* v = u.get();
            int parentWeight;
            if (isa<Instruction>(v) || isa<GlobalVariable>(v) || isa<Argument>(v))
              parentWeight = taffoInfo.getValueWeight(*v);
            else
              continue;
            // only consider parameters with the same metadata
            if (!taffoInfo.hasValueInfo(*v))
              continue;
            std::shared_ptr<ValueInfo> parentInfo = taffoInfo.getValueInfo(*v);
            if (parentInfo != valueInfo)
              continue;
            if (parentWeight < weight) {
              root = false;
              break;
            }
          }
        }
        if (!root)
          continue;
        LLVM_DEBUG(
          Logger->lineHead();
          log() << " Considering input valueInfo of " << inst << " (weight=" << weight << ")\n");
        if (auto scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo)) {
          if (isValidRange(scalarInfo->range.get()))
            UserInput[&inst] = scalarInfo;
        }
        else if (auto structInfo = std::dynamic_ptr_cast<StructInfo>(valueInfo)) {
          if (!inst.getType()->isVoidTy())
            UserInput[&inst] = structInfo;
        }
      }
    }

  } // end iteration over Function in Module
  return;
}

void VRAGlobalStore::saveResults(Module& m) {
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  for (GlobalVariable& v : m.globals()) {
    if (const std::shared_ptr<ValueInfoWithRange> valueInfoWithRange = fetchRangeNode(&v)) {
      // retrieve existing info about global var v, if any
      if (taffoInfo.hasValueInfo(v)) {
        std::shared_ptr<ValueInfo> copiedValueInfo = taffoInfo.getValueInfo(v)->clone();
        updateValueInfo(copiedValueInfo, valueInfoWithRange);
        taffoInfo.setValueInfo(v, copiedValueInfo);
      }
      else {
        std::shared_ptr<ValueInfo> newValueInfo = valueInfoWithRange;
        taffoInfo.setValueInfo(v, newValueInfo);
      }
    }
  } // end globals

  for (Function& f : m.functions()) {
    for (Argument& arg : f.args())
      if (const std::shared_ptr<ValueInfoWithRange> argInfoWithRange = fetchRangeNode(&arg)) {
        if (taffoInfo.hasValueInfo(arg))
          updateValueInfo(taffoInfo.getValueInfo(arg), argInfoWithRange);
        else
          taffoInfo.setValueInfo(arg, argInfoWithRange);
      }

    // retrieve info about instructions, for each basic block bb
    for (BasicBlock& bb : f) {
      for (Instruction& inst : bb) {
        setConstRangeMetadata(inst);
        if (inst.getOpcode() == Instruction::Store)
          continue;
        if (const std::shared_ptr<ValueInfoWithRange> valueInfoWithRange = fetchRangeNode(&inst)) {
          if (taffoInfo.hasValueInfo(inst)) {
            std::shared_ptr<ValueInfo> copiedValueInfo = taffoInfo.getValueInfo(inst)->clone();
            updateValueInfo(copiedValueInfo, valueInfoWithRange);
            taffoInfo.setValueInfo(inst, copiedValueInfo);
          }
          else if (std::shared_ptr<ValueInfo> newValueInfo = valueInfoWithRange) {
            if (std::isa_ptr<ScalarInfo>(newValueInfo) || getFullyUnwrappedType(&inst)->isStructTy())
              taffoInfo.setValueInfo(inst, newValueInfo);
          }
        }
      } // end instruction
    } // end bb
  } // end function
  return;
}

void VRAGlobalStore::updateValueInfo(const std::shared_ptr<ValueInfo>& valueInfo,
                                     const std::shared_ptr<ValueInfoWithRange>& valueInfoWithRange) {
  if (!valueInfo || !valueInfoWithRange)
    return;
  if (const std::shared_ptr<ScalarInfo> newScalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfoWithRange)) {
    if (std::shared_ptr<Range> range = newScalarInfo->range) {
      if (std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo))
        scalarInfo->range = std::make_shared<Range>(*range);
      else
        LLVM_DEBUG(log() << "WARNING: mismatch between computed range type and metadata.\n");
    }
  }
  else if (const std::shared_ptr<StructInfo> newStructInfo = std::dynamic_ptr_cast<StructInfo>(valueInfoWithRange)) {
    if (std::shared_ptr<StructInfo> structInfo = std::dynamic_ptr_cast<StructInfo>(valueInfo)) {
      auto newFieldsIter = newStructInfo->begin();
      for (unsigned i = 0; i < structInfo->getNumFields(); i++) {
        if (newFieldsIter == newStructInfo->end())
          break;
        if (std::shared_ptr<ValueInfo> field = structInfo->getField(i))
          updateValueInfo(field, fetchRange(*newFieldsIter));
        else
          structInfo->setField(i, fetchRange(*newFieldsIter));
        newFieldsIter++;
      }
    }
  }
  else {
    llvm_unreachable("Unknown range type.");
  }
}

void VRAGlobalStore::setConstRangeMetadata(Instruction& inst) {
  unsigned opCode = inst.getOpcode();
  if (!(inst.isBinaryOp() || inst.isUnaryOp() || opCode == Instruction::Store || opCode == Instruction::Call
        || opCode == Instruction::Invoke))
    return;

  for (Value* op : inst.operands()) {
    if (ConstantFP* floatConst = dyn_cast<ConstantFP>(op)) {
      APFloat apf = floatConst->getValueAPF();
      bool discard;
      apf.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToAway, &discard);
      double value = apf.convertToDouble();
      auto constInfo = std::make_shared<ScalarInfo>(nullptr, std::make_shared<Range>(value, value));
      TaffoInfo::getInstance().setValueInfo(*floatConst, constInfo);
    }
  }
}

std::shared_ptr<Range> VRAGlobalStore::fetchRange(const Value* v) {
  if (const std::shared_ptr<Range> derived = VRAStore::fetchRange(v))
    return derived;

  if (std::shared_ptr<ValueInfoWithRange> inputRange = getUserInput(v)) {
    inputRange = inputRange->clone<ValueInfoWithRange>();
    saveValueRange(v, inputRange);
    if (const std::shared_ptr<ScalarInfo> InputScalar = std::dynamic_ptr_cast<ScalarInfo>(inputRange))
      return InputScalar->range;
  }

  return nullptr;
}

std::shared_ptr<ValueInfo> VRAGlobalStore::getNode(const Value* v) {
  std::shared_ptr<ValueInfo> valueInfo = VRAStore::getNode(v);
  if (valueInfo)
    return valueInfo;

  if (const Constant* constant = dyn_cast_or_null<Constant>(v)) {
    std::shared_ptr<ValueInfo> constantInfo = fetchConstant(constant);
    DerivedRanges[v] = constantInfo;
    return constantInfo;
  }

  return nullptr;
}

std::shared_ptr<ValueInfoWithRange> VRAGlobalStore::fetchRangeNode(const Value* v) {
  const std::shared_ptr<ValueInfoWithRange> derived = VRAStore::fetchRangeNode(v);

  if (auto inputRange = getUserInput(v)) {
    inputRange = inputRange->clone<ValueInfoWithRange>();
    if (derived && std::isa_ptr<StructInfo>(derived))
      return fillRangeHoles(derived, inputRange);

    const auto scalarInput = std::dynamic_ptr_cast<ScalarInfo>(inputRange);
    if (scalarInput && scalarInput->isFinal())
      return inputRange;
  }

  return derived;
}

std::shared_ptr<ValueInfoWithRange> VRAGlobalStore::getUserInput(const Value* v) const {
  auto iter = UserInput.find(v);
  if (iter != UserInput.end())
    return iter->second;
  return nullptr;
}

std::shared_ptr<ValueInfo> VRAGlobalStore::fetchConstant(const Constant* constant) {
  if (const ConstantInt* intConst = dyn_cast<ConstantInt>(constant)) {
    const double val = static_cast<double>(intConst->getSExtValue());
    return std::make_shared<ScalarInfo>(nullptr, std::make_shared<Range>(val, val));
  }
  if (const ConstantFP* floatConst = dyn_cast<ConstantFP>(constant)) {
    APFloat floatVal = floatConst->getValueAPF();
    bool losesInfo;
    floatVal.convert(APFloatBase::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
    const double doubleVal = floatVal.convertToDouble();
    return std::make_shared<ScalarInfo>(nullptr, std::make_shared<Range>(doubleVal, doubleVal));
  }
  if (isa<ConstantTokenNone>(constant)) {
    LLVM_DEBUG(Logger->logInfo("Warning: treating ConstantTokenNone as 0"));
    return std::make_shared<ScalarInfo>(nullptr, std::make_shared<Range>(0, 0));
  }
  if (isa<ConstantPointerNull>(constant)) {
    LLVM_DEBUG(Logger->logInfo("Warning: found ConstantPointerNull"));
    return std::make_shared<PointerInfo>(nullptr);
  }
  if (isa<UndefValue>(constant)) {
    LLVM_DEBUG(Logger->logInfo("Warning: treating UndefValue as nullptr"));
    return nullptr;
  }
  if (const ConstantAggregateZero* zeroAggConst = dyn_cast<ConstantAggregateZero>(constant)) {
    Type* zeroAggConstType = zeroAggConst->getType();
    if (isa<StructType>(zeroAggConstType)) {
      SmallVector<std::shared_ptr<ValueInfo>, 2> Fields;
      const unsigned num_elements = zeroAggConst->getElementCount().getFixedValue();
      Fields.reserve(num_elements);
      for (unsigned i = 0; i < num_elements; i++)
        Fields.push_back(fetchConstant(zeroAggConst->getElementValue(i)));
      return std::make_shared<StructInfo>(Fields);
    }
    if (isa<ArrayType>(zeroAggConstType) || isa<VectorType>(zeroAggConstType)) {
      // arrayType or VectorType
      return fetchConstant(zeroAggConst->getElementValue(0U));
    }
    LLVM_DEBUG(Logger->logInfo("Found aggrated zeros which is neither struct neither array neither vector"));
    return nullptr;
  }
  if (const ConstantDataSequential* constSeq = dyn_cast<ConstantDataSequential>(constant)) {
    const unsigned numElements = constSeq->getNumElements();
    std::shared_ptr<Range> seqRange = nullptr;
    for (unsigned i = 0; i < numElements; i++) {
      std::shared_ptr<Range> otherRange =
        std::static_ptr_cast<ScalarInfo>(fetchConstant(constSeq->getElementAsConstant(i)))->range;
      seqRange = getUnionRange(seqRange, otherRange);
    }
    return std::make_shared<ScalarInfo>(nullptr, seqRange);
  }
  if (isa<ConstantData>(constant)) {
    // FIXME should never happen -- all subcases handled before
    LLVM_DEBUG(Logger->logInfo("Extract value from ConstantData not implemented yet"));
    return nullptr;
  }
  if (const ConstantExpr* constExpr = dyn_cast<ConstantExpr>(constant)) {
    if (auto gepOperator = dyn_cast<GEPOperator>(constExpr)) {
      Value* ptrOperand = gepOperator->getOperand(0);
      SmallVector<unsigned, 1> offset;
      if (extractGEPOffset(gepOperator->getSourceElementType(),
                           iterator_range(constExpr->op_begin() + 1, constExpr->op_end()),
                           offset)) {
        return std::make_shared<GEPInfo>(getNode(ptrOperand), offset);
      }
    }
    LLVM_DEBUG(Logger->logInfo("Could not fold a ConstantExpr"));
    return nullptr;
  }
  if (const ConstantAggregate* constAggr = dyn_cast<ConstantAggregate>(constant)) {
    // TODO implement
    if (dyn_cast<ConstantStruct>(constAggr)) {
      LLVM_DEBUG(Logger->logInfo("Constant structs not supported yet"));
      return nullptr;
    }
    // ConstantArray or ConstantVector
    std::shared_ptr<ValueInfoWithRange> range = nullptr;
    for (unsigned idx = 0; idx < constAggr->getNumOperands(); idx++) {
      std::shared_ptr<ValueInfoWithRange> elementRange =
        std::dynamic_ptr_cast_or_null<ValueInfoWithRange>(fetchConstant(constAggr->getAggregateElement(idx)));
      range = getUnionRange(range, elementRange);
    }
    return range;
    return nullptr;
  }
  if (isa<BlockAddress>(constant)) {
    LLVM_DEBUG(Logger->logInfo("Could not fetch range from BlockAddress"));
    return nullptr;
  }
  if (isa<GlobalValue>(constant)) {
    if (const auto* globalVarConst = dyn_cast<GlobalVariable>(constant)) {
      if (globalVarConst->hasInitializer()) {
        const Constant* initVal = globalVarConst->getInitializer();
        if (initVal)
          return std::make_shared<PointerInfo>(fetchConstant(initVal));
      }
      LLVM_DEBUG(Logger->logInfo("Could not derive range from a Global Variable"));
      return nullptr;
    }
    if (const auto* globalAlias = dyn_cast<GlobalAlias>(constant)) {
      LLVM_DEBUG(Logger->logInfo("Found alias"));
      const Constant* aliasee = globalAlias->getAliasee();
      return aliasee ? fetchConstant(aliasee) : nullptr;
    }
    if (isa<Function>(constant)) {
      LLVM_DEBUG(Logger->logInfo("Could not derive range from a Constant Function"));
      return nullptr;
    }
    if (isa<GlobalIFunc>(constant)) {
      LLVM_DEBUG(Logger->logInfo("Could not derive range from a Function declaration"));
      return nullptr;
    }
    // this line should never be reached
    LLVM_DEBUG(Logger->logInfo("Could not fetch range from GlobalValue"));
    return nullptr;
  }
  LLVM_DEBUG(Logger->logInfo("Could not fetch range from Constant"));
  return nullptr;
}
