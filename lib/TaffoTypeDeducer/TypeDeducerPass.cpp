#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TypeDeducerPass.hpp"
#include "Types/TransparentType.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>

#define DEBUG_TYPE "taffo-typededucer"

using namespace llvm;
using namespace taffo;

PreservedAnalyses TypeDeducerPass::run(Module& m, ModuleAnalysisManager&) {
  LLVM_DEBUG(
    Logger& logger = log();
    logger.logln("[TypeDeducerPass]", raw_ostream::Colors::MAGENTA);
    logger.logln("[Deduction iteration 0]", raw_ostream::Colors::BLUE););
  for (Function& f : m) {
    if (f.isDeclaration()) {
      // Cannot deduce the type of a declaration: just save the transparent type of the value (could be opaque pointer)
      taffoInfo.setTransparentType(f, TransparentTypeFactory::create(&f));
      continue;
    }
    // Deduce instructions' types
    for (Instruction& inst : instructions(f))
      if (inst.getType()->isPointerTy()) {
        std::shared_ptr<TransparentType> deduced = deducePointerType(&inst);
        deducedTypes.insert({&inst, deduced});
      }
      else // If there is nothing to deduce just save the transparent type of the value
        taffoInfo.setTransparentType(inst, TransparentTypeFactory::create(&inst));
    // Deduce function arguments' types
    for (Argument& arg : f.args())
      if (arg.getType()->isPointerTy()) {
        std::shared_ptr<TransparentType> deduced = deduceArgumentPointerType(&arg);
        deducedTypes.insert({&arg, deduced});
      }
      else // If there is nothing to deduce just save the transparent type of the value
        taffoInfo.setTransparentType(arg, TransparentTypeFactory::create(&arg));
    // Deduce functions' types
    if (f.getReturnType()->isPointerTy()) {
      std::shared_ptr<TransparentType> deduced = deduceFunctionPointerType(&f);
      deducedTypes.insert({&f, deduced});
    }
    else // If there is nothing to deduce just save the transparent type of the value
      taffoInfo.setTransparentType(f, TransparentTypeFactory::create(&f));
  }
  // Deduce global values' types
  for (GlobalValue& globalValue : m.globals())
    if (globalValue.getValueType()->isPointerTy()) {
      std::shared_ptr<TransparentType> deduced = deducePointerType(&globalValue);
      deducedTypes.insert({&globalValue, deduced});
    }
    else // If there is nothing to deduce just save the transparent type of the value
      taffoInfo.setTransparentType(globalValue, TransparentTypeFactory::create(&globalValue));

  // Continue deducing types until there is no change
  unsigned int iterations = 1;
  bool deducedTypesChanged = true;
  while (deducedTypesChanged) {
    LLVM_DEBUG(log() << raw_ostream::Colors::BLUE << "[Deduction iteration " << iterations << "]\n"
                     << raw_ostream::Colors::RESET);
    iterations++;
    deducedTypesChanged = false;
    for (const auto& [value, deducedType] : deducedTypes)
      if (!deducedType || deducedType->isOpaquePointer()) {
        std::shared_ptr<TransparentType> newPointerType;
        if (auto* function = dyn_cast<Function>(value))
          newPointerType = deduceFunctionPointerType(function);
        else if (auto* argument = dyn_cast<Argument>(value))
          newPointerType = deduceArgumentPointerType(argument);
        else
          newPointerType = deducePointerType(value);
        if (!newPointerType)
          continue;
        if (!deducedType || newPointerType != deducedType) {
          deducedTypes[value] = newPointerType;
          deducedTypesChanged = true;
        }
      }
  }
  LLVM_DEBUG(log().logln("[Deduction completed]", raw_ostream::Colors::BLUE));

  // Save deduced transparent types
  for (const auto& [value, deducedType] : deducedTypes)
    if (deducedType)
      TaffoInfo::getInstance().setTransparentType(*value, deducedType);
  LLVM_DEBUG(logDeducedTypes());

  TaffoInfo::getInstance().dumpToFile("taffo_typededucer.json", m);
  LLVM_DEBUG(log().logln("[End of TypeDeducerPass]", raw_ostream::Colors::MAGENTA));
  return PreservedAnalyses::all();
}

std::shared_ptr<TransparentType> TypeDeducerPass::deducePointerType(Value* value) {
  if (auto* globalValue = dyn_cast<GlobalValue>(value))
    assert(globalValue->getValueType()->isPointerTy()
           && "Trying to deduce the pointer type of a global value that is not pointer");
  else
    assert(value->getType()->isPointerTy() && "Trying to deduce the pointer type of a value that is not a pointer");

  CandidateSet& candidateTypes = this->candidateTypes[value];

  // Deduce from value
  if (auto* allocaInst = dyn_cast<AllocaInst>(value))
    candidateTypes.insert(TransparentTypeFactory::create(allocaInst->getAllocatedType(), 1));
  else if (auto* loadInst = dyn_cast<LoadInst>(value)) {
    std::shared_ptr<TransparentType> type = getDeducedType(loadInst->getPointerOperand());
    if (type->getIndirections() > 0) {
      type = type->clone();
      type->incrementIndirections(-1);
    }
    candidateTypes.insert(type);
  }
  else if (auto* gepInst = dyn_cast<GetElementPtrInst>(value)) {
    std::shared_ptr<TransparentType> pointerOperandType = getDeducedType(gepInst->getPointerOperand());
    std::shared_ptr<TransparentType> type = pointerOperandType->clone();
    bool first = true;
    for (Value* index : gepInst->indices()) {
      if (first) {
        first = false;
        if (!isa<GlobalVariable>(gepInst->getPointerOperand()))
          type->incrementIndirections(-1);
        continue;
      }
      if (auto structType = std::dynamic_ptr_cast<TransparentStructType>(type)) {
        unsigned fieldIndex = cast<ConstantInt>(index)->getZExtValue();
        type = structType->getFieldType(fieldIndex);
      }
      else if (auto arrayType = std::dynamic_ptr_cast<TransparentArrayType>(type)) {
        // any array index selects the element type
        type = arrayType->getArrayElementType();
      }
      else {
        Logger& logger = log();
        logger.log("Gep instruction ", raw_ostream::Colors::RED).logValueln(gepInst);
        logger.logln("is trying to extract from a value that is neither a struct or an array",
                     raw_ostream::Colors::RED);
        llvm_unreachable("Check this gep instruction to know what happened");
      }
    }
    type->incrementIndirections(1);
    candidateTypes.insert(type);
  }
  else if (auto* callInst = dyn_cast<CallInst>(value)) {
    candidateTypes.insert(getDeducedType(callInst->getCalledFunction()));
  }

  // Deduce from users
  for (User* user : value->users()) {
    if (auto* loadInst = dyn_cast<LoadInst>(user)) {
      std::shared_ptr<TransparentType> type = getDeducedType(loadInst)->clone();
      type->incrementIndirections(1);
      candidateTypes.insert(type);
    }
    else if (auto* storeInst = dyn_cast<StoreInst>(user)) {
      if (value == storeInst->getPointerOperand()) {
        std::shared_ptr<TransparentType> type = getDeducedType(storeInst->getValueOperand())->clone();
        type->incrementIndirections(1);
        candidateTypes.insert(type);
      }
    }
    else if (auto* gepInst = dyn_cast<GetElementPtrInst>(user)) {
      if (value == gepInst->getPointerOperand()) {
        // Search for a coherent candide type, or create it if not present
        auto iter = std::ranges::find_if(candidateTypes, [gepInst](const std::shared_ptr<TransparentType>& type) {
          return type->getUnwrappedType() == gepInst->getSourceElementType() && type->getIndirections() == 1;
        });
        std::shared_ptr<TransparentType> type;
        if (iter != candidateTypes.end())
          type = (*iter)->clone();
        else
          type = TransparentTypeFactory::create(gepInst->getSourceElementType(), 1);
        std::shared_ptr<TransparentType> containedType = type;
        unsigned int indexCount = 0, numIndices = gepInst->getNumIndices();
        bool isLastIndex = false;
        bool first = true;
        for (Value* index : gepInst->indices()) {
          indexCount++;
          isLastIndex = (indexCount == numIndices);
          if (first) {
            first = false;
            continue;
          }
          if (auto structType = std::dynamic_ptr_cast<TransparentStructType>(containedType)) {
            unsigned int fieldIndex = cast<ConstantInt>(index)->getZExtValue();
            if (isLastIndex) {
              std::shared_ptr<TransparentType> candidate = getDeducedType(gepInst)->clone();
              if (candidate->getIndirections() > 0)
                candidate->incrementIndirections(-1);
              CandidateSet fieldCandidates = {structType->getFieldType(fieldIndex), candidate};
              std::shared_ptr<TransparentType> bestCandidate = getBestCandidateType(fieldCandidates);
              structType->setFieldType(fieldIndex, bestCandidate);
            }
            else
              containedType = structType->getFieldType(fieldIndex);
          }
          else if (auto arrayType = std::dynamic_ptr_cast<TransparentArrayType>(containedType)) {
            // any array index selects the element type
            if (isLastIndex) {
              std::shared_ptr<TransparentType> candidate = getDeducedType(gepInst)->clone();
              if (candidate->getIndirections() > 0)
                candidate->incrementIndirections(-1);
              CandidateSet arrayElementCandidates = {arrayType->getArrayElementType(), candidate};
              std::shared_ptr<TransparentType> bestCandidate = getBestCandidateType(arrayElementCandidates);
              arrayType->setArrayElementType(bestCandidate);
            }
            else
              containedType = arrayType->getArrayElementType();
          }
          else {
            Logger& logger = log();
            logger.log("Gep instruction ", raw_ostream::Colors::RED).logValueln(gepInst);
            logger.logln("is trying to extract from a value that is neither a struct or an array",
                         raw_ostream::Colors::RED);
            llvm_unreachable("Check this gep instruction to know what happened");
          }
        }
        candidateTypes.insert(type);
      }
    }
    else if (auto* callInst = dyn_cast<CallInst>(user)) {
      for (unsigned int i = 0; i < callInst->getCalledFunction()->arg_size(); i++) {
        Value* arg = callInst->getArgOperand(i);
        if (arg == value) {
          candidateTypes.insert(getDeducedType(callInst->getCalledFunction()->getArg(i)));
          break;
        }
      }
    }
  }
  std::shared_ptr<TransparentType> bestCandidate = getBestCandidateType(candidateTypes);
  LLVM_DEBUG(logDeduction(value, bestCandidate, candidateTypes));
  return bestCandidate;
}

std::shared_ptr<TransparentType> TypeDeducerPass::deduceFunctionPointerType(Function* function) {
  assert(function->getReturnType()->isPointerTy()
         && "Trying to deduce the pointer type of a function that doesn't return a pointer");

  CandidateSet& candidateTypes = this->candidateTypes[function];
  for (BasicBlock& bb : *function) {
    Instruction* termInst = bb.getTerminator();
    if (auto* returnInst = dyn_cast<ReturnInst>(termInst))
      candidateTypes.insert(getDeducedType(returnInst->getReturnValue()));
  }
  std::shared_ptr<TransparentType> bestCandidate = getBestCandidateType(candidateTypes);
  LLVM_DEBUG(logDeduction(function, bestCandidate, candidateTypes));
  return bestCandidate;
}

std::shared_ptr<TransparentType> TypeDeducerPass::deduceArgumentPointerType(Argument* argument) {
  assert(argument->getType()->isPointerTy() && "Trying to deduce the pointer type of a argument that is not a pointer");

  CandidateSet& candidateTypes = this->candidateTypes[argument];

  unsigned int argIndex = argument->getArgNo();
  Function* parentF = argument->getParent();
  for (User* functionUser : parentF->users())
    if (auto* callInst = dyn_cast<CallInst>(functionUser)) {
      Value* value = callInst->getArgOperand(argIndex);
      candidateTypes.insert(getDeducedType(value));
    }
  std::shared_ptr<TransparentType> bestCandidate = getBestCandidateType(candidateTypes);
  LLVM_DEBUG(logDeduction(argument, bestCandidate, candidateTypes));
  return bestCandidate;
}

std::shared_ptr<TransparentType> TypeDeducerPass::getDeducedType(Value* value) const {
  std::shared_ptr<TransparentType> type;
  auto iter = deducedTypes.find(value);
  if (iter != deducedTypes.end() && iter->second)
    type = iter->second;
  else
    type = TransparentTypeFactory::create(value);
  return type;
}

std::shared_ptr<TransparentType> TypeDeducerPass::getBestCandidateType(const CandidateSet& candidates) const {
  if (candidates.empty())
    return nullptr;

  std::shared_ptr<TransparentType> bestCandidate;
  for (const std::shared_ptr<TransparentType>& candidate : candidates) {
    if (!candidate)
      continue;
    if (!bestCandidate)
      bestCandidate = candidate;
    else if (candidate->compareTransparency(*bestCandidate) == 1) {
      // TODO implement strict aliasing rule and most information detain
      bestCandidate = candidate;
    }
  }
  return bestCandidate;
}

void TypeDeducerPass::logDeduction(Value* value,
                                   const std::shared_ptr<TransparentType>& bestCandidate,
                                   const CandidateSet& candidates) {
  Logger& logger = log();
  logger.log("[Deducing type of] ", raw_ostream::Colors::RESET).logValueln(value);
  auto indenter = logger.getIndenter();
  indenter.increaseIndent();
  logger.log("current candidates: ").logln(candidates);
  logger.log("best candidate is ");
  if (bestCandidate)
    logger.logln(bestCandidate, raw_ostream::Colors::CYAN);
  else
    logger.logln("ambiguous", raw_ostream::Colors::YELLOW);
}

void TypeDeducerPass::logDeducedTypes() {
  Logger& logger = log();
  logger.logln("[Results]", raw_ostream::Colors::GREEN);
  for (const auto& [value, deducedType] : deducedTypes) {
    logger.log("[Value] ", raw_ostream::Colors::RESET).logValueln(value);
    auto indenter = logger.getIndenter();
    indenter.increaseIndent();
    logger.log("deduced pointer type: ");
    if (deducedType) {
      auto color = deducedType->isOpaquePointer() ? raw_ostream::Colors::YELLOW : raw_ostream::Colors::GREEN;
      logger.logln(deducedType, color);
    }
    else {
      logger.log("ambiguous: ", raw_ostream::Colors::YELLOW);
      CandidateSet& candidates = candidateTypes[value];
      candidates.erase(nullptr);
      if (!candidates.empty()) {
        logger.log("candidate types: ", raw_ostream::Colors::YELLOW);
        logger.logln(candidates, raw_ostream::Colors::YELLOW);
      }
      else
        logger.logln("no candidate types", raw_ostream::Colors::RED);
    }
  }
}
