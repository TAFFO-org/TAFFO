#include "TypeDeducerPass.hpp"

#include "Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/Support/WithColor.h>

#define DEBUG_TYPE "taffo-typededucer"

using namespace llvm;
using namespace taffo;

PreservedAnalyses TypeDeducerPass::run(Module &m, ModuleAnalysisManager &) {
  LLVM_DEBUG(Logger::getInstance().logln("[TypeDeducerPass]", raw_ostream::Colors::MAGENTA));
  for (Function &f : m) {
    if (f.isDeclaration())
      continue;
    // Deduce instructions' types
    for (Instruction &inst : instructions(f))
      if (inst.getType()->isPointerTy()) {
        DeducedPointerType deduced = deducePointerType(&inst);
        deducedTypes.insert({&inst, deduced});
      }
    // Deduce function arguments' types
    for (Argument &arg : f.args())
      if (arg.getType()->isPointerTy()) {
        DeducedPointerType deduced = deduceArgumentPointerType(&arg);
        deducedTypes.insert({&arg, deduced});
      }
    // Deduce functions' types
    if (f.getReturnType()->isPointerTy()) {
      DeducedPointerType deduced = deduceFunctionPointerType(&f);
      deducedTypes.insert({&f, deduced});
    }
  }
  // Deduce global values' types
  for (GlobalValue &globalValue : m.globals())
    if (globalValue.getValueType()->isPointerTy()) {
      DeducedPointerType deduced = deducePointerType(&globalValue);
      deducedTypes.insert({&globalValue, deduced});
    }
  // Continue deducing types until there is no change
  bool deducedTypesChanged = true;
  while (deducedTypesChanged) {
    deducedTypesChanged = false;
    for (const auto &[value, deducedPointerType] : deducedTypes)
      if (deducedPointerType.isAmbiguous() || deducedPointerType.isOpaque()) {
        DeducedPointerType newPointerType;
        if (auto *function = dyn_cast<Function>(value))
          newPointerType = deduceFunctionPointerType(function);
        else if (auto *argument = dyn_cast<Argument>(value))
          newPointerType = deduceArgumentPointerType(argument);
        else
          newPointerType = deducePointerType(value);
        if (newPointerType != deducedPointerType) {
          deducedTypes[value] = newPointerType;
          deducedTypesChanged = true;
        }
      }
  }
  // Save deduced types
  for (const auto &[value, deducedPointerType] : deducedTypes)
    TaffoInfo::getInstance().setDeducedPointerType(*value, deducedPointerType);

  LLVM_DEBUG(logDeducedTypes());

  TaffoInfo::getInstance().dumpToFile("taffo_typededucer.json", m);

  LLVM_DEBUG(Logger::getInstance().logln("[End of TypeDeducerPass]", raw_ostream::Colors::MAGENTA));
  return PreservedAnalyses::all();
}

DeducedPointerType TypeDeducerPass::deducePointerType(Value *value) {
  if (GlobalValue *globalValue = dyn_cast<GlobalValue>(value))
    assert(globalValue->getValueType()->isPointerTy()
      && "Trying to deduce the pointer type of a global value that is not pointer");
  else
    assert(value->getType()->isPointerTy()
      && "Trying to deduce the pointer type of a value that is not a pointer");

  std::set<DeducedPointerType> &candidateTypes = this->candidateTypes[value];

  if (auto *allocaInst = dyn_cast<AllocaInst>(value))
    candidateTypes.insert(DeducedPointerType(allocaInst->getAllocatedType(), 1));
  else if (auto *loadInst = dyn_cast<LoadInst>(value))
    candidateTypes.insert(getDeducedType(loadInst->getPointerOperand(), -1));
  else if (auto *callInst = dyn_cast<CallInst>(value))
    candidateTypes.insert(getDeducedType(callInst->getCalledFunction()));

  for (User *user : value->users()) {
    if (auto *loadInst = dyn_cast<LoadInst>(user)) {
      candidateTypes.insert(getDeducedType(loadInst, 1));
    }
    else if (auto *storeInst = dyn_cast<StoreInst>(user)) {
      if (value == storeInst->getPointerOperand())
        candidateTypes.insert(getDeducedType(storeInst->getValueOperand(), 1));
    }
    else if (auto *gepInst = dyn_cast<GetElementPtrInst>(user)) {
      if (value == gepInst->getPointerOperand())
        candidateTypes.insert(DeducedPointerType(gepInst->getSourceElementType(), 1));
    }
    else if (auto *callInst = dyn_cast<CallInst>(user)) {
      for (unsigned int i = 0; i < callInst->getCalledFunction()->arg_size(); i++) {
        Value *arg = callInst->getArgOperand(i);
        if (arg == value) {
          candidateTypes.insert(getDeducedType(callInst->getCalledFunction()->getArg(i)));
          break;
        }
      }
    }
  }
  return getBestCandidateType(candidateTypes);
}

DeducedPointerType TypeDeducerPass::deduceFunctionPointerType(Function *function) {
  assert(function->getReturnType()->isPointerTy()
    && "Trying to deduce the pointer type of a function that doesn't return a pointer");

  std::set<DeducedPointerType> &candidateTypes = this->candidateTypes[function];
  for (BasicBlock &bb : *function) {
    Instruction *termInst = bb.getTerminator();
    if (ReturnInst *returnInst = llvm::dyn_cast<ReturnInst>(termInst))
      candidateTypes.insert(getDeducedType(returnInst->getReturnValue()));
  }
  return getBestCandidateType(candidateTypes);
}

DeducedPointerType TypeDeducerPass::deduceArgumentPointerType(Argument *argument) {
  assert(argument->getType()->isPointerTy()
    && "Trying to deduce the pointer type of a argument that is not a pointer");

  std::set<DeducedPointerType> &candidateTypes = this->candidateTypes[argument];

  unsigned int argIndex = argument->getArgNo();
  Function *parentF = argument->getParent();
  for (User *functionUser : parentF->users())
    if (auto *callInst = dyn_cast<CallInst>(functionUser)) {
      Value *value = callInst->getArgOperand(argIndex);
      candidateTypes.insert(getDeducedType(value));
    }
  return getBestCandidateType(candidateTypes);
}

DeducedPointerType TypeDeducerPass::getDeducedType(Value *value, unsigned int additionalIndirections) const {
  DeducedPointerType type;
  auto iter = deducedTypes.find(value);
  if (iter != deducedTypes.end()) {
    type = iter->second;
    type.indirections += additionalIndirections;
  }
  else {
    if (auto *function = dyn_cast<Function>(value))
      type = DeducedPointerType(function->getReturnType(), additionalIndirections);
    else if (auto *global = dyn_cast<GlobalValue>(value))
      type = DeducedPointerType(global->getValueType(), additionalIndirections);
    else
      type = DeducedPointerType(value->getType(), additionalIndirections);
  }
  return type;
}

DeducedPointerType TypeDeducerPass::getBestCandidateType(const std::set<DeducedPointerType> &candidates) const {
  if (candidates.empty())
    return DeducedPointerType();

  DeducedPointerType bestCandidate;
  for (const DeducedPointerType &candidate : candidates) {
    if (bestCandidate.isAmbiguous())
      bestCandidate = candidate;
    else if (!candidate.isOpaque() || candidate.indirections > bestCandidate.indirections) {
      if (!bestCandidate.isOpaque()) {
        // Different non-opaque candidate types => ambiguous
        return DeducedPointerType();
      }
      bestCandidate = candidate;
    }
  }
  return bestCandidate;
}

void TypeDeducerPass::logDeducedTypes() const {
  Logger &logger = Logger::getInstance();
  for (const auto &[value, pointerType] : deducedTypes) {
    logger.log("[Value] ", raw_ostream::Colors::BLACK);
    if (auto *f = dyn_cast<Function>(value)) {
      logger.logln(f->getName());
      logger.increaseIndent();
    }
    else {
      logger.log(toString(value));
      logger.increaseIndent();
      if (auto *inst = dyn_cast<Instruction>(value)) {
        logger.log(" [in fun] ", raw_ostream::Colors::BLACK);
        logger.logln(inst->getFunction()->getName());
      }
      else if (auto *arg = dyn_cast<Argument>(value)) {
        logger.log(" [arg of fun] ", raw_ostream::Colors::BLACK);
        logger.logln(arg->getParent()->getName());
      }
    }
    logger.log("Deduced pointer type: ");
    if (!pointerType.isAmbiguous())
      logger.logln(pointerType.toString(), raw_ostream::Colors::GREEN);
    else {
      logger.log("Ambiguous. ", raw_ostream::Colors::YELLOW);
      auto iter = candidateTypes.find(value);
      if (iter != candidateTypes.end() && !iter->second.empty()) {
        logger.log("Candidate types: [", raw_ostream::Colors::YELLOW);
        const std::set<DeducedPointerType> &candidates = iter->second;
        bool first = true;
        for (const auto &candidate : candidates) {
          if (!first) logger.log(", ", raw_ostream::Colors::YELLOW);
          else first = false;
          logger.log(candidate.toString(), raw_ostream::Colors::YELLOW);
        }
        logger.logln("]", raw_ostream::Colors::YELLOW);
      }
      else
        logger.logln("No candidate types", raw_ostream::Colors::RED);
    }
    logger.decreaseIndent();
  }
}
