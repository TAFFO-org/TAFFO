#include "Optimizer.h"

#include "PtrCasts.hpp"
#include "LoopAnalyzerUtil.h"
#include "MetricBase.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Analysis/ScalarEvolution.h>

#define DEBUG_TYPE "taffo-dta"

using namespace llvm;
using namespace taffo;
using namespace tuner;


Optimizer::Optimizer(Module &mm, TaffoTuner *tuner, MetricBase *met, string modelFile, CPUCosts::CostType cType)
: metric(met), model(Model::MIN), module(mm), tuner(tuner), DisabledSkipped(0) {
  if (cType == CPUCosts::CostType::Performance) {
    cpuCosts = CPUCosts(modelFile);
  } else if (cType == CPUCosts::CostType::Size) {
    auto &TTI = tuner->getFunctionAnalysisResult<TargetIRAnalysis>(*(mm.begin()));
    cpuCosts = CPUCosts(mm, TTI);
  }

  LLVM_DEBUG(dbgs() << "\n\n\n[WARNING] Mixed precision mode enabled. This is an experimental feature. Use it at your own risk!\n\n\n";);
  cpuCosts.dump();
  LLVM_DEBUG(dbgs() << "ENOB tuning knob: " << to_string(TUNING_ENOB) << "\n";);
  LLVM_DEBUG(dbgs() << "Time tuning knob: " << to_string(TUNING_MATH) << "\n";);
  LLVM_DEBUG(dbgs() << "Time tuning CAST knob: " << to_string(TUNING_CASTING) << "\n";);
  metric->setOpt(this);

  LLVM_DEBUG(dbgs() << "has double: " << to_string(hasDouble) << "\n";);
  LLVM_DEBUG(dbgs() << "has half: " << to_string(hasHalf) << "\n";);
  LLVM_DEBUG(dbgs() << "has Quad: " << to_string(hasQuad) << "\n";);
  LLVM_DEBUG(dbgs() << "has PPC128: " << to_string(hasPPC128) << "\n";);
  LLVM_DEBUG(dbgs() << "has FP80: " << to_string(hasFP80) << "\n";);
  LLVM_DEBUG(dbgs() << "has BF16: " << to_string(hasBF16) << "\n";);
}

Optimizer::~Optimizer() = default;


void Optimizer::initialize()
{

  for (Function &f : module.functions()) {
    LLVM_DEBUG(dbgs() << "\nGetting info of " << f.getName() << ":\n");
    if (f.empty()) {
      continue;
    }
    const std::string name = f.getName().str();
    known_functions[name] = &f;
    functions_still_to_visit[name] = &f;
  }
}

void Optimizer::handleGlobal(GlobalObject *glob, shared_ptr<TunerInfo> tunerInfo)
{
  LLVM_DEBUG(dbgs() << "handleGlobal called.\n");

  auto *globalVar = dyn_cast_or_null<GlobalVariable>(glob);
  assert(globalVar && "glob is not a global variable!");

  if (!glob->getValueType()->isPointerTy()) {
    if (!tunerInfo->metadata->isConversionEnabled()) {
      LLVM_DEBUG(dbgs() << "Skipping as conversion is disabled!");
      return;
    }
    if (tunerInfo->metadata->getKind() == ValueInfo::K_Scalar) {
      LLVM_DEBUG(dbgs() << " ^ This is a real field\n");
      auto fieldInfo = std::dynamic_ptr_cast<ScalarInfo>(tunerInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n");
        return;
      }

      auto fptype = std::dynamic_ptr_cast<FixpType>(fieldInfo->numericType);
      if (!fptype) {
        LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n");
        return;
      }
      auto optInfo = metric->allocateNewVariableForValue(glob, fptype, fieldInfo->range, fieldInfo->error, false);
      metric->saveInfoForValue(glob, make_shared<OptimizerPointerInfo>(optInfo));
    } else if (tunerInfo->metadata->getKind() == ValueInfo::K_Struct) {
      LLVM_DEBUG(dbgs() << " ^ This is a real structure\n");

      auto fieldInfo = std::dynamic_ptr_cast<StructInfo>(tunerInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "No struct info. Bailing out.\n");
        return;
      }

      auto optInfo = metric->loadStructInfo(glob, fieldInfo, "");
      metric->saveInfoForValue(glob, optInfo);

    } else {
      llvm_unreachable("Unknown metadata!");
    }


  } else {
    if (!tunerInfo->metadata->isConversionEnabled()) {
      LLVM_DEBUG(dbgs() << "Skipping as conversion is disabled!");
      return;
    }
    LLVM_DEBUG(dbgs() << " ^ this is a pointer.\n");

    if (tunerInfo->metadata->getKind() == ValueInfo::K_Scalar) {
      LLVM_DEBUG(dbgs() << " ^ This is a real field ptr\n");
      auto fieldInfo = std::dynamic_ptr_cast<ScalarInfo>(tunerInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n");
        return;
      }

      auto fptype = std::dynamic_ptr_cast<FixpType>(fieldInfo->numericType);
      if (!fptype) {
        LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n");
        return;
      }
      // FIXME: hack, this is done to respect the fact that a pointer (yes, even a simple pointer) may be used by ugly people
      // as array, that are allocated through a malloc. In this way we must use this as a form of bypass. We allocate a new
      // value even if it may be overwritten at some time...

      if (globalVar->hasInitializer() && !globalVar->getInitializer()->isNullValue()) {
        LLVM_DEBUG(dbgs() << "Has initializer and it is not a null value! Need more processing!\n");
      } else {
        LLVM_DEBUG(dbgs() << "No initializer, or null value!\n");
        auto optInfo = metric->allocateNewVariableForValue(glob, fptype, fieldInfo->range, fieldInfo->error, false);
        // This is a pointer, so the reference to it is a pointer to a pointer yay
        metric->saveInfoForValue(glob, make_shared<OptimizerPointerInfo>(make_shared<OptimizerPointerInfo>(optInfo)));
      }

    } else if (tunerInfo->metadata->getKind() == ValueInfo::K_Struct) {
      LLVM_DEBUG(dbgs() << " ^ This is a real structure ptr\n");

      auto fieldInfo = std::dynamic_ptr_cast<StructInfo>(tunerInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "No struct info. Bailing out.\n");
        return;
      }

      auto optInfo = metric->loadStructInfo(glob, fieldInfo, "");
      metric->saveInfoForValue(glob, make_shared<OptimizerPointerInfo>(optInfo));

    } else {
      llvm_unreachable("Unknown metadata!");
    }
    return;
  }
}


void Optimizer::handleCallFromRoot(Function *f)
{
  // Therefore this should be added as a cost, not simply ignored

  LLVM_DEBUG(dbgs() << "\n============ FUNCTION FROM ROOT: " << f->getName() << " ============\n";);
  const std::string calledFunctionName = f->getName().str();
  LLVM_DEBUG(dbgs() << ("We are calling " + calledFunctionName + " from root\n"););

  auto function = known_functions.find(calledFunctionName);
  if (function == known_functions.end()) {
    LLVM_DEBUG(dbgs() << "Calling an external function, UNSUPPORTED at the moment.\n";);
    return;
  }

  // In teoria non dobbiamo mai pushare variabili per quanto riguarda una chiamata da root
  // Infatti, la chiamata da root implica la compatibilità con codice esterno che si aspetta che non vengano modificate
  // le call ad altri tipi. Per lo stesso motivo non serve nulla per il valore di ritorno.
  /*
  // fetch ranges of arguments
  std::list<shared_ptr<OptimizerInfo>> arg_errors;
  std::list<shared_ptr<OptimizerScalarInfo>> arg_scalar_errors;
  LLVM_DEBUG(dbgs() << ("Arguments:\n"););
  for (auto arg = f->arg_begin(); arg != f->arg_end(); arg++) {
      LLVM_DEBUG(dbgs() << "info for ";);
      (arg)->print(LLVM_DEBUG(dbgs()););
      LLVM_DEBUG(dbgs() << " --> ";);

      //if a variable was declared for type
      auto info = getInfoOfValue(arg);
      if (!info) {
          //This is needed to resolve eventual constants in function call (I'm looking at you, LLVM)
          LLVM_DEBUG(dbgs() << "No error for the argument!\n";);
      } else {
          LLVM_DEBUG(dbgs() << "Got this error: " << info->toString() << "\n";);
      }

      //Even if is a null value, we push it!
      arg_errors.push_back(info);

      //If the error is a scalar, collect it also as a scalar
      auto arg_info_scalar = std::dynamic_ptr_cast<OptimizerScalarInfo>(info);
      if (arg_info_scalar) {
          arg_scalar_errors.push_back(arg_info_scalar);
      }
      //}
      LLVM_DEBUG(dbgs() << "\n\n";);
  }
  LLVM_DEBUG(dbgs() << ("Arguments end.");*/


  auto it = functions_still_to_visit.find(calledFunctionName);
  if (it != functions_still_to_visit.end()) {
    // We mark the called function as visited from the global queue, so we will not visit it starting from root.
    functions_still_to_visit.erase(calledFunctionName);
    LLVM_DEBUG(dbgs() << "Function " << calledFunctionName << " marked as visited in global queue.\n";);
  } else {
    LLVM_DEBUG(dbgs() << "[WARNING] We already visited this function, for example when called from another function. Ignoring.\n";);

    return;
  }

  // Allocating variable for result: all returns will have the same type, and therefore a cast, if needed
  // SEE COMMENT BEFORE!
  /*shared_ptr<OptimizerInfo> retInfo;
  if (auto inputInfo = std::dynamic_ptr_cast<InputInfo>(valueInfo->metadata)) {
      auto fptype = std::dynamic_ptr_cast<FixpType>(inputInfo->IType);
      if (fptype) {
          LLVM_DEBUG(dbgs() << fptype->toString(););
          shared_ptr<OptimizerScalarInfo> result = allocateNewVariableForValue(instruction, fptype, inputInfo->IRange);
          retInfo = result;
      } else {
          LLVM_DEBUG(dbgs() << "There was an input info but no fix point associated.\n";);
      }
  } else if (auto pInfo = std::dynamic_ptr_cast<StructInfo>(valueInfo->metadata)) {
      auto info = loadStructInfo(instruction, pInfo, "");
      saveInfoForValue(instruction, info);
      retInfo = info;
  } else {
      LLVM_DEBUG(dbgs() << "No info available on return value, maybe it is not a floating point call.\n";);
  }*/

  // in retInfo we now have a variable for the return value of the function. Every return should be casted against it!

  // Obviously the type should be sufficient to contain the result

  // In this case we have no known math function.
  // We will have, when enabled, math functions. In this case these will be handled here!

  LLVM_DEBUG(dbgs() << ("The function belongs to the current module.\n"););
  // got the Function


  // check for recursion
  // no stack check for recursion from root, I hope
  /*size_t call_count = 0;
  for (size_t i = 0; i < call_stack.size(); i++) {
      if (call_stack[i] == f) {
          call_count++;
      }
  }*/

  std::list<shared_ptr<OptimizerInfo>> arg_errors;
  LLVM_DEBUG(dbgs() << ("Arguments:\n"););
  for (auto arg_i = f->arg_begin(); arg_i != f->arg_end(); arg_i++) {
    Value *value = &(*arg_i);
    LLVM_DEBUG(dbgs() << "**** ARG " << *value << "\n");

    if (!tuner->hasTunerInfo(value)) {
      LLVM_DEBUG(dbgs() << "Arg " << *value << " has no TUNER INFO, not creating variable\n");
      // Even if is a null value, we push it!
      arg_errors.push_back(nullptr);
      continue;
    }
    std::shared_ptr<TunerInfo> valueInfo = tuner->getTunerInfo(value);

    std::shared_ptr<OptimizerInfo> optScalInfo;

    /* FIXME: this is basically a copy-paste from handleAlloca. Similar instances should be de-duplicated */
    if (valueInfo->metadata->getKind() == ValueInfo::K_Scalar) {
      LLVM_DEBUG(dbgs() << "Arg " << *value << " This is a real field\n";);
      auto fieldInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n";);
        return;
      }

      auto fptype = std::dynamic_ptr_cast<FixpType>(fieldInfo->numericType);
      if (!fptype) {
        LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n";);
        return;
      }
      optScalInfo = metric->allocateNewVariableForValue(value, fptype, fieldInfo->range, fieldInfo->error,
                                              false);
    } else if (valueInfo->metadata->getKind() == ValueInfo::K_Struct) {
      LLVM_DEBUG(dbgs() << "Arg " << *value << " This is a real structure\n";);

      auto fieldInfo = std::dynamic_ptr_cast<StructInfo>(valueInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "No struct info. Bailing out.\n";);
        return;
      }

      optScalInfo = metric->loadStructInfo(value, fieldInfo, "");
    } else {
      llvm_unreachable("Unknown metadata!");
    }

    // Wrap pointers
    if (value->getType()->isPointerTy()) {
      std::shared_ptr<OptimizerPointerInfo> optPtrInfo(new OptimizerPointerInfo(optScalInfo));
      arg_errors.push_back(optPtrInfo);
    } else {
      arg_errors.push_back(optScalInfo);
    }
  }

  LLVM_DEBUG(dbgs() << ("Processing function...\n"););

  // Initialize trip count
  currentInstruction = nullptr;
  currentInstructionTripCount = 1;

  // See comment before to understand why these variable are set to nulls here
  processFunction(*f, arg_errors, nullptr);
  return;
}


list<shared_ptr<OptimizerInfo>> Optimizer::fetchFunctionCallArgumentInfo(const CallBase *call_i)
{
  // fetch ranges of arguments
  std::list<shared_ptr<OptimizerInfo>> arg_errors;
  //std::list<shared_ptr<OptimizerScalarInfo>> arg_scalar_errors; // UNUSED
  LLVM_DEBUG(dbgs() << ("Arguments:\n"););
  for (auto arg_it = call_i->arg_begin(); arg_it != call_i->arg_end(); ++arg_it) {
    LLVM_DEBUG(dbgs() << "info for ";);
    LLVM_DEBUG((*arg_it)->print(dbgs()););
    LLVM_DEBUG(dbgs() << " --> ";);

    // if a variable was declared for type
    auto info = getInfoOfValue(*arg_it);
    if (!info) {
      // This is needed to resolve eventual constants in function call (I'm looking at you, LLVM)
      LLVM_DEBUG(dbgs() << "No tuner information for the argument!\n";);
    } else {
      LLVM_DEBUG(dbgs() << "Got this error: " << info->toString() << "\n";);
    }

    // Even if is a null value, we push it!
    arg_errors.push_back(info);

    /*if (const generic_range_ptr_t arg_info = fetchInfo(*arg_it)) {*/
    // If the error is a scalar, collect it also as a scalar
    //auto arg_info_scalar = std::dynamic_ptr_cast<OptimizerScalarInfo>(info);
    //if (arg_info_scalar) {
    //  arg_scalar_errors.push_back(arg_info_scalar);
    //}
    //}
    LLVM_DEBUG(dbgs() << "\n\n";);
  }
  LLVM_DEBUG(dbgs() << ("Arguments end.\n"););
  
  return arg_errors;
}


void Optimizer::processFunction(Function &f, list<shared_ptr<OptimizerInfo>> argInfo,
                                shared_ptr<OptimizerInfo> retInfo)
{
  LLVM_DEBUG(dbgs() << "\n============ FUNCTION " << f.getName() << " ============\n";);

  if (f.arg_size() != argInfo.size()) {
    llvm_unreachable("Sizes should be equal!");
  }

  LLVM_DEBUG(dbgs() << "Processing arguments...\n");
  auto argInfoIt = argInfo.begin();
  for (auto arg = f.arg_begin(); arg != f.arg_end(); arg++, argInfoIt++) {
    if (*argInfoIt) {
      LLVM_DEBUG(dbgs() << "Copying info of argument " << *arg << ", coming from caller\n");
      metric->saveInfoForValue(&(*arg), *argInfoIt);
    } else {
      LLVM_DEBUG(dbgs() << "Argument " << *arg << " has no info, ignoring\n");
    }
  }
  LLVM_DEBUG(dbgs() << "Finished processing arguments.\n\n");

  // Even if null, we push this on the stack. The return will handle it hopefully
  retStack.push(retInfo);

  // As we have copy of the same function for
  for (inst_iterator iIt = inst_begin(&f), iItEnd = inst_end(&f); iIt != iItEnd; iIt++) {
    // C++ is horrible
    Instruction *I = &(*iIt);
    LLVM_DEBUG(dbgs() << *I << "     -having-     ");
    if (!tuner->hasTunerInfo(I) || !tuner->getTunerInfo(I)->metadata) {
      LLVM_DEBUG(dbgs() << "No TUNER INFO available.\n";);
    } else {
      LLVM_DEBUG(dbgs() << tuner->getTunerInfo(I)->metadata->toString() << "\n";);

      if (!tuner->getTunerInfo(I)->metadata->isConversionEnabled()) {
        LLVM_DEBUG(dbgs() << "Skipping as conversion is disabled!\n";);
        DisabledSkipped++;
        continue;
      } else {
        /* The VRA may leave null ranges even when conversion is enabled
         * for code that is unreachable from the starting point, so we check
         * the range and if it is null we skip this instruction */
        std::shared_ptr<TunerInfo> VI = tuner->getTunerInfo(&(*iIt));
        std::shared_ptr<ValueInfo> MDI = VI->metadata;
        std::shared_ptr<ScalarInfo> II = std::dynamic_ptr_cast<ScalarInfo>(MDI);
        if (II && II->range == nullptr && I->getType()->isFloatingPointTy()) {
          LLVM_DEBUG(dbgs() << "Skipping because there is no range!\n";);
          continue;
        }
      }
    }

    handleInstruction(I, tuner->getTunerInfo(I));
    LLVM_DEBUG(dbgs() << "\n\n";);
  }

  // When the analysis is completed, we remove the info from the stack, as it is no more necessary.
  retStack.pop();
}


shared_ptr<OptimizerInfo> Optimizer::getInfoOfValue(Value *value)
{
  assert(value && "Value must not be nullptr!");

  // Global object are constant too but we have already seen them :)
  auto findIt = valueToVariableName.find(value);
  if (findIt != valueToVariableName.end()) {
    return findIt->second;
  }

  if (auto constant = dyn_cast_or_null<Constant>(value)) {
    return metric->processConstant(constant);
  }

  LLVM_DEBUG(dbgs() << "Could not find any OPTIMIZER INFO for ");
  LLVM_DEBUG(value->print(dbgs()););
  LLVM_DEBUG(dbgs() << "     :( \n");

  return nullptr;
}

void Optimizer::handleBinaryInstruction(Instruction *instr, const unsigned OpCode, const shared_ptr<TunerInfo> &valueInfos)
{
  // We are only handling operations between floating point, as we do not care about other values when building the model
  // This is ok as floating point instruction can only be used inside floating point operations in LLVM! :D
  auto binop = dyn_cast_or_null<BinaryOperator>(instr);


  switch (OpCode) {
  case Instruction::FAdd:
    metric->handleFAdd(binop, OpCode, valueInfos);
    break;
  case Instruction::FSub:
    metric->handleFSub(binop, OpCode, valueInfos);
    break;
  case Instruction::FMul:
    metric->handleFMul(binop, OpCode, valueInfos);
    break;
  case Instruction::FDiv:;
    metric->handleFDiv(binop, OpCode, valueInfos);
    break;
  case Instruction::FRem:
    metric->handleFRem(binop, OpCode, valueInfos);
    break;

  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    LLVM_DEBUG(dbgs() << "Skipping operation between integers...\n";);
    break;
  default:
    emitError("Unhandled binary operator " + to_string(OpCode)); // unsupported operation
    break;
  }
}


void Optimizer::handleInstruction(Instruction *instruction, shared_ptr<TunerInfo> valueInfo)
{
  // This will be a mess. God bless you.
  LLVM_DEBUG(dbgs() << "Handling instruction " << (instruction->dump(), "\n"));
  currentInstruction = instruction;
  Module& M = *(instruction->getFunction()->getParent());
  unsigned int info = computeFullTripCount(tuner->getFAM(M), instruction);
  LLVM_DEBUG(dbgs() << "Optimizer: got trip count " << info << "\n");
  unsigned int prevInstrTripCount = currentInstructionTripCount;
  currentInstructionTripCount *= info;
  LLVM_DEBUG(dbgs() << "Current cumulative trip count: " << currentInstructionTripCount << "\n");

  const unsigned opCode = instruction->getOpcode();
  if (opCode == Instruction::Call) {
    metric->handleCall(instruction, valueInfo);
  } else if (Instruction::isTerminator(opCode)) {
    handleTerminators(instruction, valueInfo);
  } else if (Instruction::isCast(opCode)) {
    metric->handleCastInstruction(instruction, valueInfo);

  } else if (Instruction::isBinaryOp(opCode)) {
    handleBinaryInstruction(instruction, opCode, valueInfo);

  } else if (Instruction::isUnaryOp(opCode)) {

    switch (opCode) {
    case Instruction::FNeg:
      metric->handleFNeg(dyn_cast<UnaryOperator>(instruction), opCode, valueInfo);
      break;
    default:
      llvm_unreachable("Not handled.");
    }

  } else {
    switch (opCode) {
    // memory operations
    case Instruction::Alloca:
      metric->handleAlloca(instruction, valueInfo);
      break;
    case Instruction::Load:
      metric->handleLoad(instruction, valueInfo);
      break;
    case Instruction::Store:
      metric->handleStore(instruction, valueInfo);
      break;
    case Instruction::GetElementPtr:
      metric->handleGEPInstr(instruction, valueInfo);
      break;
    case Instruction::Fence:
      emitError("Handling of Fence not supported yet");
      break; // TODO implement
    case Instruction::AtomicCmpXchg:
      emitError("Handling of AtomicCmpXchg not supported yet");
      break; // TODO implement
    case Instruction::AtomicRMW:
      emitError("Handling of AtomicRMW not supported yet");
      break; // TODO implement

      // other operations
    case Instruction::ICmp: {
      LLVM_DEBUG(dbgs() << "Comparing two integers, skipping...\n");
      break;
    }
    case Instruction::FCmp: {
      metric->handleFCmp(instruction, valueInfo);
    } break;
    case Instruction::PHI: {
      metric->handlePhi(instruction, valueInfo);
    } break;
    case Instruction::Select:
      metric->handleSelect(instruction, valueInfo);
      break;
    case Instruction::UserOp1: // TODO implement
    case Instruction::UserOp2: // TODO implement
      emitError("Handling of UserOp not supported yet");
      break;
    case Instruction::VAArg: // TODO implement
      emitError("Handling of VAArg not supported yet");
      break;
    case Instruction::ExtractElement: // TODO implement
      emitError("Handling of ExtractElement not supported yet");
      break;
    case Instruction::InsertElement: // TODO implement
      emitError("Handling of InsertElement not supported yet");
      break;
    case Instruction::ShuffleVector: // TODO implement
      emitError("Handling of ShuffleVector not supported yet");
      break;
    case Instruction::ExtractValue: // TODO implement
      emitError("Handling of ExtractValue not supported yet");
      break;
    case Instruction::InsertValue: // TODO implement
      emitError("Handling of InsertValue not supported yet");
      break;
    case Instruction::LandingPad: // TODO implement
      emitError("Handling of LandingPad not supported yet");
      break;
    default:
      emitError("unknown instruction " + std::to_string(opCode));
      break;
    }
    // TODO here be dragons
  } // end else

  currentInstruction = nullptr;
  currentInstructionTripCount = prevInstrTripCount;
}

int Optimizer::getCurrentInstructionCost()
{
  if (MixedTripCount == false) {
    LLVM_DEBUG(dbgs() << __FUNCTION__ << ": option -mixedtripcount off, returning 1.\n");
    return 1;
  }
  if (currentInstruction == nullptr) {
    LLVM_DEBUG(dbgs() << __FUNCTION__ << ": wait, we are not processing any instruction right now... Returning 1.\n");
    return 1;
  }
  LLVM_DEBUG(dbgs() << __FUNCTION__ << ": cost appears to be trip count of " << *currentInstruction << "\n");
  return currentInstructionTripCount;
}

void Optimizer::handleTerminators(Instruction *term, shared_ptr<TunerInfo> valueInfo)
{
  const unsigned opCode = term->getOpcode();
  switch (opCode) {
  case Instruction::Ret:
    metric->handleReturn(term, valueInfo);
    break;
  case Instruction::Br:
    // TODO improve by checking condition and relatevely update BB weigths
    // do nothing
    break;
  case Instruction::Switch:
    emitError("Handling of Switch not implemented yet");
    break; // TODO implement
  case Instruction::IndirectBr:
    emitError("Handling of IndirectBr not implemented yet");
    break; // TODO implement
  case Instruction::Invoke:
    metric->handleCall(term, valueInfo);
    break;
  case Instruction::Resume:
    emitError("Handling of Resume not implemented yet");
    break; // TODO implement
  case Instruction::Unreachable:
    emitError("Handling of Unreachable not implemented yet");
    break; // TODO implement
  case Instruction::CleanupRet:
    emitError("Handling of CleanupRet not implemented yet");
    break; // TODO implement
  case Instruction::CatchRet:
    emitError("Handling of CatchRet not implemented yet");
    break; // TODO implement
  case Instruction::CatchSwitch:
    emitError("Handling of CatchSwitch not implemented yet");
    break; // TODO implement
  default:
    break;
  }

  return;
}

void Optimizer::emitError(const string &stringhina)
{
  LLVM_DEBUG(dbgs() << "[ERROR] " << stringhina << "\n");
}


bool Optimizer::finish()
{
  LLVM_DEBUG(dbgs() << "[Phi] Phi node state:\n");
  phiWatcher.dumpState();

  LLVM_DEBUG(dbgs() << "[Mem] MemPhi node state:\n");
  memWatcher.dumpState();

  bool result = model.finalizeAndSolve();

  LLVM_DEBUG(dbgs() << "Skipped conversions due to disabled flag: " << DisabledSkipped << "\n");

  return result;
}

void Optimizer::insertTypeEqualityConstraint(shared_ptr<OptimizerScalarInfo> op1, shared_ptr<OptimizerScalarInfo> op2,
                                             bool forceFixBitsConstraint)
{
  assert(op1 && op2 && "One of the info is nullptr!");


  auto constraint = vector<pair<string, double>>();
  // Inserting constraint about of the very same type


  auto eqconstraintlambda = [&](const string (tuner::OptimizerScalarInfo::*getFirstVariable)(), const std::string desc) mutable {
    constraint.clear();
    constraint.push_back(make_pair(((*op1).*getFirstVariable)(), 1.0));
    constraint.push_back(make_pair(((*op2).*getFirstVariable)(), -1.0));
    model.insertLinearConstraint(constraint, Model::EQ, 0 /*, desc*/);
  };

  eqconstraintlambda(&tuner::OptimizerScalarInfo::getFixedSelectedVariable, "fix equality");

  eqconstraintlambda(&tuner::OptimizerScalarInfo::getFloatSelectedVariable, "float equality");

  if (hasDouble)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getDoubleSelectedVariable, "double equality");

  if (hasHalf)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getHalfSelectedVariable, "Half equality");

  if (hasQuad)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getQuadSelectedVariable, "Quad equality");

  if (hasPPC128)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getPPC128SelectedVariable, "PPC128 equality");

  if (hasFP80)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getFP80SelectedVariable, "FP80 equality");

  if (hasBF16)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getBF16SelectedVariable, "FP80 equality");

  if (forceFixBitsConstraint) {
    constraint.clear();
    constraint.push_back(make_pair(op1->getFractBitsVariable(), 1.0));
    constraint.push_back(make_pair(op2->getFractBitsVariable(), -1.0));
    model.insertLinearConstraint(constraint, Model::EQ, 0);
  }
}


bool Optimizer::valueHasInfo(Value *value)
{
  return valueToVariableName.find(value) != valueToVariableName.end();
}


/*This is ugly as hell, but we use this data type to prevent creating other custom classes for nothing*/
shared_ptr<ValueInfo> Optimizer::getAssociatedMetadata(Value *pValue)
{
  auto res = getInfoOfValue(pValue);
  if (res == nullptr) {
    LLVM_DEBUG(dbgs() << __FUNCTION__ << " failed because getInfoOfValue returned nullptr.\n");
    return nullptr;
  }

  if (res->getKind() == OptimizerInfo::K_Pointer) {
    // FIXME: do we support double pointers?
    auto res1 = std::dynamic_ptr_cast<OptimizerPointerInfo>(res);
    // Unwrap pointer
    res = res1->getOptInfo();
  }

  return buildDataHierarchy(res);
}

shared_ptr<ValueInfo> Optimizer::buildDataHierarchy(shared_ptr<OptimizerInfo> info) {
  if (!info) {
    LLVM_DEBUG(dbgs() << "OptimizerInfo null, returning null\n");
    return nullptr;
  }

  if (info->getKind() == OptimizerInfo::K_Field) {
    auto i = modelvarToTType(std::dynamic_ptr_cast<OptimizerScalarInfo>(info));
    auto result = make_shared<ScalarInfo>(nullptr);
    result->numericType = i;
    return result;
  } else if (info->getKind() == OptimizerInfo::K_Struct) {
    auto sti = std::dynamic_ptr_cast<OptimizerStructInfo>(info);
    auto result = make_shared<StructInfo>(nullptr, sti->size());
    for (unsigned int i = 0; i < sti->size(); i++) {
      result->setField(i, buildDataHierarchy(sti->getField(i)));
    }

    return result;
  } else if (info->getKind() == OptimizerInfo::K_Pointer) {
    auto apr = std::dynamic_ptr_cast<OptimizerPointerInfo>(info);
    LLVM_DEBUG(dbgs() << "Unwrapping pointer...\n");
    return buildDataHierarchy(apr->getOptInfo());
  }

  LLVM_DEBUG(dbgs() << "Unknown OptimizerInfo: " << info->toString() << "\n");
  llvm_unreachable("Unknown data type");
}

shared_ptr<NumericType> Optimizer::modelvarToTType(shared_ptr<OptimizerScalarInfo> scalarInfo)
{
  if (!scalarInfo) {
    LLVM_DEBUG(dbgs() << "Nullptr scalar info!");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "model var values\n");
  double selectedFixed = model.getVariableValue(scalarInfo->getFixedSelectedVariable());
  LLVM_DEBUG(dbgs() << scalarInfo->getFixedSelectedVariable() << " " << selectedFixed << "\n");
  double selectedFloat = model.getVariableValue(scalarInfo->getFloatSelectedVariable());
  LLVM_DEBUG(dbgs() << scalarInfo->getFloatSelectedVariable() << " " << selectedFloat << "\n");
  double selectedDouble = 0;
  double selectedHalf = 0;
  double selectedFP80 = 0;
  double selectedPPC128 = 0;
  double selectedQuad = 0;
  double selectedBF16 = 0;

  if (hasDouble) {
    selectedDouble = model.getVariableValue(scalarInfo->getDoubleSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getDoubleSelectedVariable() << " " << selectedDouble << "\n");
  }
  if (hasHalf) {
    selectedHalf = model.getVariableValue(scalarInfo->getHalfSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getHalfSelectedVariable() << " " << selectedHalf << "\n");
  }
  if (hasQuad) {
    selectedQuad = model.getVariableValue(scalarInfo->getQuadSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getQuadSelectedVariable() << " " << selectedQuad << "\n");
  }
  if (hasPPC128) {
    selectedPPC128 = model.getVariableValue(scalarInfo->getPPC128SelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getPPC128SelectedVariable() << " " << selectedPPC128 << "\n");
  }
  if (hasFP80) {
    selectedFP80 = model.getVariableValue(scalarInfo->getFP80SelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getFP80SelectedVariable() << " " << selectedFP80 << "\n");
  }
  if (hasBF16) {
    selectedBF16 = model.getVariableValue(scalarInfo->getBF16SelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getBF16SelectedVariable() << " " << selectedBF16 << "\n");
  }

  double fracbits = model.getVariableValue(scalarInfo->getFractBitsVariable());

  assert(selectedDouble + selectedFixed + selectedFloat + selectedHalf + selectedFP80 + selectedPPC128 + selectedQuad + selectedBF16 == 1 &&
         "OMG! Catastrophic failure! Exactly one variable should be selected here!!!");

  if (selectedFixed == 1) {
    StatSelectedFixed++;
    return make_shared<FixpType>(scalarInfo->getTotalBits(), (int)fracbits, scalarInfo->isSigned);
  }

  if (selectedFloat == 1) {
    StatSelectedFloat++;
    return make_shared<FloatType>(FloatType::Float_float, 0);
  }

  if (selectedDouble == 1) {
    StatSelectedDouble++;
    return make_shared<FloatType>(FloatType::Float_double, 0);
  }


  if (selectedHalf == 1) {
    StatSelectedHalf++;
    return make_shared<FloatType>(FloatType::Float_half, 0);
  }

  if (selectedQuad == 1) {
    StatSelectedQuad++;
    return make_shared<FloatType>(FloatType::Float_fp128, 0);
  }

  if (selectedPPC128 == 1) {
    StatSelectedPPC128++;
    return make_shared<FloatType>(FloatType::Float_ppc_fp128, 0);
  }

  if (selectedFP80 == 1) {
    StatSelectedFP80++;
    return make_shared<FloatType>(FloatType::Float_x86_fp80, 0);
  }

  if (selectedBF16 == 1) {
    StatSelectedBF16++;
    return make_shared<FloatType>(FloatType::Float_bfloat, 0);
  }


  llvm_unreachable("Trying to implement a new datatype? look here :D");
}


void Optimizer::printStatInfos()
{
  LLVM_DEBUG(dbgs() << "Converted to fix: " << StatSelectedFixed << "\n");
  LLVM_DEBUG(dbgs() << "Converted to float: " << StatSelectedFloat << "\n");
  LLVM_DEBUG(dbgs() << "Converted to double: " << StatSelectedDouble << "\n");
  LLVM_DEBUG(dbgs() << "Converted to half: " << StatSelectedHalf << "\n");

  int total = StatSelectedFixed + StatSelectedFloat + StatSelectedDouble + StatSelectedHalf;

  LLVM_DEBUG(dbgs() << "Conversion entropy as equally distributed variables: " << -(((double)StatSelectedDouble / total) * log2(((double)StatSelectedDouble) / total) + ((double)StatSelectedFloat / total) * log2(((double)StatSelectedFloat) / total) + ((double)StatSelectedDouble / total) * log2(((double)StatSelectedDouble) / total)) << "\n";);

  /*
      ofstream statFile;
      statFile.open("./stats.txt", ios::out|ios::trunc);
      assert(statFile.is_open() && "File open failed!");
      statFile << "TOFIX, " << StatSelectedFixed << "\n";
      statFile << "TOFLOAT, " << StatSelectedFloat << "\n";
      statFile << "TODOUBLE, " << StatSelectedDouble << "\n";
      statFile << "TOHALF, " << StatSelectedHalf << "\n";
      statFile.flush();
      statFile.close();
  */
}
