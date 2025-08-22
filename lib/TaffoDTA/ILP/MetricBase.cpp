#include "MetricBase.h"
#include "Optimizer.h"
#include "PtrCasts.hpp"

#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/Debug.h>

extern llvm::cl::opt<int> TotalBits;
extern llvm::cl::opt<int> FracThreshold;

using namespace std;
using namespace llvm;
using namespace taffo;

#define DEBUG_TYPE "taffo-dta"

bool MetricBase::valueHasInfo(Value* value) { return opt->valueHasInfo(value); }

Model& MetricBase::getModel() { return opt->model; }

std::unordered_map<std::string, Function*>& MetricBase::getFunctions_still_to_visit() {
  return opt->functions_still_to_visit;
}

std::vector<Function*>& MetricBase::getCall_stack() { return opt->call_stack; }

DenseMap<Value*, std::shared_ptr<OptimizerInfo>>& MetricBase::getValueToVariableName() {
  return opt->valueToVariableName;
}

std::stack<shared_ptr<OptimizerInfo>>& MetricBase::getRetStack() { return opt->retStack; }

void MetricBase::addDisabledSkipped() { opt->DisabledSkipped++; }

shared_ptr<OptimizerInfo> MetricBase::getInfoOfValue(Value* value) { return opt->getInfoOfValue(value); }

DataTypeAllocationPass* MetricBase::getTuner() { return opt->tuner; }
PhiWatcher& MetricBase::getPhiWatcher() { return opt->phiWatcher; }
std::unordered_map<std::string, Function*>& MetricBase::getKnown_functions() { return opt->known_functions; }
MemWatcher& MetricBase::getMemWatcher() { return opt->memWatcher; }
CPUCosts& MetricBase::getCpuCosts() { return opt->cpuCosts; }

shared_ptr<OptimizerInfo> MetricBase::processConstant(Constant* constant) {
  // Constant variable should, in general, not be saved anywhere.
  // In fact, the same constant may be used in different ways, but by the fact that
  // it is a constant, it may be modified in the final code
  // For example, a double 1.00 can become a float 1.00 in one place and a fixp 1 in another!
  LLVM_DEBUG(log() << "Processing constant...\n";);

  if (dyn_cast_or_null<GlobalObject>(constant)) {
    if (getTuner()->hasTunerInfo(constant)) {
      llvm_unreachable("This should already have been handled!");
    }
    else {
      LLVM_DEBUG(log() << "Trying to process a non float global...\n";);
      return nullptr;
    }
  }

  if (dyn_cast_or_null<ConstantData>(constant)) {
    // ATM: only handling FP types, should be enough
    if (auto constantFP = dyn_cast_or_null<ConstantFP>(constant)) {
      LLVM_DEBUG(log() << "Processing FPconstant...\n";);

      APFloat tmp = constantFP->getValueAPF();
      bool losesInfo;
      tmp.convert(APFloatBase::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
      double a = tmp.convertToDouble();

      double min, max;
      min = a;
      max = a;

      Range rangeInfo(min, max);

      FixedPointTypeGenError fpgerr;
      FixedPointInfo fpInfo = fixedPointTypeFromRange(rangeInfo, &fpgerr, TotalBits, FracThreshold, 64, TotalBits);
      if (fpgerr != FixedPointTypeGenError::NoError) {
        LLVM_DEBUG(log() << "Error generating infos for constant propagation!"
                         << "\n";);
        return nullptr;
      }

      // ENOB should not be considered for constant.... It is a constant and will be converted as best as possible
      // WE DO NOT SAVE CONSTANTS INFO!
      auto info = allocateNewVariableForValue(
        constantFP, make_shared<FixedPointInfo>(fpInfo), make_shared<Range>(rangeInfo), nullptr, false, "", false);
      info->setReferToConstant(true);
      return info;
    }

    LLVM_DEBUG(log() << "[ERROR] handling unknown ConstantData, I don't know what to do: ";);
    LLVM_DEBUG(constant->print(log()););
    LLVM_DEBUG(log() << "\n";);
    return nullptr;
  }

  if (auto constantExpr = dyn_cast_or_null<ConstantExpr>(constant)) {
    if (isa<GEPOperator>(constantExpr))
      return handleGEPConstant(constantExpr);
    LLVM_DEBUG(log() << "Unknown constant expr!\n";);
    return nullptr;
  }

  LLVM_DEBUG(log() << "Cannot handle ";);
  LLVM_DEBUG(constant->print(log()););
  LLVM_DEBUG(log() << "!\n\n";);
  LLVM_DEBUG(constant->getType()->print(log()););
  llvm_unreachable("Constant not handled!");
}

shared_ptr<OptimizerInfo> MetricBase::handleGEPConstant(const ConstantExpr* cexp_i) {
  // The first operand is the beautiful object
  Value* operand = cexp_i->getOperand(0U);

  std::vector<unsigned> offset;

  // We compute all the offsets that will be used in our "data structure" to navigate it, to reach the correct range
  if (extractGEPOffset(operand->getType(),
                       iterator_range<User::const_op_iterator>(cexp_i->op_begin() + 1, cexp_i->op_end()),
                       offset)) {

    LLVM_DEBUG(log() << "Exctracted offset: [";);
    for (unsigned i = 0; i < offset.size(); i++)
      LLVM_DEBUG(log() << offset[i] << ", ";);
    LLVM_DEBUG(log() << "]\n";);
    // When we load an address from a "thing" we need to store a reference to it in order to successfully update the
    // error
    auto optInfo_t = dynamic_ptr_cast<OptimizerPointerInfo>(getInfoOfValue(operand));
    if (!optInfo_t) {
      LLVM_DEBUG(log() << "Probably trying to access a non float element, bailing out.\n";);
      return nullptr;
    }

    auto optInfo = optInfo_t->getOptInfo();
    if (!optInfo) {
      LLVM_DEBUG(log() << "Probably trying to access a non float element, bailing out.\n";);
      return nullptr;
    }
    // This will only contain displacements for struct fields...
    for (unsigned i = 0; i < offset.size(); i++) {
      auto structInfo = dynamic_ptr_cast<OptimizerStructInfo>(optInfo);
      if (!structInfo) {
        LLVM_DEBUG(log() << "Probably trying to access a non float element, bailing out.\n";);
        return nullptr;
      }

      optInfo = structInfo->getField(offset[i]);
    }

    return make_shared<OptimizerPointerInfo>(optInfo);
  }
  return nullptr;
}

void emitError(string stringhina) { LLVM_DEBUG(log() << "[ERROR] " << stringhina << "\n"); }

shared_ptr<OptimizerStructInfo> MetricBase::loadStructInfo(Value* glob, shared_ptr<StructInfo> pInfo, string name) {
  shared_ptr<OptimizerStructInfo> optInfo = make_shared<OptimizerStructInfo>(pInfo->getNumFields());

  int i = 0;
  for (auto it = pInfo->begin(); it != pInfo->end(); it++) {
    if (auto structInfo = dynamic_ptr_cast<StructInfo>(*it)) {
      optInfo->setField(i, loadStructInfo(glob, structInfo, name + "_" + to_string(i)));
    }
    else if (auto ii = dyn_cast_or_null<ScalarInfo>(it->get())) {
      auto fptype = dynamic_ptr_cast<FixedPointInfo>(ii->numericType);
      if (!fptype) {
        LLVM_DEBUG(log() << "No fixed point info associated. Bailing out.\n");
      }
      else {
        auto info = allocateNewVariableForValue(glob, fptype, ii->range, ii->error, false, name + "_" + to_string(i));
        optInfo->setField(i, info);
      }
    }
    else {
      LLVM_DEBUG(log() << "no info for struct member " << i << " of " << *glob);
    }
    i++;
  }

  return optInfo;
}

void MetricBase::handleGEPInstr(Instruction* gep, shared_ptr<TunerInfo> valueInfo) {
  const GetElementPtrInst* gep_i = dyn_cast<GetElementPtrInst>(gep);
  LLVM_DEBUG(log() << "Handling GEP. \n";);

  Value* operand = gep_i->getOperand(0);
  LLVM_DEBUG(log() << "Operand: ";);
  LLVM_DEBUG(operand->print(log()););
  LLVM_DEBUG(log() << "\n";);

  LLVM_DEBUG(log() << "type = " << *gep_i->getType() << "\n");

  std::vector<unsigned> offset;

  if (extractGEPOffset(gep_i->getPointerOperandType(),
                       iterator_range<User::const_op_iterator>(gep_i->idx_begin(), gep_i->idx_end()),
                       offset)) {
    LLVM_DEBUG(log() << "Extracted offset: [";);
    for (unsigned i = 0; i < offset.size(); i++)
      LLVM_DEBUG(log() << offset[i] << ", ";);
    LLVM_DEBUG(log() << "]\n";);
    // When we load an address from a "thing" we need to store a reference to it in order to successfully update the
    // error
    auto baseinfo = getInfoOfValue(operand);
    if (!baseinfo) {
      LLVM_DEBUG(
        log() << "Operand pointer info missing; probably trying to access a non float element, bailing out.\n";);
      return;
    }
    auto optInfo_t = dynamic_ptr_cast<OptimizerPointerInfo>(baseinfo);
    if (!optInfo_t) {
      LLVM_DEBUG(log() << "Operand pointer info has the wrong type!! Probably trying to access a non float element, "
                          "bailing out.\n";);
      LLVM_DEBUG(log() << "wrong info: " << baseinfo->toString() << "\n");
      return;
    }

    auto optInfo = optInfo_t->getOptInfo();
    if (!optInfo) {
      LLVM_DEBUG(
        log() << "Operand pointed value info null; probably trying to access a non float element, bailing out.\n";);
      return;
    }

    // This will only contain displacements for struct fields...
    for (unsigned i = 0; i < offset.size(); i++) {
      auto structInfo = dynamic_ptr_cast<OptimizerStructInfo>(optInfo);
      if (!structInfo) {
        LLVM_DEBUG(
          log()
            << "Pointer value info kind is not struct, probably trying to access a non float element, bailing out.\n";);
        return;
      }

      optInfo = structInfo->getField(offset[i]);
    }

    LLVM_DEBUG(log() << "Infos associated: " << optInfo->toString() << "\n";);
    saveInfoForValue(gep, make_shared<OptimizerPointerInfo>(optInfo));
    return;
  }
  emitError("Cannot extract GEPOffset!");
}

bool MetricBase::extractGEPOffset(const Type* source_element_type,
                                  const iterator_range<User::const_op_iterator> indices,
                                  std::vector<unsigned>& offset) {
  assert(source_element_type != nullptr);
  LLVM_DEBUG((log() << "extractGEPOffset() BEGIN\n"););

  for (auto idx_it = indices.begin(); idx_it != indices.end(); ++idx_it) {
    if (isa<ArrayType>(source_element_type) || isa<VectorType>(source_element_type)
        || isa<PointerType>(source_element_type)) {
      // This is needed to skip the array element in array of structures
      // In facts, we treats arrays as "scalar" things, so we just do not want to deal with them
      source_element_type = source_element_type->getContainedType(0);
      LLVM_DEBUG(log() << "skipping array/vector/pointer...\n");
      continue;
    }

    const ConstantInt* int_i = dyn_cast<ConstantInt>(*idx_it);
    if (int_i) {
      int n = static_cast<int>(int_i->getSExtValue());

      source_element_type = source_element_type->getContainedType(n);
      offset.push_back(n);
      /*source_element_type =
              cast<StructType>(source_element_type)->getTypeAtIndex(n);*/
      LLVM_DEBUG(log() << "contained type " << n << ": " << *source_element_type
                       << " (ID=" << source_element_type->getTypeID() << ")\n");
    }
    else {
      // We can skip only if is a sequential i.e. we are accessing an index of an array
      emitError("Index of GEP not constant");
      return false;
    }
  }
  LLVM_DEBUG((log() << "extractGEPOffset() END\n"););
  return true;
}

void MetricBase::handleFCmp(Instruction* instr, shared_ptr<TunerInfo> valueInfo) {
  assert(instr->getOpcode() == Instruction::FCmp && "Operand mismatch!");

  auto op1 = instr->getOperand(0);
  auto op2 = instr->getOperand(1);

  auto info1 = getInfoOfValue(op1);
  auto info2 = getInfoOfValue(op2);

  if (!info1 || !info2) {
    LLVM_DEBUG(log() << "One of the two values does not have info, ignoring...\n";);
    return;
  }

  if (auto scalar = dynamic_ptr_cast<OptimizerScalarInfo>(info1)) {
    if (scalar->doesReferToConstant()) {
      LLVM_DEBUG(log() << "Info1 is a constant, skipping as no further cast cost will be introduced.\n";);
      return;
    }
  }

  if (auto scalar = dynamic_ptr_cast<OptimizerScalarInfo>(info2)) {
    if (scalar->doesReferToConstant()) {
      LLVM_DEBUG(log() << "Info2 is a constant, skipping as no further cast cost will be introduced.\n";);
      return;
      ;
    }
  }

  shared_ptr<OptimizerScalarInfo> varCast1 = allocateNewVariableWithCastCost(op1, instr);
  shared_ptr<OptimizerScalarInfo> varCast2 = allocateNewVariableWithCastCost(op2, instr);

  // The two variables must only contain the same data type, no floating point value returned.
  opt->insertTypeEqualityConstraint(varCast1, varCast2, true);
}

void MetricBase::openPhiLoop(PHINode* phiNode, Value* value) { getPhiWatcher().openPhiLoop(phiNode, value); }

void MetricBase::openMemLoop(LoadInst* load, Value* value) { getMemWatcher().openPhiLoop(load, value); }

shared_ptr<OptimizerScalarInfo> MetricBase::handleUnaryOpCommon(Instruction* instr,
                                                                Value* op1,
                                                                bool forceFixEquality,
                                                                shared_ptr<TunerInfo> valueInfos) {
  auto info1 = getInfoOfValue(op1);

  if (!info1) {
    LLVM_DEBUG(log() << "Value does not have info, ignoring...\n";);
    return nullptr;
  }

  auto inputInfo = dynamic_ptr_cast<ScalarInfo>(valueInfos->metadata);
  if (!inputInfo) {
    LLVM_DEBUG(log() << "No info on destination, bailing out, bug in VRA?\n";);
    return nullptr;
  }

  auto fptype = dynamic_ptr_cast<FixedPointInfo>(inputInfo->numericType);
  if (!fptype) {
    LLVM_DEBUG(log() << "No fixed point info associated. Bailing out.\n";);
    return nullptr;
  }

  shared_ptr<OptimizerScalarInfo> varCast1 = allocateNewVariableWithCastCost(op1, instr);

  // Obviously the type should be sufficient to contain the result
  shared_ptr<OptimizerScalarInfo> result =
    allocateNewVariableForValue(instr, fptype, inputInfo->range, inputInfo->error);

  opt->insertTypeEqualityConstraint(varCast1, result, forceFixEquality);

  return result;
}

shared_ptr<OptimizerScalarInfo> MetricBase::handleBinOpCommon(
  Instruction* instr, Value* op1, Value* op2, bool forceFixEquality, shared_ptr<TunerInfo> valueInfos) {
  auto info1 = getInfoOfValue(op1);
  auto info2 = getInfoOfValue(op2);

  if (!info1 || !info2) {
    LLVM_DEBUG(log() << "One of the two values does not have info, ignoring...\n";);
    return nullptr;
  }

  auto inputInfo = dynamic_ptr_cast<ScalarInfo>(valueInfos->metadata);
  if (!inputInfo) {
    LLVM_DEBUG(log() << "No info on destination, bailing out, bug in VRA?\n";);
    return nullptr;
  }

  auto fptype = dynamic_ptr_cast<FixedPointInfo>(inputInfo->numericType);
  if (!fptype) {
    LLVM_DEBUG(log() << "No fixed point info associated. Bailing out.\n";);
    return nullptr;
  }

  shared_ptr<OptimizerScalarInfo> varCast1 = allocateNewVariableWithCastCost(op1, instr);
  shared_ptr<OptimizerScalarInfo> varCast2 = allocateNewVariableWithCastCost(op2, instr);

  // Obviously the type should be sufficient to contain the result
  shared_ptr<OptimizerScalarInfo> result =
    allocateNewVariableForValue(instr, fptype, inputInfo->range, inputInfo->error);

  opt->insertTypeEqualityConstraint(varCast1, varCast2, forceFixEquality);
  opt->insertTypeEqualityConstraint(varCast1, result, forceFixEquality);

  return result;
}

void MetricBase::handleCall(Instruction* instruction, shared_ptr<TunerInfo> valueInfo) {
  const auto* call_i = dyn_cast<CallBase>(instruction);
  if (!call_i) {
    llvm_unreachable("Cannot cast a call instruction to CallBase");
    return;
  }

  // FIXME: each time a call is not handled, a forced casting to the older type is needed.
  // Therefore this should be added as a cost, not simply ignored

  // fetch function name
  Function* callee = call_i->getCalledFunction();
  if (callee == nullptr) {
    emitError("Indirect calls not supported!");
    return;
  }

  const std::string calledFunctionName = callee->getName().str();
  LLVM_DEBUG(log() << ("We are calling " + calledFunctionName + "\n"););

  auto function = getKnown_functions().find(calledFunctionName);
  if (function == getKnown_functions().end()) {
    const auto intrinsicsID = callee->getIntrinsicID();
    if (intrinsicsID != Intrinsic::not_intrinsic) {
      switch (intrinsicsID) {
        // TODO: implement the rest of the libc...
        /*case Intrinsic::log2:
        case Intrinsic::sqrt:
        case Intrinsic::powi:
        case Intrinsic::sin:
        case Intrinsic::cos:
        case Intrinsic::pow:
        case Intrinsic::exp:
        case Intrinsic::exp2:
        case Intrinsic::log:
        case Intrinsic::log10:
        case Intrinsic::fma:
        case Intrinsic::fabs:
        case Intrinsic::floor:
        case Intrinsic::ceil:
        case Intrinsic::trunc:
        case Intrinsic::rint:
        case Intrinsic::nearbyint:
        case Intrinsic::round:*/
        // FIXME: we, for now, emulate the support for these intrinsic; in the real case, these calls have
        //  their counterpart in all the versions, so they can be treated in some other ways

        break;
      default: emitError("skipping intrinsic " + calledFunctionName); return;
      }
    }
    LLVM_DEBUG(log() << "Handling external function call, we will convert all to original parameters.\n";);
    handleUnknownFunction(instruction, valueInfo);
    return;
  }

  // fetch ranges of arguments
  std::list<shared_ptr<OptimizerInfo>> arg_errors = opt->fetchFunctionCallArgumentInfo(call_i);

  // Allocating variable for result: all returns will have the same type, and therefore a cast, if needed
  shared_ptr<OptimizerInfo> retInfo;
  if (auto inputInfo = dynamic_ptr_cast<ScalarInfo>(valueInfo->metadata)) {
    if (instruction->getType()->isFloatingPointTy()) {
      auto fptype = dynamic_ptr_cast<FixedPointInfo>(inputInfo->numericType);
      if (fptype) {
        LLVM_DEBUG(
          log() << fptype->toString();
          log() << "\n";
          log() << "Info: " << inputInfo->toString() << "\n";);
        shared_ptr<OptimizerScalarInfo> result =
          allocateNewVariableForValue(instruction, fptype, inputInfo->range, inputInfo->error);
        retInfo = result;
        LLVM_DEBUG(log() << "Allocated variable for returns.\n";);
      }
      else {
        LLVM_DEBUG(log() << "There was an input info but no fix point associated.\n";);
      }
    }
    else {
      LLVM_DEBUG(log() << "Has metadata but is not a floating point!!!\n";);
    }
  }
  else if (auto pInfo = dynamic_ptr_cast<StructInfo>(valueInfo->metadata)) {
    auto info = loadStructInfo(instruction, pInfo, "");
    saveInfoForValue(instruction, info);
    retInfo = info;
    LLVM_DEBUG(log() << "Allocated variable for struct returns (?).\n";);
  }
  else {
    LLVM_DEBUG(log() << "No info available on return value, maybe it is not a floating point returning function.\n";);
  }

  // in retInfo we now have a variable for the return value of the function. Every return should be casted against it!

  auto it = getFunctions_still_to_visit().find(calledFunctionName);
  if (it != getFunctions_still_to_visit().end()) {
    // We mark the called function as visited from the global queue, so we will not visit it starting from root.
    getFunctions_still_to_visit().erase(calledFunctionName);
    LLVM_DEBUG(log() << "Function " << calledFunctionName << " marked as visited in global queue.\n";);
  }
  else {
    LLVM_DEBUG(
      log() << "\n\n==================================================\n";
      log() << "FUNCTION ALREADY VISITED!\n";
      log() << "As we have already visited the function we can not visit it again, as it will cause errors.\n";
      log() << "Probably, some precedent component of TAFFO did not clone this function, therefore this error.\n";
      log() << "==================================================\n\n";);
    // Ok it may happen to visit the same function two times. In this case, just reuse the variable. If the function was
    // cloneable, TAFFO would have already done it!
    return;
  }

  // Obviously the type should be sufficient to contain the result
  /* THIS IS A WHITELIST! USE FOR DEBUG
   * auto nm = callee->getName();
  if (!nm.equals("main") &&
      !nm.equals("_Z9bs_threadPv") &&
      !nm.equals("_Z19BlkSchlsEqEuroNoDivdddddifPdS_.3") &&
          !nm.equals("_Z19BlkSchlsEqEuroNoDivfffffifPfS_.5")&&
          !nm.equals("_Z4CNDFf.2.13")&&
          !nm.equals("CNDF.1")) {
      log() << "HALTING CALLING DUE TO DEBUG REQUEST!";
      return;
  }*/

  // In this case we have no known math function.
  // We will have, when enabled, math functions. In this case these will be handled here!

  // if not a whitelisted then try to fetch it from Module
  // fetch Function
  if (function != getKnown_functions().end()) {
    LLVM_DEBUG(log() << ("The function belongs to the current module.\n"););
    // got the Function
    Function* f = function->second;

    // check for recursion
    size_t call_count = 0;
    for (size_t i = 0; i < getCall_stack().size(); i++)
      if (getCall_stack()[i] == f)
        call_count++;

    // WE DO NOT SUPPORT RECURSION!
    if (call_count <= 1) {
      // Can process
      // update parameter metadata
      LLVM_DEBUG(log() << ("Processing function...\n"););
      opt->processFunction(*f, arg_errors, retInfo);
      LLVM_DEBUG(log() << "Finished processing call " << calledFunctionName << " : ";);
    }
    else {
      emitError("Recursion NOT supported!");
      return;
    }
  }
  else {
    // FIXME: this branch is totally avoided as it is handled before, make the correct corrections
    // the function was not found in current module: it is undeclared
    const auto intrinsicsID = callee->getIntrinsicID();
    if (intrinsicsID == Intrinsic::not_intrinsic) {
      // TODO: handle case of external function call: element must be in original type form and result is forced to be a
      // double
    }
    else {
      switch (intrinsicsID) {
      case Intrinsic::memcpy:
        // handleMemCpyIntrinsics(call_i);
        llvm_unreachable("Memcpy not handled atm!");
        break;
      default: emitError("skipping intrinsic " + calledFunctionName);
      }
      // TODO handle case of llvm intrinsics function call
    }
  }

  return;
}

void MetricBase::handleReturn(Instruction* instr, shared_ptr<TunerInfo> valueInfo) {
  const auto* ret_i = dyn_cast<ReturnInst>(instr);
  if (!ret_i) {
    llvm_unreachable("Could not convert Return Instruction to ReturnInstr");
    return;
  }

  Value* ret_val = ret_i->getReturnValue();

  if (!ret_val) {
    LLVM_DEBUG(log() << "Handling return void, doing nothing.\n";);
    return;
  }

  // When returning, we must return the same data type used.
  // Therefore we should eventually take into account the conversion cost.
  auto regInfo = getInfoOfValue(ret_val);
  if (!regInfo) {
    LLVM_DEBUG(log() << "No info on returned value, maybe a non float return, forgetting about it.\n";);
    return;
  }

  auto info = getRetStack().top();
  if (!info) {
    emitError("We wanted to save a result, but on the stack there is not an info available. This maybe an error!");
    return;
  }

  auto infoLinear = dynamic_ptr_cast<OptimizerScalarInfo>(info);
  if (!infoLinear)
    llvm_unreachable("Structure return still not handled!");

  auto allocated = allocateNewVariableWithCastCost(ret_val, instr);
  opt->insertTypeEqualityConstraint(infoLinear, allocated, true);
}

void MetricBase::saveInfoForPointer(Value* value, shared_ptr<OptimizerPointerInfo> pointerInfo) {
  assert(value && "Value cannot be null!");
  assert(pointerInfo && "Pointer info cannot be nullptr!");

  auto info = getInfoOfValue(value);
  if (!info) {
    LLVM_DEBUG(log() << "Storing new info for the value!\n";);
    saveInfoForValue(value, pointerInfo);
    return;
  }

  LLVM_DEBUG(log() << "Updating info of pointer...\n";);

  // PointerInfo() -> PointerInfo() -> Value[s]
  auto info_old = dynamic_ptr_cast<OptimizerPointerInfo>(info);
  assert(info_old && info_old->getOptInfo()->getKind() == OptimizerInfo::K_Pointer);
  info_old = dynamic_ptr_cast<OptimizerPointerInfo>(info_old->getOptInfo());
  assert(info_old);
  // We here should have info about the pointed element
  auto info_old_pointee = info_old->getOptInfo();
  if (info_old_pointee->getKind() == OptimizerInfo::K_Pointer) {
    LLVM_DEBUG(log() << "[WARNING] not handling pointer to pointer update!\n";);
    return;
  }

  // Same code but for new data
  auto info_new = dynamic_ptr_cast<OptimizerPointerInfo>(pointerInfo);
  assert(info_new && info_new->getOptInfo()->getKind() == OptimizerInfo::K_Pointer);
  info_new = dynamic_ptr_cast<OptimizerPointerInfo>(info_new->getOptInfo());
  assert(info_new);
  // We here should have info about the pointed element
  auto info_new_pointee = info_new->getOptInfo();
  if (info_new_pointee->getKind() == OptimizerInfo::K_Pointer) {
    LLVM_DEBUG(log() << "[WARNING] not handling pointer to pointer update (and also unpredicted state!)!\n";);
    return;
  }

  if (info_old_pointee->getKind() != info_new_pointee->getKind()) {
    LLVM_DEBUG(log() << "[WARNING] This pointer will in a point refer to two different variable that may have "
                        "different data types.\n"
                        "The results may be unpredictable, you have been warned!\n";);
  }

  if (!info_old_pointee->operator==(*info_new_pointee)) {
    LLVM_DEBUG(log() << "[WARNING] This pointer will in a point refer to two different variable that may have "
                        "different data types.\n"
                        "The results may be unpredictable, you have been warned!\n";);
  }

  getValueToVariableName()[value] = pointerInfo;
}

void MetricBase::handleUnknownFunction(Instruction* instruction, shared_ptr<TunerInfo> valueInfo) {

  assert(instruction && "Instruction is nullptr");
  const auto* call_i = dyn_cast<CallBase>(instruction);
  LLVM_DEBUG(log() << "=====> Unknown function handling: " << call_i->getCalledFunction()->getName() << "\n";);

  assert(call_i && "Cannot cast instruction to call!");
  shared_ptr<OptimizerScalarInfo> retInfo;
  // handling return value. We will force it to be in the original type.
  if (auto inputInfo = dynamic_ptr_cast<ScalarInfo>(valueInfo->metadata)) {
    if (inputInfo->range && instruction->getType()->isFloatingPointTy()) {
      auto fptype = dynamic_ptr_cast<FixedPointInfo>(inputInfo->numericType);
      if (fptype) {
        LLVM_DEBUG(log() << fptype->toString(););
        shared_ptr<OptimizerScalarInfo> result =
          allocateNewVariableForValue(instruction, fptype, inputInfo->range, inputInfo->error, true, "", true, false);
        retInfo = result;
        LLVM_DEBUG(log() << "Correctly handled. New info: " << retInfo->toString() << "\n";);
      }
      else {
        LLVM_DEBUG(log() << "There was an input info but no fix point associated.\n";);
      }
    }
    else {
      LLVM_DEBUG(log() << "The call does not return a floating point value.\n";);
    }
  }
  else if (auto pInfo = dynamic_ptr_cast<StructInfo>(valueInfo->metadata)) {
    emitError("The function considered returns a struct [?]\n");
    return;
  }
  else {
    LLVM_DEBUG(log() << "No info available on return value, maybe it is not a floating point returning function.\n";);
  }

  // If we have info on return value, forcing the return value in the model to be of the returned type of function
  if (retInfo) {
    if (hasDouble && instruction->getType()->isDoubleTy()) {
      auto constraint = vector<pair<string, double>>();
      constraint.clear();
      constraint.push_back(make_pair(retInfo->getDoubleSelectedVariable(), 1.0));
      getModel().insertLinearConstraint(constraint, Model::EQ, 1 /*, "Type constraint for return value"*/);
      LLVM_DEBUG(log() << "Forced return cast to double.\n";);
    }
    else if (instruction->getType()->isFloatingPointTy()) {
      auto constraint = vector<pair<string, double>>();
      constraint.clear();
      constraint.push_back(make_pair(retInfo->getFloatSelectedVariable(), 1.0));
      getModel().insertLinearConstraint(constraint, Model::EQ, 1 /*, "Type constraint for return value"*/);
      LLVM_DEBUG(log() << "Forced return cast to float.\n";);
    }
    else if (instruction->getType()->isFloatingPointTy()) {
      LLVM_DEBUG(log() << "The function returns a floating point type not implemented in the model. Bailing out.\n";);
    }
    else {
      LLVM_DEBUG(log() << "Probably the functions returns a pointer but i do not known what to do!\n";);
    }
  }

  // Return value handled, now it's time for parameters
  LLVM_DEBUG(log() << ("Arguments:\n"););
  int arg = 0;
  for (auto arg_it = call_i->arg_begin(); arg_it != call_i->arg_end(); ++arg_it, arg++) {
    LLVM_DEBUG(log() << "[" << arg << "] info for ";);
    LLVM_DEBUG((*arg_it)->print(log()););
    LLVM_DEBUG(log() << " --> ";);

    // if a variable was declared for type
    auto info = getInfoOfValue(*arg_it);
    if (!info) {
      // This is needed to resolve eventual constants in function call (I'm looking at you, LLVM)
      LLVM_DEBUG(log() << "No info for the argument!\n";);
      continue;
    }
    else {
      LLVM_DEBUG(log() << "Got this info: " << info->toString() << "\n";);
    }

    /*if (const generic_range_ptr_t arg_info = fetchInfo(*arg_it)) {*/
    // If the error is a scalar, collect it also as a scalar
    auto arg_info_scalar = dynamic_ptr_cast<OptimizerScalarInfo>(info);
    if (arg_info_scalar) {
      // Ok, we have info and it is a scalar, let's hope that it's not a pointer
      if ((*arg_it)->getType()->isFloatingPointTy()) {
        auto info2 = allocateNewVariableWithCastCost(arg_it->get(), instruction);
        auto constraint = vector<pair<string, double>>();
        constraint.clear();
        constraint.push_back(make_pair(info2->getFloatSelectedVariable(), 1.0));
        getModel().insertLinearConstraint(constraint, Model::EQ, 1 /*, "Type constraint for argument value"*/);
        LLVM_DEBUG(log() << "Forcing argument to float type.\n";);
      }
      else if (hasDouble && (*arg_it)->getType()->isDoubleTy()) {
        auto info2 = allocateNewVariableWithCastCost(arg_it->get(), instruction);
        auto constraint = vector<pair<string, double>>();
        constraint.clear();
        constraint.push_back(make_pair(info2->getDoubleSelectedVariable(), 1.0));
        getModel().insertLinearConstraint(constraint, Model::EQ, 1 /*, "Type constraint for argument value"*/);
        LLVM_DEBUG(log() << "Forcing argument to double type.\n";);
      }
      else if ((*arg_it)->getType()->isFloatingPointTy()) {
        LLVM_DEBUG(log() << "The function uses a floating point type not implemented in the model. Bailing out.\n";);
      }
      else {
        LLVM_DEBUG(log() << "Probably the functions uses a pointer but I do not known what to do!\n";);
      }
    }
    else {
      LLVM_DEBUG(log() << "This is a struct passed to an external function but has been optimized by TAFFO. Is this "
                          "even possible???\n";);
    }

    LLVM_DEBUG(log() << "\n\n";);
  }
  LLVM_DEBUG(log() << ("Arguments end.\n"););

  LLVM_DEBUG(log() << "Function should be correctly handled now.\n";);
}

void MetricBase::handleAlloca(Instruction* instruction, shared_ptr<TunerInfo> valueInfo) {
  if (!valueInfo) {
    LLVM_DEBUG(log() << "No value info, skipping...\n";);
    return;
  }

  if (!valueInfo->metadata) {
    LLVM_DEBUG(log() << "No value metadata, skipping...\n";);
    return;
  }

  auto* alloca = dyn_cast<AllocaInst>(instruction);

  if (!alloca->getAllocatedType()->isPointerTy()) {
    if (valueInfo->metadata->getKind() == ValueInfo::K_Scalar) {
      LLVM_DEBUG(log() << " ^ This is a real field\n";);
      auto fieldInfo = dynamic_ptr_cast<ScalarInfo>(valueInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(log() << "Not enough information. Bailing out.\n\n";);
        return;
      }

      auto fptype = dynamic_ptr_cast<FixedPointInfo>(fieldInfo->numericType);
      if (!fptype) {
        LLVM_DEBUG(log() << "No fixed point info associated. Bailing out.\n";);
        return;
      }
      auto info = allocateNewVariableForValue(alloca, fptype, fieldInfo->range, fieldInfo->error, false);
      saveInfoForValue(alloca, make_shared<OptimizerPointerInfo>(info));
    }
    else if (valueInfo->metadata->getKind() == ValueInfo::K_Struct) {
      LLVM_DEBUG(log() << " ^ This is a real structure\n";);

      auto fieldInfo = dynamic_ptr_cast<StructInfo>(valueInfo->metadata);
      if (!fieldInfo) {
        LLVM_DEBUG(log() << "No struct info. Bailing out.\n";);
        return;
      }

      auto optInfo = loadStructInfo(alloca, fieldInfo, "");
      saveInfoForValue(alloca, make_shared<OptimizerPointerInfo>(optInfo));
    }
    else {
      llvm_unreachable("Unknown metadata!");
    }
  }
  else {
    LLVM_DEBUG(log() << " ^ this is a pointer, skipping as it is unsupported at the moment.\n";);
    return;
  }
}
