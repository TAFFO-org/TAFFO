#include <llvm/IR/Instructions.h>

#include "../ConversionPass.hpp"
#include "../Types/ConversionType.hpp"
#include "Debug/Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <memory>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conv"

Value* ConversionPass::convertInstruction(Instruction* inst) {
  auto sanitizeValueName = [](const Value* value) {
    if (value->hasName())
      return value->getName().str();
    std::string s;
    raw_string_ostream os(s);
    value->printAsOperand(os, false);
    s = os.str();
    if (s.front() == '%')
      s.erase(0, 1);
    if (s.front() >= '0' && s.front() <= '9')
      s.insert(s.begin(), 'v');
    return s;
  };
  std::string sanitizedName = sanitizeValueName(inst);

  Value* res = unsupported;
  if (auto* alloca = dyn_cast<AllocaInst>(inst))
    res = convertAlloca(alloca);
  else if (auto* load = dyn_cast<LoadInst>(inst))
    res = convertLoad(load);
  else if (auto* store = dyn_cast<StoreInst>(inst))
    res = convertStore(store);
  else if (auto* gep = dyn_cast<GetElementPtrInst>(inst))
    res = convertGep(gep);
  else if (auto* ev = dyn_cast<ExtractValueInst>(inst))
    res = convertExtractValue(ev);
  else if (auto* iv = dyn_cast<InsertValueInst>(inst))
    res = convertInsertValue(iv);
  else if (auto* phi = dyn_cast<PHINode>(inst))
    res = convertPhi(phi);
  else if (auto* select = dyn_cast<SelectInst>(inst))
    res = convertSelect(select);
  else if (isa<CallInst>(inst) || isa<InvokeInst>(inst))
    res = convertCall(dyn_cast<CallBase>(inst));
  else if (auto* ret = dyn_cast<ReturnInst>(inst))
    res = convertRet(ret);
  else if (inst->isBinaryOp())
    res = convertBinOp(inst, *taffoConvInfo.getNewOrOldType<ConversionScalarType>(inst));
  else if (auto* atomicRMW = dyn_cast<AtomicRMWInst>(inst))
    res = convertAtomicRMW(atomicRMW);
  else if (auto* cast = dyn_cast<CastInst>(inst))
    res = convertCast(cast);
  else if (auto* fcmp = dyn_cast<FCmpInst>(inst))
    res = convertCmp(fcmp);
  else if (inst->isUnaryOp())
    res = convertUnaryOp(inst);

  bool didFallback = false;
  if (res == unsupported) {
    res = fallback(dyn_cast<Instruction>(inst));
    didFallback = true;
  }

  auto hasInstChanged = [](Instruction* oldInst, Value* newValue) {
    if (newValue != oldInst)
      return true;
    auto* newInst = cast<Instruction>(newValue);
    for (auto&& [oldOperand, newOperand] : zip(oldInst->operands(), newInst->operands()))
      if (newOperand.get() != oldOperand.get())
        return true;
    return false;
  };
  bool instChanged = hasInstChanged(inst, res);

  if (res && !res->getType()->isVoidTy() && instChanged) {
    std::string s;
    raw_string_ostream os(s);
    os << sanitizedName << ".";
    if (didFallback)
      os << "fallback";
    else if (*taffoInfo.getTransparentType(*inst) != *taffoInfo.getTransparentType(*res))
      os << *taffoConvInfo.getNewOrOldType(res);
    else
      os << "taffo";
    res->setName(os.str());
  }

  assert(res);
  return res;
}

Value* ConversionPass::convertAlloca(AllocaInst* alloca) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(alloca);
  if (valueConvInfo->isConversionDisabled())
    return alloca;

  ConversionType* convType = valueConvInfo->getNewOrOldType();
  if (*convType == *valueConvInfo->getOldType()) {
    LLVM_DEBUG(log().logln("Conversion not needed", Logger::Yellow));
    setConversionResultInfo(alloca);
    return alloca;
  }

  Type* newAllocatedLLVMType = convType->toTransparentType()->getPointedType()->toLLVMType();

  Value* arraySizeValue = alloca->getArraySize();
  Align align = alloca->getAlign();
  auto* res = new AllocaInst(newAllocatedLLVMType, alloca->getType()->getPointerAddressSpace(), arraySizeValue, align);
  res->setUsedWithInAlloca(alloca->isUsedWithInAlloca());
  res->setSwiftError(alloca->isSwiftError());
  res->insertAfter(alloca);
  setConversionResultInfo(res, alloca, convType);
  return res;
}

Value* ConversionPass::convertLoad(LoadInst* load) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(load);

  Value* ptrOperand = load->getPointerOperand();
  Value* newPtrOperand = nullptr;

  auto iter = convertedValues.find(ptrOperand);
  if (iter != convertedValues.end())
    newPtrOperand = iter->second;
  if (!newPtrOperand || newPtrOperand == ptrOperand) {
    LLVM_DEBUG(log().logln("Pointer operand was not converted: conversion not needed", Logger::Yellow));
    setConversionResultInfo(load);
    return load;
  }

  if (valueConvInfo->isConversionDisabled())
    return unsupported;

  ConversionType* newPtrOperandConvType = taffoConvInfo.getNewOrOldType(newPtrOperand);
  std::unique_ptr<TransparentType> newType = newPtrOperandConvType->toTransparentType()->getPointedType();
  Type* newLLVMType = newType->toLLVMType();
  std::unique_ptr<ConversionType> newConvType = newPtrOperandConvType->clone(*newType);

  /*if (load->getFunction()->getCallingConv() == CallingConv::SPIR_KERNEL || MetadataManager::isCudaKernel(m,
  load->getFunction())) { align = Align(fullyUnwrapPointerOrArrayType(PELType)->getScalarSizeInBits() / 8); } else*/
  Align align = load->getAlign(); // dataLayout->getABITypeAlign(newLLVMType);
  IRBuilder builder(load);
  LoadInst* res = builder.CreateLoad(newLLVMType, newPtrOperand, load->isVolatile());
  res->setAlignment(align);
  setConversionResultInfo(res, load, newConvType.get());

  if (valueConvInfo->isConversionDisabled()) {
    assert(res->getType()->isIntegerTy() && "DTA bug; improperly tagged struct/pointer!");
    return genConvertConvToFloat(res,
                                 *taffoConvInfo.getNewOrOldType<ConversionScalarType>(newPtrOperand),
                                 cast<ConversionScalarType>(*newConvType));
  }
  return res;
}

Value* ConversionPass::convertStore(StoreInst* store) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(store);

  Value* pointerOperand = store->getPointerOperand();
  Value* newPointerOperand = pointerOperand;
  auto iter = convertedValues.find(pointerOperand);
  if (iter != convertedValues.end())
    newPointerOperand = iter->second;
  bool foundConvertedPointerOperand = newPointerOperand && newPointerOperand != pointerOperand;

  Value* valueOperand = store->getValueOperand();
  Value* newValueOperand = valueOperand;
  iter = convertedValues.find(valueOperand);
  if (iter != convertedValues.end())
    newValueOperand = iter->second;
  bool foundConvertedValueOperand = newValueOperand && newValueOperand != valueOperand;

  if (!foundConvertedPointerOperand && !foundConvertedValueOperand) {
    LLVM_DEBUG(log().logln("Operands were not converted: conversion not needed", Logger::Yellow));
    setConversionResultInfo(store);
    return store;
  }

  if (valueConvInfo->isConversionDisabled())
    return unsupported;

  TransparentType* valueOperandType = taffoInfo.getTransparentType(*valueOperand);
  auto valueOperandConvType = taffoConvInfo.getNewOrOldType(newPointerOperand)->clone(*valueOperandType);
  newValueOperand = getConvertedOperand(newValueOperand, *valueOperandConvType, nullptr, ConvTypePolicy::ForceHint);

  /*if (store->getFunction()->getCallingConv() == CallingConv::SPIR_KERNEL || mdutils::MetadataManager::isCudaKernel(m,
  store->getFunction())) { align = Align(fullyUnwrapPointerOrArrayType(peltype)->getScalarSizeInBits() / 8); } else */
  Align align = store->getAlign(); // dataLayout->getABITypeAlign(valueOperandConvType->toLLVMType());
  IRBuilder builder(store);
  StoreInst* res = builder.CreateStore(newValueOperand, newPointerOperand, store->isVolatile());
  res->setAlignment(align);
  // The type of a store is always void
  setConversionResultInfo(res);
  return res;
}

Value* ConversionPass::convertGep(GetElementPtrInst* gep) {
  Value* pointerOperand = gep->getPointerOperand();
  Value* newPointerOperand = pointerOperand;
  bool convertedPointerOperand = false;
  auto iter = convertedValues.find(pointerOperand);
  if (iter != convertedValues.end()) {
    newPointerOperand = iter->second;
    if (newPointerOperand != pointerOperand)
      convertedPointerOperand = true;
  }

  SmallVector<Value*, 4> newIndices;
  bool anyConvertedIndexOperand = false;
  for (Use& indexOperand : gep->indices()) {
    auto iter = convertedValues.find(indexOperand);
    if (iter != convertedValues.end()) {
      Value* newIndexOperand = iter->second;
      newIndices.push_back(newIndexOperand);
      if (newIndexOperand != indexOperand)
        anyConvertedIndexOperand = true;
    }
    else
      newIndices.push_back(indexOperand);
  }

  if (!convertedPointerOperand && !anyConvertedIndexOperand) {
    LLVM_DEBUG(log().logln("No operand was not converted: conversion not needed", Logger::Yellow));
    setConversionResultInfo(gep);
    return gep;
  }

  if (taffoConvInfo.getValueConvInfo(gep)->isConversionDisabled())
    return unsupported;

  ConversionType* newPointerOperandConvType = taffoConvInfo.getNewOrOldType(newPointerOperand);
  std::unique_ptr<ConversionType> resConvType = newPointerOperandConvType->getGepConvType(gep->indices());

  IRBuilder<NoFolder> builder(gep);
  Value* res = builder.CreateInBoundsGEP(
    newPointerOperandConvType->toTransparentType()->getPointedType()->toLLVMType(), newPointerOperand, newIndices);

  setConversionResultInfo(res, gep, resConvType.get());
  return res;
}

Value* ConversionPass::convertExtractValue(ExtractValueInst* extractValue) {
  if (taffoConvInfo.getValueConvInfo(extractValue)->isConversionDisabled())
    return unsupported;
  IRBuilder<NoFolder> builder(extractValue);
  Value* oldval = extractValue->getAggregateOperand();
  Value* newval = convertedValues.at(oldval);
  std::unique_ptr<ConversionType> newConvType =
    taffoConvInfo.getNewOrOldType(newval)->getGepConvType(extractValue->getIndices());
  std::vector idxlist(extractValue->indices().begin(), extractValue->indices().end());
  Value* res = builder.CreateExtractValue(newval, idxlist);
  setConversionResultInfo(res, extractValue, newConvType.get());
  return res;
}

Value* ConversionPass::convertInsertValue(InsertValueInst* insertValue) {
  if (taffoConvInfo.getValueConvInfo(insertValue)->isConversionDisabled())
    return unsupported;
  IRBuilder<NoFolder> builder(insertValue);
  Value* oldAggVal = insertValue->getAggregateOperand();
  Value* newAggVal = convertedValues.at(oldAggVal);
  auto newConvType = taffoConvInfo.getNewOrOldType(newAggVal)->getGepConvType(insertValue->getIndices());
  Value* oldInsertVal = insertValue->getInsertedValueOperand();
  Value* newInsertVal;
  newInsertVal = getConvertedOperand(oldInsertVal, *newConvType, nullptr, ConvTypePolicy::ForceHint);
  std::vector idxlist(insertValue->indices().begin(), insertValue->indices().end());
  auto* res = builder.CreateInsertValue(newAggVal, newInsertVal, idxlist);
  setConversionResultInfo(res, insertValue, newConvType.get());
  return res;
}

Value* ConversionPass::convertPhi(PHINode* phi) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(phi);
  ConversionType* newConvType = valueConvInfo->getNewOrOldType();

  if (!phi->getType()->isFloatingPointTy() || valueConvInfo->isConversionDisabled()) {
    /* in the conversion chain the floating point number was converted to
     * an int at some point; we just upgrade the incoming values in place */
    /* if all of our incoming values were not converted, we want to propagate
     * that information across the phi. If at least one of them was converted
     * the phi is converted as well; otherwise it is not. */
    bool doneSomething = false;
    for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
      Value* value = phi->getIncomingValue(i);
      if (!hasConvertedValue(value))
        continue;
      Value* newValue = convertedValues.at(value);
      phi->setIncomingValue(i, newValue);
      doneSomething = true;
    }
    return doneSomething ? phi : nullptr;
  }

  /* if we have to do a type change, create a new phi node. The new type is for
   * sure that of a fixed point value; because the original type was a float
   * and thus all of its incoming values were floats */
  PHINode* res = PHINode::Create(newConvType->toLLVMType(), phi->getNumIncomingValues());
  for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
    Value* incomingValue = phi->getIncomingValue(i);
    BasicBlock* incomingBB = phi->getIncomingBlock(i);
    Value* newIncomingValue =
      getConvertedOperand(incomingValue, *newConvType, incomingBB->getTerminator(), ConvTypePolicy::ForceHint);
    if (auto* inst2 = dyn_cast<Instruction>(newIncomingValue))
      LLVM_DEBUG(log() << "warning: new phi value " << *inst2 << " not coming from the obviously correct BB\n");
    res->addIncoming(newIncomingValue, incomingBB);
  }
  res->insertAfter(phi);
  setConversionResultInfo(res, phi, newConvType);
  return res;
}

Value* ConversionPass::convertSelect(SelectInst* select) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(select);
  auto* newConvType = valueConvInfo->getNewOrOldType<ConversionScalarType>();

  if (valueConvInfo->isConversionDisabled())
    return unsupported;
  /* the condition is always a bool (i1) or a vector of bools */
  Value* newcond = convertedValues.at(select->getCondition());
  /* otherwise create a new one */
  Value* newtruev = getConvertedOperand(select->getTrueValue(), *newConvType, select, ConvTypePolicy::ForceHint);
  Value* newfalsev = getConvertedOperand(select->getFalseValue(), *newConvType, select, ConvTypePolicy::ForceHint);
  auto* res = SelectInst::Create(newcond, newtruev, newfalsev);
  res->insertAfter(select);
  setConversionResultInfo(res, select, newConvType);
  return res;
}

Value* ConversionPass::convertCall(CallBase* call) {
  Function* oldF = call->getCalledFunction();

  if (isSupportedOpenCLFunction(oldF))
    return convertOpenCLCall(call);
  if (isSupportedCudaFunction(oldF))
    return convertCudaCall(call);
  if (isSupportedMathIntrinsicFunction(oldF))
    return convertMathIntrinsicFunction(call);
  if (isSpecialFunction(oldF))
    return unsupported;

  Function* newF = functionPool[oldF];
  if (!newF) {
    LLVM_DEBUG(log().logln("no function clone: engaging fallback", Logger::Yellow));
    return unsupported;
  }
  LLVM_DEBUG(
    Logger& logger = log();
    logger << "will use converted function: ";
    logFunctionSignature(newF);
    logger << "\n";);

  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(call);
  ConversionType* newConvType = valueConvInfo->getNewOrOldType();

  std::vector<Value*> newArgs;
  for (const auto&& [callArg, funArg] : zip(call->args(), newF->args())) {
    ConversionType* argConvType = taffoConvInfo.getNewOrOldType(&funArg);
    Value* newArg = getConvertedOperand(callArg, *argConvType, call, ConvTypePolicy::ForceHint);
    newArgs.push_back(newArg);
  }

  IRBuilder builder(call);
  CallBase* res;
  if (isa<CallInst>(call))
    res = builder.CreateCall(newF, newArgs);
  else if (isa<InvokeInst>(call)) {
    auto* invoke = dyn_cast<InvokeInst>(call);
    res = builder.CreateInvoke(newF, invoke->getNormalDest(), invoke->getUnwindDest(), newArgs);
  }
  else
    llvm_unreachable("Unknown CallBase type");

  res->setCallingConv(call->getCallingConv());
  setConversionResultInfo(res, call, newConvType);
  return res;
}

Value* ConversionPass::convertRet(ReturnInst* ret) {
  Value* retValue = ret->getReturnValue();
  if (!retValue) // AKA return void
    return ret;

  auto* fun = dyn_cast<Function>(ret->getParent()->getParent());
  ValueConvInfo* funConvInfo = taffoConvInfo.getValueConvInfo(fun);
  auto* convType = funConvInfo->getNewOrOldType<ConversionType>();
  Value* newRetValue = getConvertedOperand(retValue, *convType, ret, ConvTypePolicy::ForceHint);
  ret->setOperand(0, newRetValue);
  // The type of a return instruction is always void, and it has valueConvInfo already: no need to call
  // setConversionResultInfo
  return ret;
}

Value* ConversionPass::convertUnaryOp(Instruction* inst) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(inst);
  auto* newConvType = valueConvInfo->getNewOrOldType<ConversionScalarType>();

  if (!inst->getType()->isFloatingPointTy() || valueConvInfo->isConversionDisabled())
    return unsupported;

  unsigned opc = inst->getOpcode();

  if (opc == Instruction::FNeg) {
    LLVM_DEBUG(log() << inst->getOperand(0) << "\n";);
    Value* newOperand = getConvertedOperand(inst->getOperand(0), *newConvType, inst, ConvTypePolicy::ForceHint);

    IRBuilder<NoFolder> builder(inst);
    Value* res = nullptr;

    if (newConvType->isFixedPoint())
      res = builder.CreateNeg(newOperand);
    else if (newConvType->isFloatingPoint())
      res = builder.CreateFNeg(newOperand);
    else
      llvm_unreachable("Unknown convType");

    copyValueInfo(res, inst);
    updateNumericTypeInfo(newOperand, *newConvType);
    setConversionResultInfo(res, inst, newConvType);
    return res;
  }
  return unsupported;
}

Value* ConversionPass::convertBinOp(Instruction* inst, const ConversionScalarType& convType) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(inst);
  if (valueConvInfo->isConversionDisabled())
    return unsupported;

  unsigned opcode = inst->getOpcode();
  if (opcode == Instruction::FAdd)
    return convertFAdd(inst, convType);
  if (opcode == Instruction::FSub)
    return convertFSub(inst, convType);
  if (opcode == Instruction::FRem)
    return convertFRem(inst, convType);
  if (opcode == Instruction::FMul)
    return convertFMul(inst, convType);
  if (opcode == Instruction::FDiv)
    return convertFDiv(inst, convType);

  return unsupported;
}

Value* ConversionPass::convertAtomicRMW(AtomicRMWInst* atomicRMW) {
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(atomicRMW);

  // TODO other float operations if any
  if (atomicRMW->getOperation() != AtomicRMWInst::FAdd)
    return unsupported;

  auto* convType = valueConvInfo->getNewOrOldType<ConversionScalarType>();

  Value* ptrOperand = atomicRMW->getPointerOperand();
  Value* newPtrOperand = ptrOperand;
  auto iter = convertedValues.find(ptrOperand);
  if (iter != convertedValues.end())
    newPtrOperand = iter->second;
  if (!newPtrOperand || newPtrOperand == ptrOperand) {
    LLVM_DEBUG(log().logln("Pointer operand was not converted: conversion not needed", Logger::Yellow));
    setConversionResultInfo(atomicRMW);
    return atomicRMW;
  }

  if (valueConvInfo->isConversionDisabled())
    return unsupported;

  Value* oldValueOperand = atomicRMW->getValOperand();
  Value* newValueOperand = getConvertedOperand(oldValueOperand, *convType, atomicRMW, ConvTypePolicy::ForceHint);

  AtomicRMWInst::BinOp binOp;
  if (convType->isFixedPoint())
    binOp = AtomicRMWInst::Add;
  else if (convType->isFloatingPoint())
    binOp = AtomicRMWInst::FAdd;
  else
    llvm_unreachable("Unknown convType");

  IRBuilder<NoFolder> builder(atomicRMW);
  AtomicRMWInst* res = builder.CreateAtomicRMW(binOp,
                                               newPtrOperand,
                                               newValueOperand,
                                               atomicRMW->getAlign(),
                                               atomicRMW->getOrdering(),
                                               atomicRMW->getSyncScopeID());

  setConversionResultInfo(res, atomicRMW, convType);
  return res;
}

Value* ConversionPass::convertFAdd(Instruction* inst, const ConversionScalarType& convType) {
  Value* newOperand1 = getConvertedOperand(inst->getOperand(0), convType, inst, ConvTypePolicy::ForceHint);
  Value* newOperand2 = getConvertedOperand(inst->getOperand(1), convType, inst, ConvTypePolicy::ForceHint);

  IRBuilder<NoFolder> builder(inst);
  Value* res;
  if (convType.isFixedPoint())
    res = builder.CreateBinOp(Instruction::Add, newOperand1, newOperand2);
  else if (convType.isFloatingPoint())
    res = builder.CreateBinOp(Instruction::FAdd, newOperand1, newOperand2);
  else
    llvm_unreachable("Unknown convType");

  updateNumericTypeInfo(newOperand1, convType);
  updateNumericTypeInfo(newOperand2, convType);

  setConversionResultInfo(res, inst, &convType);
  return res;
}

Value* ConversionPass::convertFSub(Instruction* inst, const ConversionScalarType& convType) {
  Value* newOperand1 = getConvertedOperand(inst->getOperand(0), convType, inst, ConvTypePolicy::ForceHint);
  Value* newOperand2 = getConvertedOperand(inst->getOperand(1), convType, inst, ConvTypePolicy::ForceHint);

  IRBuilder<NoFolder> builder(inst);
  Value* res;
  // TODO: improve overflow resistance by shifting late
  if (convType.isFixedPoint())
    res = builder.CreateBinOp(Instruction::Sub, newOperand1, newOperand2);
  else if (convType.isFloatingPoint())
    res = builder.CreateBinOp(Instruction::FSub, newOperand1, newOperand2);
  else
    llvm_unreachable("Unknown convType");

  updateNumericTypeInfo(newOperand1, convType);
  updateNumericTypeInfo(newOperand2, convType);

  setConversionResultInfo(res, inst, &convType);
  return res;
}

Value* ConversionPass::convertFRem(Instruction* inst, const ConversionScalarType& convType) {
  Value* newOperand1 = getConvertedOperand(inst->getOperand(0), convType, inst, ConvTypePolicy::ForceHint);
  Value* newOperand2 = getConvertedOperand(inst->getOperand(1), convType, inst, ConvTypePolicy::ForceHint);

  IRBuilder<NoFolder> builder(inst);
  Value* res;
  if (convType.isFixedPoint())
    if (convType.isSigned())
      res = builder.CreateBinOp(Instruction::SRem, newOperand1, newOperand2);
    else
      res = builder.CreateBinOp(Instruction::URem, newOperand1, newOperand2);
  else if (convType.isFloatingPoint())
    res = builder.CreateBinOp(Instruction::FRem, newOperand1, newOperand2);
  else
    llvm_unreachable("Unknown convType");

  updateNumericTypeInfo(newOperand1, convType);
  updateNumericTypeInfo(newOperand2, convType);

  setConversionResultInfo(res, inst, &convType);
  return res;
}

Value* ConversionPass::convertFMul(Instruction* inst, const ConversionScalarType& convType) {
  Logger& logger = log();

  if (convType.isFixedPoint()) {
    Value* operand1 = inst->getOperand(0);
    Value* operand2 = inst->getOperand(1);
    std::unique_ptr<ConversionType> convType1 = nullptr;
    std::unique_ptr<ConversionType> convType2 = nullptr;
    Value* newOperand1 = getConvertedOperand(operand1, convType, inst, ConvTypePolicy::RangeOverHint, &convType1);
    Value* newOperand2 = getConvertedOperand(operand2, convType, inst, ConvTypePolicy::RangeOverHint, &convType2);

    const ConversionScalarType& scalarConvType1 = cast<ConversionScalarType>(*convType1);
    const ConversionScalarType& scalarConvType2 = cast<ConversionScalarType>(*convType2);
    auto resConvType = ConversionScalarType(*convType.toTransparentType(),
                                            convType.isSigned(),
                                            scalarConvType1.getBits() + scalarConvType2.getBits(),
                                            scalarConvType1.getFractionalBits() + scalarConvType2.getFractionalBits());
    Type* resLLVMType = resConvType.toScalarLLVMType(inst->getContext());

    IRBuilder<NoFolder> builder(inst);
    Value* extOperand1 = newOperand1;
    Value* extOperand2 = newOperand2;
    ConversionScalarType extConvType1 = scalarConvType1;
    ConversionScalarType extConvType2 = scalarConvType2;
    Value* res = nullptr;
    if (resLLVMType->getScalarSizeInBits() > maxTotalBitsConv) {
      auto extendValueSize = [&](Value* src,
                                 const ConversionScalarType& srcConvType,
                                 const ConversionScalarType& dstConvType) -> std::pair<Value*, ConversionScalarType> {
        auto srcLLVMType = srcConvType.toScalarLLVMType(inst->getContext());
        auto dstLLVMType = dstConvType.toScalarLLVMType(inst->getContext());
        Value* extendedValue =
          srcConvType.isSigned() ? builder.CreateSExt(src, dstLLVMType) : builder.CreateZExt(src, dstLLVMType);
        updateNumericTypeInfo(extendedValue, dstConvType);

        unsigned bitsDiff = dstLLVMType->getScalarSizeInBits() - srcLLVMType->getScalarSizeInBits();
        Value* res = builder.CreateShl(extendedValue, bitsDiff);
        updateNumericTypeInfo(
          res, srcConvType.isSigned(), srcConvType.getFractionalBits() + bitsDiff, srcConvType.getBits() + bitsDiff);

        auto resConvType = ConversionScalarType(*convType.toTransparentType(),
                                                srcConvType.isSigned(),
                                                srcConvType.getBits() + bitsDiff,
                                                srcConvType.getFractionalBits() + bitsDiff);
        return {res, resConvType};
      };

      // Adjust to the same size
      if (scalarConvType1.getBits() > scalarConvType2.getBits())
        std::tie(extOperand2, extConvType2) = extendValueSize(newOperand2, scalarConvType2, scalarConvType1);
      else if (scalarConvType1.getBits() < scalarConvType2.getBits())
        std::tie(extOperand1, extConvType1) = extendValueSize(newOperand1, scalarConvType1, scalarConvType2);

      resLLVMType = convType.toScalarLLVMType(inst->getContext());

      const unsigned fracBits1 = extConvType1.getFractionalBits();
      const unsigned fracBits2 = extConvType2.getFractionalBits();
      unsigned targetFracBits = convType.getFractionalBits();
      unsigned shiftRight1 = 0;
      unsigned shiftRight2 = 0;
      unsigned newFracBits1 = fracBits1;
      unsigned newFracBits2 = fracBits2;

      if (targetFracBits % 2 == 0) {
        unsigned requiredFractBits = targetFracBits / 2;
        shiftRight1 = fracBits1 - requiredFractBits;
        shiftRight2 = fracBits2 - requiredFractBits;
      }
      else {
        unsigned required_fract = targetFracBits / 2;
        if (fracBits1 > fracBits2) {
          shiftRight1 = fracBits1 - (required_fract + 1);
          shiftRight2 = fracBits2 - required_fract;
        }
        else {
          shiftRight2 = fracBits2 - (required_fract + 1);
          shiftRight1 = fracBits1 - required_fract;
        }
      }

      // Shift to make space for all possible value
      if (shiftRight1 > 0) {
        extOperand1 = extConvType1.isSigned() ? builder.CreateAShr(extOperand1, shiftRight1)
                                              : builder.CreateLShr(extOperand1, shiftRight1);
        newFracBits1 = newFracBits1 - shiftRight1;
      }
      if (shiftRight2 > 0) {
        extOperand2 = extConvType2.isSigned() ? builder.CreateAShr(extOperand2, shiftRight2)
                                              : builder.CreateLShr(extOperand2, shiftRight2);
        newFracBits2 = newFracBits2 - shiftRight2;
      }

      unsigned newFracBits = newFracBits1 + newFracBits2;

      extConvType1.setFractionalBits(newFracBits1);
      extConvType2.setFractionalBits(newFracBits2);

      copyValueInfo(extOperand1, newOperand1);
      updateNumericTypeInfo(
        extOperand1, extConvType1.isSigned(), extConvType1.getFractionalBits(), extConvType1.getBits());
      copyValueInfo(extOperand2, newOperand2);
      updateNumericTypeInfo(
        extOperand2, extConvType2.isSigned(), extConvType2.getFractionalBits(), extConvType2.getBits());

      resConvType.setBits(extConvType1.getBits());
      resConvType.setFractionalBits(newFracBits);
      res = builder.CreateMul(extOperand1, extOperand2);

      updateNumericTypeInfo(extOperand1, extConvType1);
      updateNumericTypeInfo(extOperand2, extConvType2);
      copyValueInfo(res, inst);
      updateNumericTypeInfo(res, resConvType.isSigned(), resConvType.getFractionalBits(), resConvType.getBits());

      LLVM_DEBUG(logger.log("result type: ").logln(resConvType, Logger::Cyan));

      setConversionResultInfo(res, inst, &resConvType);
      return genConvertConvToConv(res, resConvType, convType, ConvTypePolicy::ForceHint, inst);
    }
    else {
      extOperand1 = extConvType1.isSigned() ? builder.CreateSExt(newOperand1, resLLVMType)
                                            : builder.CreateZExt(newOperand1, resLLVMType);
      extOperand2 = extConvType2.isSigned() ? builder.CreateSExt(newOperand2, resLLVMType)
                                            : builder.CreateZExt(newOperand2, resLLVMType);
      res = builder.CreateMul(extOperand1, extOperand2);
      copyValueInfo(extOperand1, newOperand1);
      copyValueInfo(extOperand2, newOperand2);

      copyValueInfo(res, inst);
      updateNumericTypeInfo(res, resConvType.isSigned(), resConvType.getFractionalBits(), resConvType.getBits());
      updateNumericTypeInfo(extOperand1, extConvType1);
      updateNumericTypeInfo(extOperand2, extConvType2);

      LLVM_DEBUG(logger.log("mul result type: ").logln(resConvType, Logger::Cyan));

      setConversionResultInfo(res, inst, &resConvType);
      return genConvertConvToConv(res, resConvType, convType, ConvTypePolicy::ForceHint, inst);
    }
  }
  else if (convType.isFloatingPoint()) {
    Value* newOperand1 = getConvertedOperand(inst->getOperand(0), convType, inst, ConvTypePolicy::ForceHint);
    Value* newOperand2 = getConvertedOperand(inst->getOperand(1), convType, inst, ConvTypePolicy::ForceHint);

    IRBuilder<NoFolder> builder(inst);
    Value* res = builder.CreateFMul(newOperand1, newOperand2);
    setConversionResultInfo(res, inst, &convType);
    return res;
  }
  llvm_unreachable("Unknown convType");
}

Value* ConversionPass::convertFDiv(Instruction* inst, const ConversionScalarType& convType) {
  Logger& logger = log();
  TransparentType* type = taffoInfo.getTransparentType(*inst);

  // TODO: fix by using HintOverRange when it is actually implemented
  if (convType.isFixedPoint()) {
    Value* operand1 = inst->getOperand(0);
    Value* operand2 = inst->getOperand(1);
    std::unique_ptr<ConversionType> convType1 = nullptr;
    std::unique_ptr<ConversionType> convType2 = nullptr;
    Value* newOperand1 = getConvertedOperand(operand1, convType, inst, ConvTypePolicy::RangeOverHint, &convType1);
    Value* newOperand2 = getConvertedOperand(operand2, convType, inst, ConvTypePolicy::RangeOverHint, &convType2);

    const ConversionScalarType& scalarConvType1 = cast<ConversionScalarType>(*convType1);
    const ConversionScalarType& scalarConvType2 = cast<ConversionScalarType>(*convType2);
    LLVM_DEBUG(
      logger << "div operand1 of type ";
      logger.log(scalarConvType1, Logger::Cyan) << ": " << *newOperand1 << "\n";
      logger << "div operand2 of type ";
      logger.log(scalarConvType2, Logger::Cyan) << ": " << *newOperand2 << "\n");

    // Compute types of the intermediates
    bool signedRes = convType.isSigned();
    unsigned extOperand2FracBits =
      std::max(0, scalarConvType2.getFractionalBits() - (signedRes && !scalarConvType2.isSigned() ? 1 : 0));
    unsigned extOperand1FracBits = convType.getFractionalBits() + extOperand2FracBits;
    unsigned bits = std::max(scalarConvType1.getBits(), scalarConvType2.getBits());
    if (extOperand1FracBits + scalarConvType1.getIntegerBits() > bits)
      bits = scalarConvType1.getBits() + scalarConvType2.getBits();

    if (bits > maxTotalBitsConv) {
      extOperand1FracBits = scalarConvType1.getFractionalBits();
      bits = convType.getBits();
      unsigned requiredFracBits = convType.getFractionalBits();

      // We want convType.getFractionalBits() == extOperand1FracBits - extOperand2FracBits
      if (extOperand1FracBits < extOperand2FracBits)
        extOperand2FracBits = extOperand1FracBits;
      if (extOperand1FracBits - extOperand2FracBits < requiredFracBits) {
        int diff = convType.getFractionalBits() - static_cast<int>(requiredFracBits);
        extOperand2FracBits =
          diff > minQuotientFrac ? static_cast<unsigned>(diff) : minQuotientFrac; // How to prevent division by 0?
      }
    }

    // Extend first operand
    auto extConvType1 =
      ConversionScalarType(*scalarConvType1.toTransparentType(), signedRes, bits, extOperand1FracBits);
    Value* extOperand1 =
      genConvertConvToConv(newOperand1, scalarConvType1, extConvType1, ConvTypePolicy::ForceHint, inst);

    // Extend second operand
    auto extConvType2 =
      ConversionScalarType(*scalarConvType2.toTransparentType(), signedRes, bits, extOperand2FracBits);
    Value* extOperand2 =
      genConvertConvToConv(newOperand2, scalarConvType2, extConvType2, ConvTypePolicy::ForceHint, inst);

    // Generate division
    IRBuilder<NoFolder> builder(inst);
    Value* res =
      convType.isSigned() ? builder.CreateSDiv(extOperand1, extOperand2) : builder.CreateUDiv(extOperand1, extOperand2);

    auto resConvType = ConversionScalarType(*type, signedRes, bits, extOperand1FracBits - extOperand2FracBits);

    LLVM_DEBUG(logger << "fdiv ext1 = " << *extOperand1 << " type = " << extConvType1 << "\n");
    LLVM_DEBUG(logger << "fdiv ext2 = " << *extOperand2 << " type = " << extConvType2 << "\n");
    LLVM_DEBUG(logger << "fdiv fixop = " << *res << " type = " << resConvType << "\n");

    copyValueInfo(extOperand1, newOperand1);
    copyValueInfo(extOperand2, newOperand2);
    copyValueInfo(res, inst);
    updateNumericTypeInfo(res, resConvType.isSigned(), resConvType.getFractionalBits(), resConvType.getBits());
    updateNumericTypeInfo(extOperand1, extConvType1);
    updateNumericTypeInfo(extOperand2, extConvType2);

    LLVM_DEBUG(logger.log("div result type: ").logln(resConvType, Logger::Cyan));

    setConversionResultInfo(res, inst, &resConvType);
    return genConvertConvToConv(res, resConvType, convType, ConvTypePolicy::ForceHint, inst);
  }
  if (convType.isFloatingPoint()) {
    Value* newOperand1 = getConvertedOperand(inst->getOperand(0), convType, inst, ConvTypePolicy::ForceHint);
    Value* newOperand2 = getConvertedOperand(inst->getOperand(1), convType, inst, ConvTypePolicy::ForceHint);

    IRBuilder<NoFolder> builder(inst);
    Value* res = builder.CreateFDiv(newOperand1, newOperand2);
    setConversionResultInfo(res, inst, &convType);
    return res;
  }
  llvm_unreachable("Unknown convType");
}

Value* ConversionPass::convertCmp(FCmpInst* fcmp) {
  Value* operand1 = fcmp->getOperand(0);
  Value* operand2 = fcmp->getOperand(1);
  ConversionScalarType* convType1 = nullptr;
  ConversionScalarType* convType2 = nullptr;
  if (taffoConvInfo.hasValueConvInfo(operand1)) {
    ValueConvInfo* valueConvInfo1 = taffoConvInfo.getValueConvInfo(operand1);
    if (!valueConvInfo1->isConstant() && !valueConvInfo1->isConversionDisabled())
      convType1 = valueConvInfo1->getNewOrOldType<ConversionScalarType>();
  }
  if (taffoConvInfo.hasValueConvInfo(operand2)) {
    ValueConvInfo* valueConvInfo2 = taffoConvInfo.getValueConvInfo(operand2);
    if (!valueConvInfo2->isConstant() && !valueConvInfo2->isConversionDisabled())
      convType2 = valueConvInfo2->getNewOrOldType<ConversionScalarType>();
  }

  if (!convType1 && !convType2)
    return fcmp;

  if (!convType2)
    convType2 = convType1;
  else if (!convType1)
    convType1 = convType2;
  bool isOneFloat = convType1->isFloatingPoint() || convType2->isFloatingPoint();

  ConversionScalarType convType = ConversionScalarType(*convType1->toTransparentType());
  if (!isOneFloat) {
    bool mixedSign = convType1->isSigned() != convType2->isSigned();
    int intBits1 = convType1->getBits() - convType1->getFractionalBits() + (mixedSign ? convType1->isSigned() : 0);
    int intBits2 = convType2->getBits() - convType2->getFractionalBits() + (mixedSign ? convType2->isSigned() : 0);
    convType.setSigned(convType1->isSigned() || convType2->isSigned());
    convType.setFractionalBits(std::max(convType1->getFractionalBits(), convType2->getFractionalBits()));
    convType.setBits(std::max(intBits1, intBits2) + convType.getFractionalBits());
    Value* convertedOperand1 = getConvertedOperand(operand1, convType, fcmp, ConvTypePolicy::ForceHint);
    Value* convertedOperand2 = getConvertedOperand(operand2, convType, fcmp, ConvTypePolicy::ForceHint);
    IRBuilder<NoFolder> builder(fcmp->getNextNode());
    CmpInst::Predicate oldPred = fcmp->getPredicate();
    CmpInst::Predicate newPred;
    bool swapped = false;
    // If unordered swap first, then convert with int, and then re-swap at the end
    if (!CmpInst::isOrdered(fcmp->getPredicate())) {
      oldPred = fcmp->getInversePredicate();
      swapped = true;
    }

    if (oldPred == CmpInst::FCMP_OEQ)
      newPred = CmpInst::ICMP_EQ;
    else if (oldPred == CmpInst::FCMP_ONE)
      newPred = CmpInst::ICMP_NE;
    else if (oldPred == CmpInst::FCMP_OGT)
      newPred = convType.isSigned() ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT;
    else if (oldPred == CmpInst::FCMP_OGE)
      newPred = convType.isSigned() ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE;
    else if (oldPred == CmpInst::FCMP_OLE)
      newPred = convType.isSigned() ? CmpInst::ICMP_SLE : CmpInst::ICMP_ULE;
    else if (oldPred == CmpInst::FCMP_OLT)
      newPred = convType.isSigned() ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
    else if (oldPred == CmpInst::FCMP_ORD) {
      // TODO gestione NaN
    }
    else if (oldPred == CmpInst::FCMP_TRUE) {
      // There is no integer-only always-true / always-false comparison operator...
      // So we roll out our own by producing a tautology
      auto* res = builder.CreateICmpEQ(ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0),
                                       ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0));
      setConversionResultInfo(res);
      return res;
    }
    else if (oldPred == CmpInst::FCMP_FALSE) {
      auto* res = builder.CreateICmpNE(ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0),
                                       ConstantInt::get(Type::getInt32Ty(fcmp->getContext()), 0));
      setConversionResultInfo(res);
      return res;
    }

    if (swapped)
      newPred = CmpInst::getInversePredicate(newPred);
    auto* res = builder.CreateICmp(newPred, convertedOperand1, convertedOperand2);
    setConversionResultInfo(res);
    return res;
  }

  // Handling the presence of at least one float:
  // Converting all to the biggest float, then comparing as before
  if (convType1->isFloatingPoint() && convType2->isFloatingPoint()) {
    // take the biggest floating point
    if (convType1->toScalarLLVMType(fcmp->getContext())->getPrimitiveSizeInBits()
        > convType2->toScalarLLVMType(fcmp->getContext())->getPrimitiveSizeInBits()) {
      // t1 is "more precise"
      convType = *convType1;
    }
    else if (convType1->toScalarLLVMType(fcmp->getContext())->getPrimitiveSizeInBits()
             < convType2->toScalarLLVMType(fcmp->getContext())->getPrimitiveSizeInBits()) {
      // t2 is "more precise"
      convType = *convType2;
    }
    else {
      // they are equal, yeah!
      convType = *convType1; // or t2, they are equal
      // FIXME: what if bfloat16 (for now unsupported) and half???
    }
  }
  else if (convType1->isFloatingPoint())
    convType = *convType1;
  else if (convType2->isFloatingPoint())
    convType = *convType2;
  else
    llvm_unreachable("There should be at least one floating point");

  Value* newOperand1 = getConvertedOperand(operand1, convType, fcmp, ConvTypePolicy::ForceHint);
  Value* newOperand2 = getConvertedOperand(operand2, convType, fcmp, ConvTypePolicy::ForceHint);
  IRBuilder<NoFolder> builder(fcmp->getNextNode());
  auto* res = builder.CreateFCmp(fcmp->getPredicate(), newOperand1, newOperand2);
  setConversionResultInfo(res);
  return res;
}

Value* ConversionPass::convertCast(CastInst* cast) {
  // Instruction opcodes:
  // - [FPToSI,FPToUI,SIToFP,UIToFP] are handled here
  // - [Trunc,ZExt,SExt] are handled as a fallback case, not here
  // - [PtrToInt,IntToPtr,BitCast,AddrSpaceCast] might cause errors

  TransparentType* type = taffoInfo.getOrCreateTransparentType(*cast);
  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(cast);

  Value* operand = cast->getOperand(0);
  Value* newOperand = nullptr;
  auto iter = convertedValues.find(operand);
  if (iter != convertedValues.end())
    newOperand = iter->second;
  if (!newOperand || newOperand == operand) {
    LLVM_DEBUG(log().logln("Operand was not converted: conversion not needed", Logger::Yellow));
    setConversionResultInfo(cast);
    return cast;
  }

  if (valueConvInfo->isConversionDisabled())
    return unsupported;

  auto* convType = valueConvInfo->getNewOrOldType<ConversionScalarType>();

  IRBuilder<NoFolder> builder(cast->getNextNode());
  if (auto* bc = dyn_cast<BitCastInst>(cast)) {
    TransparentType* newType = convType->toTransparentType();
    Type* newLLVMType = newType->toLLVMType();
    if (newOperand)
      return builder.CreateBitCast(newOperand, newLLVMType);
    else
      return builder.CreateBitCast(operand, newLLVMType);
  }
  if (operand->getType()->isFloatingPointTy()) {
    /* fptosi, fptoui, fptrunc, fpext */
    if (cast->getOpcode() == Instruction::FPToSI)
      return getConvertedOperand(operand, ConversionScalarType(*type, true), cast, ConvTypePolicy::ForceHint);
    else if (cast->getOpcode() == Instruction::FPToUI)
      return getConvertedOperand(operand, ConversionScalarType(*type, false), cast, ConvTypePolicy::ForceHint);
    else if (cast->getOpcode() == Instruction::FPTrunc || cast->getOpcode() == Instruction::FPExt)
      return getConvertedOperand(operand, *convType, cast, ConvTypePolicy::ForceHint);
  }
  else {
    TransparentType* newOperandType = taffoInfo.getTransparentType(*newOperand);
    /* sitofp, uitofp */
    if (cast->getOpcode() == Instruction::SIToFP)
      return genConvertConvToConv(
        newOperand, ConversionScalarType(*newOperandType, true), *convType, ConvTypePolicy::ForceHint, cast);
    else if (cast->getOpcode() == Instruction::UIToFP)
      return genConvertConvToConv(
        newOperand, ConversionScalarType(*newOperandType, false), *convType, ConvTypePolicy::ForceHint, cast);
  }
  return unsupported;
}

/**
 * When an instruction couldn't be converted, fallback is performed instead:
 * - the instruction keeps its original type
 * - its operands get reconverted to the original type if they were already converted
 */
Value* ConversionPass::fallback(Instruction* inst) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger << "[" << __FUNCTION__ << "]\n";
    indenter.increaseIndent(););

  fallbackCount++;

  std::vector<Value*> newOperands;
  bool anyReplaced = false;
  for (Value* operand : inst->operands()) {
    Value* newOperand = operand;
    if (taffoConvInfo.hasValueConvInfo(operand)) {
      TransparentType* operandType = taffoInfo.getTransparentType(*operand);
      std::unique_ptr<ConversionType> operandConvType = ConversionTypeFactory::create(*operandType);
      newOperand = getConvertedOperand(operand, *operandConvType, inst, ConvTypePolicy::ForceHint);
      if (newOperand != operand)
        anyReplaced = true;
    }
    newOperands.push_back(newOperand);
  }

  Instruction* res;
  if (!inst->isTerminator()) {
    res = inst->clone();
    res->insertAfter(inst);
  }
  else
    res = inst;

  for (unsigned i = 0; i < res->getNumOperands(); i++)
    res->setOperand(i, newOperands[i]);
  LLVM_DEBUG(
    if (anyReplaced)
      logger << "operands replaced\n";
    else
      logger << "no operands needed to be replaced\n";);

  // res is just the original instruction or a clone of it, type does not change
  setConversionResultInfo(res, inst);
  return res;
}
