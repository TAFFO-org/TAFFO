#include "ConversionPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Utils/PtrCasts.hpp"

#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <iostream>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conv"

void ConversionPass::buildGlobalConvInfo(Module& m, SmallVectorImpl<Value*>& values) {
  LLVM_DEBUG(log().logln("[Building conversionInfo of global values]", Logger::Blue));
  for (GlobalVariable& gv : m.globals())
    buildConvInfo(&values, &gv);
}

void ConversionPass::buildLocalConvInfo(Function& f, SmallVectorImpl<Value*>& values, bool argsOnly) {
  if (!taffoConvInfo.hasValueConvInfo(&f))
    buildConvInfo(nullptr, &f); // No need to enqueue function

  for (Argument& arg : f.args())
    if (!taffoConvInfo.hasValueConvInfo(&arg) && !arg.getType()->isMetadataTy()) {
      // Don't enqueue function arguments because they will be handled by the function cloning step
      buildConvInfo(nullptr, &arg);
    }

  if (argsOnly)
    return;

  for (Instruction& inst : instructions(f))
    buildConvInfo(&values, &inst);
}

void ConversionPass::buildAllLocalConvInfo(Module& m, SmallVectorImpl<Value*>& values) {
  LLVM_DEBUG(log().logln("[Building conversionInfo of local values]", Logger::Blue));
  for (Function& f : m.functions()) {
    bool argsOnly = false;
    if (taffoInfo.isCloneFunction(f)) {
      LLVM_DEBUG(log() << __FUNCTION__ << " skipping function body of " << f.getName() << " because it is cloned\n");
      functionPool[&f] = nullptr;
      argsOnly = true;
    }

    buildLocalConvInfo(f, values, argsOnly);

    /* Otherwise dce pass ignore the function
     * (removed also where it's not required) */
    f.removeFnAttr(Attribute::OptimizeNone);
  }
}

bool ConversionPass::buildConvInfo(SmallVectorImpl<Value*>* convQueue, Value* value) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger.log("[Value] ", Logger::Bold).logValueln(value);
    indenter.increaseIndent(););

  if (taffoConvInfo.hasValueConvInfo(value)) {
    auto existing = taffoConvInfo.getValueConvInfo(value);
    if (existing->isArgumentPlaceholder) {
      LLVM_DEBUG(logger << "value is an argument placeholder: skipping\n");
      return false;
    }
  }

  ValueConvInfo* valueConvInfo = taffoConvInfo.createValueConvInfo(value);

  if (!taffoInfo.hasValueInfo(*value))
    return false;
  std::shared_ptr<ValueInfo> valueInfo = taffoInfo.getValueInfo(*value);
  TransparentType* type = taffoInfo.getTransparentType(*value);

  if (std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo)) {
    if (!scalarInfo->isConversionEnabled() && !isAlwaysConvertible(value)) {
      LLVM_DEBUG(
        logger << "conversion disabled: skipping\n";
        logger.log("new conversionInfo: ").logln(*valueConvInfo, Logger::Cyan););
      return false;
    }
    if (taffoInfo.getTransparentType(*value)->containsFloatingPointType()) {
      if (auto numericTypeInfo = scalarInfo->numericType)
        valueConvInfo->setNewType(std::make_unique<ConversionScalarType>(*type, numericTypeInfo.get()));
      else {
        LLVM_DEBUG(logger << Logger::Yellow << "numericType in valueInfo is null: ");
        if (isAlwaysConvertible(value)) {
          LLVM_DEBUG(logger << "converting anyway because this instruction is always convertible\n"
                            << Logger::Reset);
        }
        else {
          LLVM_DEBUG(
            logger << "skipping\n" << Logger::Reset;
            logger.log("new conversionInfo: ").logln(*valueConvInfo, Logger::Cyan););
          return false;
        }
      }
    }
  }
  else if (std::shared_ptr<StructInfo> structInfo = std::dynamic_ptr_cast<StructInfo>(valueInfo)) {
    if (!value->getType()->isVoidTy()) {
      auto* structType = cast<TransparentStructType>(type);
      bool conversionEnabled;
      auto newConvType = std::make_unique<ConversionStructType>(*structType, structInfo, &conversionEnabled);
      if (!conversionEnabled && !isAlwaysConvertible(value)) {
        LLVM_DEBUG(
          logger << "conversion disabled: skipping\n";
          logger.log("new conversionInfo: ").logln(*valueConvInfo, Logger::Cyan););
        return false;
      }
      valueConvInfo->setNewType(std::move(newConvType));
    }
  }
  else
    llvm_unreachable("Unrecognized valueInfo");

  valueConvInfo->enableConversion();
  if (convQueue && !is_contained(*convQueue, value))
    convQueue->push_back(value);
  LLVM_DEBUG(logger.log("new conversionInfo: ").logln(*valueConvInfo, Logger::Cyan));
  return true;
}

bool ConversionPass::isAlwaysConvertible(Value* value) {
  if (isa<LoadInst>(value) || isa<StoreInst>(value) || isa<GetElementPtrInst>(value))
    return true; // Load, store and gep convType depends only on the pointer operand
  if (auto* call = dyn_cast<CallBase>(value)) {
    Function* fun = call->getCalledFunction();
    if (isSupportedOpenCLFunction(fun))
      return true;
    if (isSupportedCudaFunction(fun))
      return true;
  }
  return false;
}
