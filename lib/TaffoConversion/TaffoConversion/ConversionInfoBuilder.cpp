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

#define DEBUG_TYPE "taffo-conversion"

void ConversionPass::buildGlobalConversionInfo(Module& m, SmallVectorImpl<Value*>& values) {
  LLVM_DEBUG(log().logln("[Building conversionInfo of global values]", Logger::Blue));
  for (GlobalVariable& gv : m.globals())
    if (taffoInfo.hasValueInfo(gv))
      buildConversionInfo(&values, taffoInfo.getValueInfo(gv), &gv);
}

void ConversionPass::buildLocalConversionInfo(Function& f, SmallVectorImpl<Value*>& values, bool argsOnly) {
  for (Argument& arg : f.args())
    if (taffoInfo.hasValueInfo(arg)) {
      // Don't enqueue function arguments because they will be handled by the function cloning step
      buildConversionInfo(nullptr, taffoInfo.getValueInfo(arg), &arg);
    }
  if (argsOnly)
    return;

  for (Instruction& inst : instructions(f))
    if (taffoInfo.hasValueInfo(inst))
      buildConversionInfo(&values, taffoInfo.getValueInfo(inst), &inst);
}

void ConversionPass::buildAllLocalConversionInfo(Module& m, SmallVectorImpl<Value*>& values) {
  LLVM_DEBUG(log().logln("[Building conversionInfo of local values]", Logger::Blue));
  for (Function& f : m.functions()) {
    bool argsOnly = false;
    if (TaffoInfo::getInstance().isTaffoCloneFunction(f)) {
      LLVM_DEBUG(log() << __FUNCTION__ << " skipping function body of " << f.getName() << " because it is cloned\n");
      functionPool[&f] = nullptr;
      argsOnly = true;
    }

    buildLocalConversionInfo(f, values, argsOnly);

    /* Otherwise dce pass ignore the function
     * (removed also where it's not required) */
    f.removeFnAttr(Attribute::OptimizeNone);
  }
}

bool ConversionPass::buildConversionInfo(SmallVectorImpl<Value*>* values,
                                         std::shared_ptr<ValueInfo> valueInfo,
                                         Value* value) {
  Logger& logger = log();
  auto indenter = logger.getIndenter();
  LLVM_DEBUG(
    logger.log("[Value] ", Logger::Bold).logValueln(value);
    indenter.increaseIndent(););

  if (hasConversionInfo(value)) {
    auto existing = getConversionInfo(value);
    if (existing->isArgumentPlaceholder) {
      LLVM_DEBUG(logger << "value is a placeholder: skipping\n");
      return false;
    }
  }

  ConversionInfo conversionInfo;
  conversionInfo.isBacktrackingNode = false;
  conversionInfo.fixpTypeRootDistance = 0;
  conversionInfo.origType = taffoInfo.getTransparentType(*value)->clone();

  if (std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo)) {
    if (!scalarInfo->isConversionEnabled()) {
      LLVM_DEBUG(logger << "conversion disabled: skipping\n");
      return false;
    }
    if (!value->getType()->isVoidTy()) {
      if (auto numericTypeInfo = scalarInfo->numericType)
        conversionInfo.fixpType = std::make_shared<FixedPointScalarType>(numericTypeInfo.get());
      else {
        LLVM_DEBUG(logger << Logger::Yellow << "numericType in valueInfo is null: ");
        if (isKnownConvertibleWithIncompleteMetadata(value)) {
          LLVM_DEBUG(log() << "since I like this instruction I'm going to give it the benefit of doubt\n"
                           << Logger::Reset);
          conversionInfo.fixpType = std::make_shared<FixedPointScalarType>();
        }
        else {
          LLVM_DEBUG(log() << "skipping\n"
                           << Logger::Reset);
          return false;
        }
      }
    }
    else
      conversionInfo.fixpType = std::make_shared<FixedPointScalarType>();
  }
  else if (std::shared_ptr<StructInfo> structInfo = std::dynamic_ptr_cast<StructInfo>(valueInfo)) {
    if (!value->getType()->isVoidTy()) {
      bool enableConversion = true;
      conversionInfo.fixpType = std::make_shared<FixedPointStructType>(structInfo, &enableConversion);
      if (!enableConversion) {
        LLVM_DEBUG(logger << "conversion disabled: skipping\n");
        return false;
      }
    }
    else
      conversionInfo.fixpType = std::make_shared<FixedPointScalarType>();
  }
  else
    llvm_unreachable("Unrecognized valueInfo");

  if (values && !is_contained(*values, value))
    values->push_back(value);
  *newConversionInfo(value) = conversionInfo;
  return true;
}

bool ConversionPass::isKnownConvertibleWithIncompleteMetadata(Value* value) {
  if (auto* inst = dyn_cast<Instruction>(value)) {
    auto* call = dyn_cast<CallBase>(inst);
    if (!call)
      return false;
    Function* fun = call->getCalledFunction();
    if (isSupportedOpenCLFunction(fun))
      return true;
    if (isSupportedCudaFunction(fun))
      return true;
  }
  return false;
}
