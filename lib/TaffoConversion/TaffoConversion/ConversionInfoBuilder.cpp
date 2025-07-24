#include "ConversionPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"
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
  for (GlobalVariable& gv : m.globals())
    if (taffoInfo.hasValueInfo(gv))
      buildConversionInfo(&values, taffoInfo.getValueInfo(gv), &gv);
}

void ConversionPass::buildLocalConversionInfo(Function& f, SmallVectorImpl<Value*>& values, bool argumentsOnly) {
  for (Argument& arg : f.args())
    if (taffoInfo.hasValueInfo(arg)) {
      // Don't enqueue function arguments because they will be handled by the function cloning step
      buildConversionInfo(nullptr, taffoInfo.getValueInfo(arg), &arg);
    }
  if (argumentsOnly)
    return;

  for (Instruction& inst : instructions(f))
    if (taffoInfo.hasValueInfo(inst))
      buildConversionInfo(&values, taffoInfo.getValueInfo(inst), &inst);
}

void ConversionPass::buildAllLocalConversionInfo(Module& m, SmallVectorImpl<Value*>& values) {
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
  if (hasConversionInfo(value)) {
    auto existing = getConversionInfo(value);
    if (existing->isArgumentPlaceholder) {
      LLVM_DEBUG(log() << "Skipping valueInfo of " << *value
                       << " because it's a placeholder and has fake valueInfo anyway\n");
      return false;
    }
  }

  LLVM_DEBUG(log().log("Collecting valueInfo of: ").logValueln(value));

  ConversionInfo conversionInfo;
  conversionInfo.isBacktrackingNode = false;
  conversionInfo.fixpTypeRootDistance = 0;
  conversionInfo.origType = taffoInfo.getTransparentType(*value)->clone();

  if (std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(valueInfo)) {
    if (!scalarInfo->isConversionEnabled())
      return false;
    if (!value->getType()->isVoidTy()) {
      if (auto numericTypeInfo = scalarInfo->numericType) {
        assert(!(getFullyUnwrappedType(value)->isStructTy()) && "input info / actual type mismatch");
        conversionInfo.fixpType = std::make_shared<FixedPointScalarType>(numericTypeInfo.get());
      }
      else {
        LLVM_DEBUG(log() << "numericType in valueInfo is null! ");
        if (isKnownConvertibleWithIncompleteMetadata(value)) {
          LLVM_DEBUG(log() << "Since I like this instruction I'm going to give it the benefit of doubt\n");
          conversionInfo.fixpType = std::make_shared<FixedPointScalarType>();
        }
        else {
          LLVM_DEBUG(log() << "Ignoring valueInfo for this instruction.\n");
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
      if (!enableConversion)
        return false;
    }
    else
      conversionInfo.fixpType = std::make_shared<FixedPointScalarType>();
  }
  else
    llvm_unreachable("Unrecognized valueInfo");

  if (values && !is_contained(*values, value))
    values->push_back(value);
  *newConversionInfo(value) = conversionInfo;

  LLVM_DEBUG(log() << "Conversion type: " << *conversionInfo.fixpType << "\n");
  return true;
}
