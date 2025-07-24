#include "ConversionPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TypeUtils.hpp"
#include "Utils/PtrCasts.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <iostream>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-conversion"

void FloatToFixed::readGlobalMetadata(Module& m, SmallVectorImpl<Value*>& variables, bool functionAnnotation) {
  auto& taffoInfo = TaffoInfo::getInstance();
  for (GlobalVariable& gv : m.globals())
    if (taffoInfo.hasValueInfo(gv))
      parseMetaData(&variables, taffoInfo.getValueInfo(gv), &gv);

  if (functionAnnotation)
    removeNoFloatTy(variables);
}

void FloatToFixed::readLocalMetadata(Function& f, SmallVectorImpl<Value*>& variables, bool argumentsOnly) {
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  for (Argument& arg : f.args()) {
    if (taffoInfo.hasValueInfo(arg)) {
      /* Don't enqueue function arguments because they will be handled by
       * the function cloning step */
      parseMetaData(nullptr, taffoInfo.getValueInfo(arg), &arg);
    }
  }

  if (argumentsOnly)
    return;

  for (Instruction& inst : instructions(f))
    if (taffoInfo.hasValueInfo(inst))
      parseMetaData(&variables, taffoInfo.getValueInfo(inst), &inst);
}

void FloatToFixed::readAllLocalMetadata(Module& m, SmallVectorImpl<Value*>& res) {
  for (Function& f : m.functions()) {
    bool argsOnly = false;
    if (TaffoInfo::getInstance().isTaffoCloneFunction(f)) {
      LLVM_DEBUG(log() << __FUNCTION__ << " skipping function body of " << f.getName() << " because it is cloned\n");
      functionPool[&f] = nullptr;
      argsOnly = true;
    }

    readLocalMetadata(f, res, argsOnly);

    /* Otherwise dce pass ignore the function
     * (removed also where it's not required) */
    f.removeFnAttr(Attribute::OptimizeNone);
  }
}

bool FloatToFixed::parseMetaData(SmallVectorImpl<Value*>* variables, std::shared_ptr<ValueInfo> raw, Value* instr) {
  if (hasConversionInfo(instr)) {
    auto existing = getConversionInfo(instr);
    if (existing->isArgumentPlaceholder) {
      LLVM_DEBUG(log() << "Skipping MD collection for " << *instr
                       << " because it's a placeholder and has fake valueInfo anyway\n");
      return false;
    }
  }

  LLVM_DEBUG(log().log("Collecting valueInfo for: ").logValueln(instr));

  ConversionInfo vi;
  auto& taffoInfo = TaffoInfo::getInstance();
  vi.isBacktrackingNode = false;
  vi.fixpTypeRootDistance = 0;
  vi.origType = taffoInfo.getOrCreateTransparentType(*instr)->clone();

  if (std::shared_ptr<ScalarInfo> fpInfo = std::dynamic_ptr_cast<ScalarInfo>(raw)) {
    if (!fpInfo->isConversionEnabled())
      return false;
    if (!instr->getType()->isVoidTy()) {
      NumericTypeInfo* fpt = dyn_cast_or_null<NumericTypeInfo>(fpInfo->numericType.get());
      if (!fpt) {
        LLVM_DEBUG(log() << "Type in valueInfo is null! ");
        if (isKnownConvertibleWithIncompleteMetadata(instr)) {
          LLVM_DEBUG(log() << "Since I like this instruction I'm going to give it the benefit of the doubt.\n");
          vi.fixpType = std::make_shared<FixedPointScalarType>();
        }
        else {
          LLVM_DEBUG(log() << "Ignoring valueInfo for this instruction.\n");
          return false;
        }
      }
      else {
        assert(!(getFullyUnwrappedType(instr)->isStructTy()) && "input info / actual type mismatch");
        vi.fixpType = std::make_shared<FixedPointScalarType>(fpt);
      }
    }
    else {
      vi.fixpType = std::make_shared<FixedPointScalarType>();
    }
  }
  else if (std::shared_ptr<StructInfo> fpInfo = std::dynamic_ptr_cast<StructInfo>(raw)) {
    if (!instr->getType()->isVoidTy()) {
      assert(getFullyUnwrappedType(instr)->isStructTy() && "input info / actual type mismatch");
      int enableConversion = 0;
      vi.fixpType = std::make_shared<FixedPointStructType>(fpInfo, &enableConversion);
      if (enableConversion == 0) {
        LLVM_DEBUG(log() << "Conversion not enabled.\n");
        return false;
      }
    }
    else {
      vi.fixpType = std::make_shared<FixedPointScalarType>();
    }
  }
  else {
    assert(false && "MDInfo type unrecognized");
  }

  if (variables && !is_contained(*variables, instr))
    variables->push_back(instr);
  *newConversionInfo(instr) = vi;

  LLVM_DEBUG(log() << "Type deducted: " << *vi.fixpType << "\n");

  return true;
}

void FloatToFixed::removeNoFloatTy(SmallVectorImpl<Value*>& res) {
  for (auto it = res.begin(); it != res.end();) {
    Type* ty;

    AllocaInst* alloca;
    GlobalVariable* global;
    if ((alloca = dyn_cast<AllocaInst>(*it))) {
      ty = alloca->getAllocatedType();
    }
    else if ((global = dyn_cast<GlobalVariable>(*it))) {
      ty = global->getType();
    }
    else {
      LLVM_DEBUG(log() << "annotated instruction " << *it << " not an alloca or a global, ignored\n");
      it = res.erase(it);
      continue;
    }

    while (ty->isArrayTy() || ty->isPointerTy()) {
      // TODO FIX SOON!
      /*if (ty->isPointerTy())
        ty = ty->getPointerElementType();
      else*/
      ty = ty->getArrayElementType();
    }
    if (!ty->isFloatingPointTy()) {
      LLVM_DEBUG(log() << "annotated instruction " << *it << " does not allocate a kind of float; ignored\n");
      it = res.erase(it);
    }
    else
      it++;
  }
}
