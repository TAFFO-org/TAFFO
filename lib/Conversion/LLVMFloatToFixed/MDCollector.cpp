#include "LLVMFloatToFixedPass.h"
#include "Metadata.h"
#include "TypeUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <sstream>


using namespace llvm;
using namespace flttofix;
using namespace mdutils;
using namespace taffo;


void FloatToFixed::readGlobalMetadata(Module &m, SmallPtrSetImpl<Value *> &variables, bool functionAnnotation)
{
  MetadataManager &MDManager = MetadataManager::getMetadataManager();

  for (GlobalVariable &gv : m.globals()) {
    MDInfo *MDI = MDManager.retrieveMDInfo(&gv);
    if (MDI) {
      parseMetaData(&variables, MDI, &gv);
    }
  }
  if (functionAnnotation)
    removeNoFloatTy(variables);
}

// No, I was not mad at all
InputInfo *FloatToFixed::getInputInfo(Value *v)
{
  MetadataManager &MDManager = MetadataManager::getMetadataManager();
  MDInfo *MDI = MDManager.retrieveMDInfo(v);

  if (auto *fpInfo = dyn_cast_or_null<InputInfo>(MDI)) {
    return fpInfo;
  }

  return nullptr;
}


void FloatToFixed::readLocalMetadata(Function &f, SmallPtrSetImpl<Value *> &variables, bool argumentsOnly)
{
  MetadataManager &MDManager = MetadataManager::getMetadataManager();

  SmallVector<mdutils::MDInfo *, 5> argsII;
  MDManager.retrieveArgumentInputInfo(f, argsII);
  auto arg = f.arg_begin();
  for (auto itII = argsII.begin(); itII != argsII.end(); itII++) {
    if (*itII != nullptr) {
      /* Don't enqueue function arguments because they will be handled by
       * the function cloning step */
      parseMetaData(nullptr, *itII, arg);
    }
    arg++;
  }

  if (argumentsOnly)
    return;

  for (inst_iterator iIt = inst_begin(&f), iItEnd = inst_end(&f); iIt != iItEnd; iIt++) {
    MDInfo *MDI = MDManager.retrieveMDInfo(&(*iIt));
    if (MDI) {
      parseMetaData(&variables, MDI, &(*iIt));
    }
  }
}


void FloatToFixed::readAllLocalMetadata(Module &m, SmallPtrSetImpl<Value *> &res)
{
  for (Function &f : m.functions()) {
    bool argsOnly = false;
    if (f.getMetadata(SOURCE_FUN_METADATA)) {
      LLVM_DEBUG(dbgs() << __FUNCTION__ << " skipping function body of " << f.getName()
                        << " because it is cloned\n");
      functionPool[&f] = nullptr;
      argsOnly = true;
    }

    SmallPtrSet<Value *, 32> t;
    readLocalMetadata(f, t, argsOnly);
    res.insert(t.begin(), t.end());

    /* Otherwise dce pass ignore the function
     * (removed also where it's not required) */
    f.removeFnAttr(Attribute::OptimizeNone);
  }
}


bool FloatToFixed::parseMetaData(SmallPtrSetImpl<Value *> *variables, MDInfo *raw, Value *instr)
{
  LLVM_DEBUG(dbgs() << "Collecting metadata for:";);
  LLVM_DEBUG(instr->print(dbgs()););
  LLVM_DEBUG(dbgs() << "\n";);


  ValueInfo vi;

  vi.isBacktrackingNode = false;
  vi.fixpTypeRootDistance = 0;
  vi.origType = instr->getType();

  if (InputInfo *fpInfo = dyn_cast<InputInfo>(raw)) {
    if (!fpInfo->IEnableConversion)
      return false;
    if (!instr->getType()->isVoidTy()) {
      assert(!(fullyUnwrapPointerOrArrayType(instr->getType())->isStructTy()) &&
             "input info / actual type mismatch");
      TType *fpt = dyn_cast_or_null<TType>(fpInfo->IType.get());
      if (!fpt) {
        LLVM_DEBUG(dbgs() << "Failed to get Metadata.\n");
        return false;
      }
      vi.fixpType = FixedPointType(fpt);
    } else {
      vi.fixpType = FixedPointType();
    }

  } else if (StructInfo *fpInfo = dyn_cast<StructInfo>(raw)) {
    if (!instr->getType()->isVoidTy()) {
      assert(fullyUnwrapPointerOrArrayType(instr->getType())->isStructTy() &&
             "input info / actual type mismatch");
      int enableConversion = 0;
      vi.fixpType = FixedPointType::get(fpInfo, &enableConversion);
      if (enableConversion == 0) {
        LLVM_DEBUG(dbgs() << "Conversion not enabled.\n");
        return false;
      }
    } else {
      vi.fixpType = FixedPointType();
    }

  } else {
    assert(false && "MDInfo type unrecognized");
  }

  if (variables)
    variables->insert(instr);
  *newValueInfo(instr) = vi;

  LLVM_DEBUG(dbgs() << "Type deducted: " << vi.fixpType.toString() << "\n");

  return true;
}


void FloatToFixed::removeNoFloatTy(SmallPtrSetImpl<Value *> &res)
{
  for (auto it : res) {
    Type *ty;

    AllocaInst *alloca;
    GlobalVariable *global;
    if ((alloca = dyn_cast<AllocaInst>(it))) {
      ty = alloca->getAllocatedType();
    } else if ((global = dyn_cast<GlobalVariable>(it))) {
      ty = global->getType();
    } else {
      LLVM_DEBUG(dbgs() << "annotated instruction " << *it << " not an alloca or a global, ignored\n");
      res.erase(it);
      continue;
    }

    while (ty->isArrayTy() || ty->isPointerTy()) {
      if (ty->isPointerTy())
        ty = ty->getPointerElementType();
      else
        ty = ty->getArrayElementType();
    }
    if (!ty->isFloatingPointTy()) {
      LLVM_DEBUG(dbgs() << "annotated instruction " << *it << " does not allocate a"
                                                              " kind of float; ignored\n");
      res.erase(it);
    }
  }
}
