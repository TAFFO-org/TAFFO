#include "TaffoInitializerPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "AnnotationParser.hpp"
#include "TypeUtils.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
using namespace taffo;

#define DEBUG_TYPE "taffo-init"

/**
 * @brief Looks for the global starting point
 *
 * The global starting point is a function annotated with the @c __taffo_vra_starting_function string:
 * if present, the function is returned and removed from the parent module;
 * if this annotation is associated with a variable, a fatal error is thrown.
 *
 * @param[in,out] M The module to search
 * @return the function annotated as global starting point if present, nullptr otherwise
 */
Function *TaffoInitializer::findStartingPointFunctionGlobal(Module &M) {
  GlobalVariable *StartFuncGlob = nullptr;

  for (GlobalVariable &Global : M.globals()) {
    if (Global.getName() == "__taffo_vra_starting_function") {
      StartFuncGlob = &Global;
      break;
    }
  }
  if (!StartFuncGlob)
    return nullptr;

  Constant *Init = StartFuncGlob->getInitializer();
  ConstantExpr *ValCExpr = dyn_cast_or_null<ConstantExpr>(Init);
  if (!ValCExpr)
    report_fatal_error("__taffo_vra_starting_function not initialized to anything or initialized incorrectly!");

  Function *Res = nullptr;
  while (ValCExpr && ValCExpr->getOpcode() == Instruction::BitCast) {
    if (isa<Function>(ValCExpr->getOperand(0))) {
      Res = dyn_cast<Function>(ValCExpr->getOperand(0));
      break;
    }
    ValCExpr = dyn_cast<ConstantExpr>(ValCExpr->getOperand(0));
  }
  if (!ValCExpr || !Res)
    report_fatal_error("__taffo_vra_starting_function initialized incorrectly!");

  StartFuncGlob->eraseFromParent();
  TaffoInfo::getInstance().eraseValue(*StartFuncGlob);

  return Res;
}

/**
 * @brief Parses the the annotated global variables and functions
 *
 * Based on the value of @c functionAnnotation parameter, this function parses either
 * global variables (if @c false) or functions (if @c true). In the latter case,
 * TODO: describe what removeNoFloatTy does
 *
 * @param[in] m the module to search
 * @param[out] variables the map of annotated values and their metadata
 * @param[in] functionAnnotation the selector for the type of Value to parse
 */
void TaffoInitializer::readGlobalAnnotations(Module &m,
                                             ConvQueueType &variables,
                                             bool functionAnnotation) {
  GlobalVariable *globAnnos = m.getGlobalVariable("llvm.global.annotations");

  if (globAnnos != nullptr) {
    if (auto *annos = dyn_cast<ConstantArray>(globAnnos->getInitializer())) {
      for (unsigned i = 0, n = annos->getNumOperands(); i < n; i++) {
        if (auto *anno = dyn_cast<ConstantStruct>(annos->getOperand(i))) {
          /* Structure of the expression (ConstantStruct operand #0 is the expression):
           * [OpType] operand,
           * [BitCast] *function,
           * [GetElementPtr] *annotation,
           * [GetElementPtr] *filename,
           * [Int] source code line
           */
          if (auto *expr = dyn_cast<ConstantExpr>(anno->getOperand(0))) {
            if (expr->getOpcode() == Instruction::BitCast && (functionAnnotation ^ !isa<Function>(expr->getOperand(0)))) {
              parseAnnotation(variables, cast<ConstantExpr>(anno->getOperand(1)), expr->getOperand(0));
            }
          }
        }
      }
    }
  }
  if (functionAnnotation)
    removeNoFloatTy(variables);
}

/**
 * @brief Parses the annotated local variables inside a function
 *
 * If at least one variable is annotated as @c target, the whole function is set as starting point.
 *
 * @param[in] f the function to search
 * @param[out] variables the map of annotated variables and their metadata
 */
void TaffoInitializer::readLocalAnnotations(Function &f, ConvQueueType &annotatedValues) {
  bool found = false;
  for (Instruction &inst : instructions(f)) {
    if (auto *call = dyn_cast<CallInst>(&inst)) {
      if (!call->getCalledFunction())
        continue;
      if (call->getCalledFunction()->getName().starts_with("llvm.var.annotation")) {
        bool startingPoint = false;
        Value *annotatedValue = inst.getOperand(0);
        Value *annotationValue = inst.getOperand(1);
        parseAnnotation(annotatedValues, annotatedValue, annotationValue, &startingPoint);
        found |= startingPoint;
      }
    }
  }
  if (found) {
    TaffoInfo::getInstance().addStartingPoint(f);
  }
}

/**
 * @brief Parses the annotated local variables inside a module
 *
 * In order to avoid the dead code elimination pass to ignore the functions of the module,
 * the @c Attribute::OptimizeNone attributes are removed from all the functions
 *
 * @param m the module to search
 * @param res the map of annotated variables and their metadata
 */
void TaffoInitializer::readAllLocalAnnotations(Module &m, ConvQueueType &res) {
  for (Function &f : m.functions()) {
    ConvQueueType annotatedValues;
    readLocalAnnotations(f, annotatedValues);
    res.insert(annotatedValues.begin(), annotatedValues.end());

    /* Otherwise dce pass ignores the function (removed also where it's not required).
     * Don't remove for OCL trampolines because we want to keep the useless code there
     * deliberately. These trampolines will be removed by conversion later anyway. */
    if (!TaffoInfo::getInstance().isOpenCLTrampoline(f))
      f.removeFnAttr(Attribute::OptimizeNone);
  }

  if (!TaffoInfo::getInstance().hasStartingPoint(m))
    TaffoInfo::getInstance().addDefaultStartingPoint(m);
}

/**
 * @brief Parses the annotation associated with a Value (function or variable)
 *
 * Based on the subclass of the Value, the first field of the @c variables map is:
 *   - @b local variable: the register in which the variable is stored
 *   - @b function: the function's users (one entry per user)
 *   - @b global variable: the variable itself
 *
 * In the case of functions, they are also added to the @c enabledFunction list.
 *
 * @param[out] variables the map of annotated values and their metadata
 * @param[in] annoPtrInst the instruction that contains the annotation pointer
 * @param[in] instr pointer to the annotated Value
 * @param[out] startingPoint if the value is annotated as @c target
 *
 * @return @c true if the annotation is correctly parsed, @c false otherwise
 */
bool TaffoInitializer::parseAnnotation(
    ConvQueueType &annotatedValues, Value *annotatedValue, Value *annotationValue, bool *startingPoint) {
  auto *annotationContent = dyn_cast<GlobalVariable>(annotationValue);
  if (!annotationContent)
    return false;
  auto *annotationStrConstant = dyn_cast<ConstantDataSequential>(annotationContent->getInitializer());
  if (!annotationStrConstant || !annotationStrConstant->isString())
    return false;
  StringRef annotationStr = annotationStrConstant->getAsString();

  AnnotationParser parser;
  if (!parser.parseAnnotationString(annotationStr, getUnwrappedType(annotatedValue))) {
    errs() << "TAFFO Annotation parser error: \n"
           << "  In annotation: \"" << annotationStr << "\" of value " << *annotatedValue << "\n"
           << "  " << parser.lastError() << "\n";
    report_fatal_error("Error parsing annotation!");
    // TODO: use Error and propagate it (as soon as it is possible...)
    // return false;
  }

  ConvQueueInfo valueConvQueueInfo;
  valueConvQueueInfo.rootDistance = 0;

  if (!parser.backtracking)
    valueConvQueueInfo.backtrackingDepthLeft = 0;
  else
    valueConvQueueInfo.backtrackingDepthLeft = parser.backtrackingDepth;
  if (startingPoint)
    *startingPoint = parser.startingPoint;

  std::shared_ptr<ValueInfo> &valueInfo = valueConvQueueInfo.valueInfo;
  valueInfo = parser.valueInfo;

  if (auto *annotatedFun = dyn_cast<Function>(annotatedValue)) {
    enabledFunctions.insert(annotatedFun);
    for (auto user : annotatedFun->users()) {
      if (!isa<CallInst>(user) && !isa<InvokeInst>(user))
        continue;
      annotatedValues.insert({user, valueConvQueueInfo});
    }
  } else {
    annotatedValues.insert({annotatedValue, valueConvQueueInfo});
  }

  return true;
}

/**
 * Removes Values not having a float type from the map
 *
 * Instructions which are not @c alloca nor global variables, if present, are erased;
 * the remaining ones are checked and, if they don't allocate any kind of float variable,
 * are erased as well.
 *
 * @param[in,out] res the map of annotated values and their metadata
 */
void TaffoInitializer::removeNoFloatTy(ConvQueueType &res) {
  for (auto PIt : res) {
    Type *ty;
    Value *it = PIt.first;

    if (auto *alloca = dyn_cast<AllocaInst>(it)) {
      ty = alloca->getAllocatedType();
    } else if (auto *global = dyn_cast<GlobalVariable>(it)) {
      ty = global->getType();
    } else if (isa<CallInst>(it) || isa<InvokeInst>(it)) {
      ty = it->getType();
      if (ty->isVoidTy())
        continue;
    } else {
      LLVM_DEBUG(dbgs() << "annotated instruction " << *it << " not an alloca or a global, ignored\n");
      res.erase(it);
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
      LLVM_DEBUG(dbgs() << "annotated instruction " << *it << " does not allocate a"
                                                              " kind of float; ignored\n");
      res.erase(it);
    }
  }
}

/**
 * @brief Shows all the annotated objects (global and local variables, functions) in a module
 *
 * @param[in] m the module to show
 */
void TaffoInitializer::printAnnotatedObj(Module &m) {
  ConvQueueType res;

  readGlobalAnnotations(m, res, true);
  errs() << "Annotated Function:\n";
  if (!res.empty()) {
    for (const auto &it : res) {
      errs() << " -> " << *it.first << "\n";
    }
    errs() << "\n";
  }

  res.clear();
  readGlobalAnnotations(m, res);
  errs() << "Global Set:\n";
  if (!res.empty()) {
    for (const auto &it : res) {
      errs() << " -> " << *it.first << "\n";
    }
    errs() << "\n";
  }

  for (auto &f : m) {
    errs() << "Function ";
    errs().write_escaped(f.getName()) << ":\n";
    res.clear();
    readLocalAnnotations(f, res);
    if (!res.empty()) {
      errs() << " Local Set:\n";
      for (const auto &it : res) {
        errs() << "  -> " << *it.first << "\n";
      }
    }
  }
}
