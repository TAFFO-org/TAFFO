#include "TaffoInitializerPass.hpp"

#include "AnnotationParser.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueInitInfo.hpp"
#include "Types/TransparentType.hpp"
#include "Types/TypeUtils.hpp"
#include "Debug/Logger.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

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
Function *TaffoInitializerPass::findStartingPointFunctionGlobal(Module &M) {
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
 * @param[in] functionAnnotation the selector for the type of Value to parse e quindi?
 */
void TaffoInitializerPass::readGlobalAnnotations(Module &m,
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
              parseAnnotation(cast<ConstantExpr>(anno->getOperand(1)), expr->getOperand(0));
            }
          }
        }
      }
    }
  }
  if (functionAnnotation)
    removeNoFloatTy();
}

/**
 * @brief Parses the annotated local variables inside a function
 *
 * If at least one variable is annotated as @c target, the whole function is set as starting point.
 *
 * @param[in] f the function to search
 * @param[out] variables the map of annotated variables and their metadata
 */
void TaffoInitializerPass::readAndRemoveLocalAnnotations(Function &f) {
  bool foundStartingPoint = false;
  for (Instruction &inst : make_early_inc_range(instructions(f))) {
    if (auto *call = dyn_cast<CallInst>(&inst)) {
      if (!call->getCalledFunction())
        continue;
      if (call->getCalledFunction()->getName().starts_with("llvm.var.annotation")) {
        bool isStartingPoint = false;
        Value *annotatedValue = inst.getOperand(0);
        Value *annotationValue = inst.getOperand(1);
        parseAnnotation(annotatedValue, annotationValue, &isStartingPoint);
        foundStartingPoint |= isStartingPoint;
        call->eraseFromParent();
      }
    }
  }
  if (foundStartingPoint)
    TaffoInfo::getInstance().addStartingPoint(f);
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
void TaffoInitializerPass::readAndRemoveLocalAnnotations(Module &m) {
  for (Function &f : m.functions()) {
    readAndRemoveLocalAnnotations(f);

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
 * @param[out] isStartingPoint if the value is annotated as @c target
 *
 * @return @c true if the annotation is correctly parsed, @c false otherwise
 */
void TaffoInitializerPass::parseAnnotation(Value *annotatedValue, Value *annotationValue, bool *isStartingPoint) {
  auto *annotationContent = cast<GlobalVariable>(annotationValue);
  auto *annotationStrConstant = cast<ConstantDataSequential>(annotationContent->getInitializer());
  StringRef annotationStr = annotationStrConstant->getAsString();

  AnnotationParser parser;
  if (!parser.parseAnnotationAndGenValueInfo(annotationStr, annotatedValue)) {
    Logger &logger = Logger::getInstance();
    logger.logln("TAFFO Annotation parser error:", raw_ostream::Colors::RED);
    logger.increaseIndent();
    logger.log("In annotation: \"", raw_ostream::Colors::RED);
    logger.log(annotationStr, raw_ostream::Colors::RED);
    logger.log("\" of value ", raw_ostream::Colors::RED);
    logger.logln(annotatedValue, raw_ostream::Colors::RED);
    logger.logln(parser.getLastError(), raw_ostream::Colors::RED);
    logger.decreaseIndent();
    llvm_unreachable("Error parsing annotation!");
  }
  



  if (isStartingPoint)
    *isStartingPoint = parser.startingPoint;

  // parseAnnotationAndGenValueInfo has generated the taffoInfo we need to generate also the taffoInitInfo.
  // For the functions taffoInitInfo is generated only on the callsite
  if (auto *annotatedFun = dyn_cast<Function>(annotatedValue))
    for (User *user : annotatedFun->users()) {
      if (!isa<CallInst>(user) && !isa<InvokeInst>(user))
        continue;
      infoPropagationQueue.push_back(user);
      taffoInitInfo.createValueInitInfo(user, 0, parser.backtracking ? parser.backtrackingDepth : 0);
    }
  else {
    infoPropagationQueue.push_back(annotatedValue);
    taffoInitInfo.createValueInitInfo(annotationValue, 0, parser.backtracking ? parser.backtrackingDepth : 0);
  }
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
void TaffoInitializerPass::removeNoFloatTy() {
  for (auto val : make_early_inc_range(infoPropagationQueue)) {
    bool containsFloatinPoint = TaffoInfo::getInstance().getTransparentType(*val)->containsFloatigPointType();
    if( !containsFloatinPoint){
    LLVM_DEBUG(
      auto& log = Logger::getInstance();
      log.log("Removing ",llvm::raw_ostream::Colors::YELLOW);
      log.log(val,llvm::raw_ostream::Colors::YELLOW);
      log.logln(" from infoPropagationQueue", raw_ostream::Colors::YELLOW);
    );
    infoPropagationQueue.remove(val);
    }
  }
}

/**
 * @brief Shows all the annotated objects (global and local variables, functions) in a module
 *
 * @param[in] m the module to show
 */
void TaffoInitializerPass::printAnnotatedObj(Module &m) {
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
