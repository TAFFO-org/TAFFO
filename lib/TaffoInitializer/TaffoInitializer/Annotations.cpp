#include "AnnotationParser.hpp"
#include "Debug/Logger.hpp"
#include "InitializerPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TransparentType.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
using namespace taffo;

#define DEBUG_TYPE "taffo-init"

Function* InitializerPass::findStartingPointFunctionGlobal(Module& m) {
  GlobalVariable* startFunGlob = nullptr;

  for (GlobalVariable& Global : m.globals()) {
    if (Global.getName() == "__taffo_vra_starting_function") {
      startFunGlob = &Global;
      break;
    }
  }
  if (!startFunGlob)
    return nullptr;

  Constant* Init = startFunGlob->getInitializer();
  ConstantExpr* ValCExpr = dyn_cast_or_null<ConstantExpr>(Init);
  if (!ValCExpr)
    report_fatal_error("__taffo_vra_starting_function not initialized to anything or initialized incorrectly!");

  Function* startingPointFun = nullptr;
  while (ValCExpr && ValCExpr->getOpcode() == Instruction::BitCast) {
    if (isa<Function>(ValCExpr->getOperand(0))) {
      startingPointFun = dyn_cast<Function>(ValCExpr->getOperand(0));
      break;
    }
    ValCExpr = dyn_cast<ConstantExpr>(ValCExpr->getOperand(0));
  }
  if (!ValCExpr || !startingPointFun)
    report_fatal_error("__taffo_vra_starting_function initialized incorrectly!");

  taffoInfo.eraseValue(startFunGlob);

  return startingPointFun;
}

void InitializerPass::readAndRemoveGlobalAnnotations(Module& m) {
  if (GlobalVariable* annotationsGlobalVar = m.getGlobalVariable("llvm.global.annotations")) {
    if (auto* annotations = dyn_cast<ConstantArray>(annotationsGlobalVar->getInitializer()))
      for (unsigned i = 0, n = annotations->getNumOperands(); i < n; i++)
        if (auto* annotation = dyn_cast<ConstantStruct>(annotations->getOperand(i))) {
          // Structure of the expression (ConstantStruct operand #0 is the expression):
          // [OpType] operand,
          // [BitCast] *function,
          // [GetElementPtr] *annotation,
          // [GetElementPtr] *filename,
          // [Int] source code line
          if (auto* expr = dyn_cast<ConstantExpr>(annotation->getOperand(0)))
            if (expr->getOpcode() == Instruction::BitCast)
              parseAnnotation(cast<ConstantExpr>(annotation->getOperand(1)), expr->getOperand(0));
        }
    taffoInfo.eraseValue(annotationsGlobalVar);
  }
}

void InitializerPass::readAndRemoveLocalAnnotations(Function& f) {
  bool foundStartingPoint = false;
  for (Instruction& inst : make_early_inc_range(instructions(f))) {
    if (auto* call = dyn_cast<CallInst>(&inst)) {
      if (!call->getCalledFunction())
        continue;
      if (call->getCalledFunction()->getName().starts_with("llvm.var.annotation")) {
        bool isStartingPoint = false;
        Value* annotatedValue = inst.getOperand(0);
        Value* annotationValue = inst.getOperand(1);
        parseAnnotation(annotatedValue, annotationValue, &isStartingPoint);
        foundStartingPoint |= isStartingPoint;
        taffoInfo.eraseValue(call);
      }
    }
  }
  if (foundStartingPoint)
    taffoInfo.addStartingPoint(f);
}

void InitializerPass::readAndRemoveLocalAnnotations(Module& m) {
  for (Function& f : m.functions()) {
    readAndRemoveLocalAnnotations(f);
    /* Otherwise dce pass ignores the function (removed also where it's not required).
     * Don't remove for OCL trampolines because we want to keep the useless code there
     * deliberately. These trampolines will be removed by conversion later anyway. */
    if (!taffoInfo.isOpenCLTrampoline(f))
      f.removeFnAttr(Attribute::OptimizeNone);
  }

  if (!taffoInfo.hasStartingPoint(m))
    taffoInfo.addDefaultStartingPoint(m);
}

void InitializerPass::parseAnnotation(Value* annotatedValue, Value* annotationValue, bool* isStartingPoint) {
  auto* annotationContent = cast<GlobalVariable>(annotationValue);
  auto* annotationStrConstant = cast<ConstantDataSequential>(annotationContent->getInitializer());
  StringRef annotationStr = annotationStrConstant->getAsString();

  AnnotationParser parser;
  if (!parser.parseAnnotationAndGenValueInfo(annotationStr, annotatedValue)) {
    Logger& logger = log();
    logger << Logger::Red << "TAFFO Annotation parser error:\n";
    auto indenter = logger.getIndenter();
    indenter.increaseIndent();
    logger << "In annotation: \"" << annotationStr << "\"\nof value ";
    logger.logValueln(annotatedValue);
    logger.logln(parser.getLastError());
    llvm_unreachable("Error parsing annotation!");
  }

  if (isStartingPoint)
    *isStartingPoint = parser.startingPoint;

  // parseAnnotationAndGenValueInfo has generated the valueInfo: we need to generate also the valueInitInfo.
  // For functions, valueInitInfo is generated only for call sites, not for the function itself.
  if (auto* annotatedFun = dyn_cast<Function>(annotatedValue)) {
    annotatedFunctions.insert(annotatedFun);
    for (User* user : annotatedFun->users()) {
      if (!isa<CallInst>(user) && !isa<InvokeInst>(user))
        continue;
      infoPropagationQueue.push_back(user);
      taffoInitInfo.createValueInitInfo(user, 0, parser.backtracking ? parser.backtrackingDepth : 0);
    }
  }
  else {
    infoPropagationQueue.push_back(annotatedValue);
    taffoInitInfo.createValueInitInfo(annotatedValue, 0, parser.backtracking ? parser.backtrackingDepth : 0);
  }
}

void InitializerPass::removeNotFloats() {
  for (auto val : make_early_inc_range(infoPropagationQueue)) {
    bool containsFloatingPoint = taffoInfo.getOrCreateTransparentType(*val)->containsFloatingPointType();
    if (!containsFloatingPoint) {
      LLVM_DEBUG(
        Logger& logger = log();
        logger.log("Removing ", Logger::Yellow);
        logger.log(val, Logger::Yellow);
        logger.logln(" from infoPropagationQueue as it is not float", Logger::Yellow););
      infoPropagationQueue.remove(val);
    }
  }
}
