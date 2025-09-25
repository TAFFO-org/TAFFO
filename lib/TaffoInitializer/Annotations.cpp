#include "AnnotationParser.hpp"
#include "Debug/Logger.hpp"
#include "InitializerPass.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TransparentType.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>

using namespace llvm;
using namespace tda;
using namespace taffo;

#define DEBUG_TYPE "taffo-init"

Function* InitializerPass::findStartingPointFunctionGlobal(Module& m) {
  GlobalVariable* startFunGlob = nullptr;

  for (GlobalVariable& global : m.globals()) {
    if (global.getName() == "__taffo_vra_starting_function") {
      startFunGlob = &global;
      break;
    }
  }
  if (!startFunGlob)
    return nullptr;

  Constant* initializer = startFunGlob->getInitializer();
  auto* initializerConstExpr = dyn_cast_or_null<ConstantExpr>(initializer);
  if (!initializerConstExpr)
    report_fatal_error("__taffo_vra_starting_function not initialized to anything or initialized incorrectly!");

  Function* startingPointFun = nullptr;
  while (initializerConstExpr && initializerConstExpr->getOpcode() == Instruction::BitCast) {
    if (isa<Function>(initializerConstExpr->getOperand(0))) {
      startingPointFun = dyn_cast<Function>(initializerConstExpr->getOperand(0));
      break;
    }
    initializerConstExpr = dyn_cast<ConstantExpr>(initializerConstExpr->getOperand(0));
  }
  if (!initializerConstExpr || !startingPointFun)
    report_fatal_error("__taffo_vra_starting_function initialized incorrectly!");

  taffoInfo.eraseValue(startFunGlob);

  return startingPointFun;
}

void InitializerPass::readAndRemoveGlobalAnnotations(Module& m) {
  if (GlobalVariable* annotationsGlobalVar = m.getGlobalVariable("llvm.global.annotations")) {
    if (auto* annotations = dyn_cast<ConstantArray>(annotationsGlobalVar->getInitializer()))
      for (unsigned i = 0, n = annotations->getNumOperands(); i < n; i++)
        if (auto* annotation = dyn_cast<ConstantStruct>(annotations->getOperand(i))) {
          parseAnnotation(annotation->getOperand(0), annotation->getOperand(1));
          taffoInfo.eraseValue(annotation);
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
  std::string annotationStr = annotationStrConstant->getAsString().str();
  if (annotationStr.back() == 0)
    annotationStr.pop_back();

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

  bool containsFloatingPoint = taffoInfo.getOrCreateTransparentType(*annotatedValue)->containsFloatingPointType();
  if (!containsFloatingPoint) {
    LLVM_DEBUG(
      Logger& logger = log();
      logger.log("[Value] ", Logger::Bold).logValueln(annotatedValue);
      auto indenter = logger.getIndenter();
      indenter.increaseIndent();
      logger.logln("disabling conversion because not a float", Logger::Yellow););
    // TODO manage disabling conversion of structs
    if (auto scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*annotatedValue)))
      scalarInfo->conversionEnabled = false;
    return;
  }

  // parseAnnotationAndGenValueInfo has generated the valueInfo: we need to generate also the valueInitInfo.
  // For functions, valueInfo needs to be copied from the function to its call sites,
  // and valueInitInfo is generated only for call sites, not for the function itself.
  if (auto* annotatedFun = dyn_cast<Function>(annotatedValue)) {
    annotatedFunctions.insert(annotatedFun);
    std::shared_ptr<ValueInfo> funInfo = taffoInfo.getValueInfo(*annotatedFun);
    for (User* user : annotatedFun->users()) {
      if (!isa<CallInst>(user) && !isa<InvokeInst>(user))
        continue;
      infoPropagationQueue.push_back(user);
      taffoInfo.setValueInfo(*user, funInfo);
      taffoInitInfo.createValueInitInfo(user, 0);
    }
  }
  else {
    infoPropagationQueue.push_back(annotatedValue);
    taffoInitInfo.createValueInitInfo(annotatedValue, 0);
  }
}
