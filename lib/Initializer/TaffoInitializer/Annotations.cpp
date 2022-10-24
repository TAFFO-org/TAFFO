#include "AnnotationParser.h"
#include "Metadata.h"
#include "TaffoInitializerPass.h"
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
Function *TaffoInitializer::findStartingPointFunctionGlobal(Module &M)
{
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
                                             MultiValueMap<Value *, ValueInfo> &variables,
                                             bool functionAnnotation)
{
  GlobalVariable *globAnnos = m.getGlobalVariable("llvm.global.annotations");

  if (globAnnos != NULL) {
    if (ConstantArray *annos = dyn_cast<ConstantArray>(globAnnos->getInitializer())) {
      for (unsigned i = 0, n = annos->getNumOperands(); i < n; i++) {
        if (ConstantStruct *anno = dyn_cast<ConstantStruct>(annos->getOperand(i))) {
          /* Structure of the expression (ConstantStruct operand #0 is the expression):
           * [OpType] operand,
           * [BitCast] *function,
           * [GetElementPtr] *annotation,
           * [GetElementPtr] *filename,
           * [Int] source code line
           */
          if (ConstantExpr *expr = dyn_cast<ConstantExpr>(anno->getOperand(0))) {
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
void TaffoInitializer::readLocalAnnotations(llvm::Function &f, MultiValueMap<Value *, ValueInfo> &variables)
{
  bool found = false;
  for (inst_iterator iIt = inst_begin(&f), iItEnd = inst_end(&f); iIt != iItEnd; iIt++) {
    if (CallInst *call = dyn_cast<CallInst>(&(*iIt))) {
      if (!call->getCalledFunction())
        continue;

      if (call->getCalledFunction()->getName() == "llvm.var.annotation") {
        bool startingPoint = false;
        parseAnnotation(variables, cast<ConstantExpr>(iIt->getOperand(1)), iIt->getOperand(0), &startingPoint);
        found |= startingPoint;
      }
    }
  }
  if (found) {
    mdutils::MetadataManager::setStartingPoint(f);
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
void TaffoInitializer::readAllLocalAnnotations(llvm::Module &m, MultiValueMap<Value *, ValueInfo> &res)
{
  for (Function &f : m.functions()) {
    MultiValueMap<Value *, ValueInfo> t;
    readLocalAnnotations(f, t);
    res.insert(res.end(), t.begin(), t.end());

    /* Otherwise dce pass ignores the function (removed also where it's not required).
     *   Don't remove for OCL trampolines because we want to keep the useless code there
     * deliberately. These trampolines will be removed by conversion later anyway. */
    if (!f.hasMetadata(INIT_OCL_TRAMPOLINE_METADATA)) {
      f.removeFnAttr(Attribute::OptimizeNone);
    }
  }
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
bool TaffoInitializer::parseAnnotation(MultiValueMap<Value *, ValueInfo> &variables,
                                       ConstantExpr *annoPtrInst, Value *instr,
                                       bool *startingPoint)
{
  ValueInfo vi;

  if (annoPtrInst->getOpcode() != Instruction::GetElementPtr)
    return false;
  auto *annoContent = dyn_cast<GlobalVariable>(annoPtrInst->getOperand(0));
  if (!annoContent)
    return false;
  auto *annoStr = dyn_cast<ConstantDataSequential>(annoContent->getInitializer());
  if (!annoStr)
    return false;
  if (!(annoStr->isString()))
    return false;

  StringRef annstr = annoStr->getAsString();
  AnnotationParser parser;
  if (!parser.parseAnnotationString(annstr)) {
    errs() << "TAFFO annotation parser syntax error: \n";
    errs() << "  In annotation: \"" << annstr << "\"\n";
    errs() << "  " << parser.lastError() << "\n";
    return false;
  }
  Type *TyCheck = instr->getType();
  if (Instruction *I = dyn_cast<Instruction>(instr))
    TyCheck = I->getOperand(0)->getType();
  else if (Function *F = dyn_cast<Function>(instr))
    TyCheck = F->getReturnType();
  if (!typecheckMetadata(TyCheck, parser.metadata.get())) {
    errs() << "TAFFO typechecker error:\n";
    errs() << "  In annotation: \"" << annstr << "\"\n";
    errs() << "  Type does not look like LLVM type " << *TyCheck << "\n";
    return false;
  }
  vi.fixpTypeRootDistance = 0;
  if (!parser.backtracking)
    vi.backtrackingDepthLeft = 0;
  else
    vi.backtrackingDepthLeft = parser.backtrackingDepth;
  vi.metadata = parser.metadata;
  if (startingPoint)
    *startingPoint = parser.startingPoint;
  vi.target = parser.target;
  vi.bufferID = parser.bufferID;

  if (auto *toconv = dyn_cast<Instruction>(instr)) {
    variables.push_back(toconv->getOperand(0), vi);
  } else if (auto *fun = dyn_cast<Function>(instr)) {
    enabledFunctions.insert(fun);
    for (auto user : fun->users()) {
      if (!(isa<CallInst>(user) || isa<InvokeInst>(user)))
        continue;
      variables.push_back(user, vi);
    }
  } else {
    variables.push_back(instr, vi);
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
void TaffoInitializer::removeNoFloatTy(MultiValueMap<Value *, ValueInfo> &res)
{
  for (auto PIt : res) {
    Type *ty;
    Value *it = PIt->first;

    if (AllocaInst *alloca = dyn_cast<AllocaInst>(it)) {
      ty = alloca->getAllocatedType();
    } else if (GlobalVariable *global = dyn_cast<GlobalVariable>(it)) {
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


/**
 * @brief Shows all the annotated objects (global and local variables, functions) in a module
 *
 * @param[in] m the module to show
 */
void TaffoInitializer::printAnnotatedObj(Module &m)
{
  MultiValueMap<Value *, ValueInfo> res;

  readGlobalAnnotations(m, res, true);
  errs() << "Annotated Function: \n";
  if (!res.empty()) {
    for (auto it : res) {
      errs() << " -> " << *it->first << "\n";
    }
    errs() << "\n";
  }

  res.clear();
  readGlobalAnnotations(m, res);
  errs() << "Global Set: \n";
  if (!res.empty()) {
    for (auto it : res) {
      errs() << " -> " << *it->first << "\n";
    }
    errs() << "\n";
  }

  for (auto fIt = m.begin(), fItEnd = m.end(); fIt != fItEnd; fIt++) {
    Function &f = *fIt;
    errs().write_escaped(f.getName()) << " : ";
    res.clear();
    readLocalAnnotations(f, res);
    if (!res.empty()) {
      errs() << "\nLocal Set: \n";
      for (auto it : res) {
        errs() << " -> " << *it->first << "\n";
      }
    }
    errs() << "\n";
  }
}
