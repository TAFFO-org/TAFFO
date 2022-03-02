#pragma once
#include "FixedPointType.h"
#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"
#include "string"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
#include <cstdarg>
#include <llvm/IR/Type.h>
#include <utility>


// value taken from Elementary Function Chapter 7. The CORDIC Algorithm
namespace TaffoMath
{

using namespace flttofix;
using namespace taffo;
// from 0 to 64
const double pi_half = 1.570796326794896619231321691639751442098584699687552910487472296153908203143104499314017412671058534;
const double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068;
const double pi_32 = 4.712388980384689857693965074919254326295754099062658731462416888461724609429313497942052238013175602;
const double pi_2 = 6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136;
const double zero = 0.0f;
const double one = 1.0f;
const double minus_one = -1.0f;


llvm::Constant *createFixedPointFromConst(
    llvm::LLVMContext &cont, FloatToFixed *float_to_fixed, double current_float,
    const FixedPointType &match);

using fxp_pair = std::pair<llvm::Constant *, FixedPointType>;


static fxp_pair make_fxp(const FixedPointType &match, double current_float, llvm::LLVMContext &ctxt, FloatToFixed *ref)
{

  if (match.isFloatingPoint()) {
    return {ConstantFP::get(match.scalarToLLVMType(ctxt), llvm::APFloat(current_float)), match};
  } else {

    return {TaffoMath::createFixedPointFromConst(ctxt, ref, current_float, match), match};
  }
}


/**
 * @param ref used to access member function
 * @param oldf function used to take ret information
 * @param fxpret output of the function *
 * @param n specify wich argument return, valid values ranges from 0 to max number of argument
 * @param found used to return if information was found
 * */
void getFixedFromArg(FloatToFixed *ref, Function *oldf,
                     FixedPointType &fxparg, int n, bool &found);
/**
 * @param ref used to access member function
 * @param oldf function used to take ret information
 * @param fxpret output of the function *
 * @param found used to return if information was found
 * */
void getFixedFromRet(FloatToFixed *ref, Function *oldf,
                     FixedPointType &fxpret, bool &found);


llvm::GlobalVariable *
createGlobalConst(llvm::Module *module, llvm::StringRef Name, llvm::Type *Ty,
                  Constant *initializer, llvm::MaybeAlign alignment);


Value *addAllocaToStart(FloatToFixed *ref, Function *oldf,
                        llvm::IRBuilder<> &builder, Type *to_alloca,
                        llvm::Value *ArraySize = (llvm::Value *)nullptr,
                        const llvm::Twine &Name = "");

template <typename... Args>
void wrapper_printf(llvm::IRBuilder<> &builder, Function *new_f, const std::string &str, Args... args)
{

  // %call3.flt = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3, i64 0, i64 0), double %6), !taffo.info !31, !taffo.
  /*
  auto cntx = &new_f->getContext();
  std::string function_name("printf");

  Function *print = nullptr;
  if ((print = new_f->getParent()->getFunction(function_name)) == 0) {
    std::vector<llvm::Type *> fun_arguments;
    fun_arguments.push_back(
        llvm::Type::getInt8PtrTy(*cntx)); // depends on your type
    FunctionType *fun_type = FunctionType::get(
        llvm::Type::getInt32Ty(*cntx), fun_arguments, true);
    print = llvm::Function::Create(fun_type, GlobalValue::ExternalLinkage,
                                   function_name, new_f->getParent());
  }

  auto constant_string = builder.CreateGlobalStringPtr(str);
  llvm::SmallVector<Value *, 4> small;

  small.insert(small.begin(), constant_string);
  small.insert(small.end(), {args...});

  llvm::dbgs() << "Pizza\n";

  print->dump();

  for (const auto &a : small)
    a->dump();


  builder.CreateCall(print, small);
*/}


  } // namespace TaffoMath
