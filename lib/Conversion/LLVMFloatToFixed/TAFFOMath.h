#pragma once
#include "FixedPointType.h"
#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <llvm/IR/Type.h>
#include <utility>

const int MathZ = 2048;

/*
cl::opt<int> MathZ("LuTsize",
                   llvm::cl::desc("Enable Lut table"), llvm::cl::init(2048));
*/
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


} // namespace TaffoMath
