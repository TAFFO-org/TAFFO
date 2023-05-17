#pragma once
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "FixedPointType.h"
#include "LLVMFloatToFixedPass.h"

namespace flttofix {

bool partialSpecialCallSinCos(
    llvm::Function *oldf, bool &foundRet, flttofix::FixedPointType &fxpret,
    llvm::SmallVector<std::pair<llvm::BasicBlock *, llvm::SmallVector<llvm::Value *, 10>>, 3>
        &to_change);


void fixrangeSinCos(FloatToFixed *ref, llvm::Function *oldf, FixedPointType &fxparg,
                    FixedPointType &fxpret, llvm::Value *arg_value,
                    llvm::IRBuilder<> &builder);

}