#pragma once
#include "TAFFOMath.h"

bool createASin(FloatToFixed *ref, llvm::Function *newfs, llvm::Function *oldf);
bool createACos(FloatToFixed *ref, llvm::Function *newfs, llvm::Function *oldf);