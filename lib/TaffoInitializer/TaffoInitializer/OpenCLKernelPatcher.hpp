#pragma once

#include "llvm/IR/Module.h"
#define DEBUG_TYPE "taffo-init"

namespace taffo {

void createOpenCLKernelTrampolines(llvm::Module& M);

} // namespace taffo

#undef DEBUG_TYPE
