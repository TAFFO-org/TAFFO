#pragma once

#include <llvm/IR/Module.h>
#define DEBUG_TYPE "taffo-init"

namespace taffo {

void createCudaKernelTrampolines(llvm::Module& M);

} // namespace taffo

#undef DEBUG_TYPE
