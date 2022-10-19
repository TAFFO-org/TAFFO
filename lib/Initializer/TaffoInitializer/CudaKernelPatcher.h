#ifndef CUDA_KERNEL_PATCHER_H
#define CUDA_KERNEL_PATCHER_H

#include "llvm/IR/Module.h"

#define DEBUG_TYPE "taffo-init"

namespace taffo
{

void createCudaKernelTrampolines(llvm::Module &M);

} // namespace taffo

#undef DEBUG_TYPE

#endif
