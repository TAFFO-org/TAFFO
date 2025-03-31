#ifndef CUDA_KERNEL_PATCHER_HPP
#define CUDA_KERNEL_PATCHER_HPP

#include <llvm/IR/Module.h>

#define DEBUG_TYPE "taffo-init"

namespace taffo {

void createCudaKernelTrampolines(llvm::Module &M);

} // namespace taffo

#undef DEBUG_TYPE

#endif // CUDA_KERNEL_PATCHER_HPP
