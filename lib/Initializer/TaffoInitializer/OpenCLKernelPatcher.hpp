#ifndef OPEN_CL_KERNEL_PATCHER_HPP
#define OPEN_CL_KERNEL_PATCHER_HPP

#include <llvm/IR/Module.h>

#define DEBUG_TYPE "taffo-init"

namespace taffo
{

void createOpenCLKernelTrampolines(llvm::Module &M);

} // namespace taffo

#undef DEBUG_TYPE

#endif // OPEN_CL_KERNEL_PATCHER_HPP
