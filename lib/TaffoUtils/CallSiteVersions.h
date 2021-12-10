

#if (LLVM_VERSION_MAJOR == 12)
#include "llvm/IR/AbstractCallSite.h"
using CallSite = llvm::CallBase;
#endif
#if (LLVM_VERSION_MAJOR == 11)
#include "llvm/IR/CallSite.h"
#endif
