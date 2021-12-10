#ifndef TAFFO_UTILS_CALL_SITE_VERSIONS
#define TAFFO_UTILS_CALL_SITE_VERSIONS

#include "llvm/Config/llvm-config.h"

#if (LLVM_VERSION_MAJOR >= 12)
#include "llvm/IR/AbstractCallSite.h"
namespace llvm {
using CallSite = CallBase;
}
#else
#include "llvm/IR/CallSite.h"
#endif

#endif // TAFFO_UTILS_CALL_SITE_VERSIONS
