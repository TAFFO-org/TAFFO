#ifndef TAFFO_INDIRECTCALLPATCHER_H
#define TAFFO_INDIRECTCALLPATCHER_H

#include <list>
#include "CallSiteVersions.h"
#include "llvm/IR/Dominators.h"

namespace taffo {

/// Check whether indirect calls are present in the given module, and patch them with dedicated trampoline calls.
/// The trampolines enable subsequent passes to better analyze the indirect calls.
void manageIndirectCalls(llvm::Module &m);
} // namespace taffo

#endif // TAFFO_INDIRECTCALLPATCHER_H
