#include "llvm/Support/Debug.h"

#ifndef TAFFOUTILS_DEBUGUTILS_H
#define TAFFOUTILS_DEBUGUTILS_H

#ifndef NDEBUG

#define IF_TAFFO_DEBUG \
  if (::llvm::DebugFlag && ::llvm::isCurrentDebugType(DEBUG_TYPE))

#else

#define IF_TAFFO_DEBUG if (false)

#endif

#endif // TAFFOUTILS_DEBUGUTILS_H
