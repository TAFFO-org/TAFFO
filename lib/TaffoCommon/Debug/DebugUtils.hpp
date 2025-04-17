#pragma once

#include <llvm/Support/Debug.h>

#ifndef NDEBUG

#define IF_TAFFO_DEBUG if (::llvm::DebugFlag && ::llvm::isCurrentDebugType(DEBUG_TYPE))

#else

#define IF_TAFFO_DEBUG if (false)

#endif
