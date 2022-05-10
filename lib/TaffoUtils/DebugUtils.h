#include "llvm/Support/Debug.h"

#ifndef TAFFOUTILS_DEBUGUTILS_H
#define TAFFOUTILS_DEBUGUTILS_H

#ifndef NDEBUG

#define IF_TAFFO_DEBUG \
  if (::llvm::DebugFlag && ::llvm::isCurrentDebugType(DEBUG_TYPE))

#define VALUE_DUMP(val)                                                             \
  LLVM_DEBUG(                                                                       \
      llvm::dbgs() << "isa<Argument> " << isa<Argument>(val) << "\n";               \
      llvm::dbgs() << "isa<BasicBlock> " << isa<BasicBlock>(val) << "\n";           \
      llvm::dbgs() << "isa<InlineAsm> " << isa<InlineAsm>(val) << "\n";             \
      llvm::dbgs() << "isa<MetadataAsValue> " << isa<MetadataAsValue>(val) << "\n"; \
      llvm::dbgs() << "isa<User> " << isa<User>(val) << "\n";                       \
      llvm::dbgs() << "\tisa<Constant> " << isa<Constant>(val) << "\n";             \
      llvm::dbgs() << "\tisa<DerivedUser> " << isa<DerivedUser>(val) << "\n";       \
      llvm::dbgs() << "\tisa<Instruction> " << isa<Instruction>(val) << "\n";       \
      llvm::dbgs() << "\tisa<Operator> " << isa<llvm::Operator>(val) << "\n";);


#else

#define IF_TAFFO_DEBUG if (false)

#endif

#endif // TAFFOUTILS_DEBUGUTILS_H
