

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Module.h"


void write_module(llvm::Twine filename, const llvm::Module &M);