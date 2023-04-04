

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Module.h"


int write_module(llvm::Twine filename, const llvm::Module &M);
int write_module(const char * filename, const llvm::Module &M);