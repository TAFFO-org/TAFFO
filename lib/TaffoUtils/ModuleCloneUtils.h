

#include "llvm/ADT/StringRef.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

std::unique_ptr<llvm::Module> getModule(llvm::StringRef Filename, llvm::LLVMContext &cntx);

void cloneGlobalVariable(llvm::Module &host_module, llvm::Module &dev_module, llvm::ValueToValueMapTy &GtoG, llvm::Twine prefix);


std::unique_ptr<llvm::Module> cloneModuleInto(llvm::StringRef Filename, llvm::Module &host_module, llvm::Twine prefix);
