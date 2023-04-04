#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include <WriteModule.h>

int write_module(llvm::Twine filename, const llvm::Module &M)
{
  int file = 0;
  auto err = llvm::sys::fs::openFileForWrite(filename, file);
  assert(!err.value() && "Fail open module");
  llvm::raw_fd_ostream stream{file, false};
  M.print(stream, nullptr);
  llvm::sys::fs::closeFile(file);
  return 0;
}

int write_module(const char * filename, const llvm::Module &M)
{
  int file = 0;
  auto err = llvm::sys::fs::openFileForWrite(filename, file);
  assert(!err.value() && "Fail open module");
  llvm::raw_fd_ostream stream{file, false};
  M.print(stream, nullptr);
  llvm::sys::fs::closeFile(file);
  return 0;
}
