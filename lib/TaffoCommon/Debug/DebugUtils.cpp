#include "DebugUtils.hpp"

#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>

using namespace llvm;

int write_module(const std::string& fileName, const Module& m) {
  int file = 0;
  std::error_code err = sys::fs::openFileForWrite(fileName, file);
  assert(!err.value() && "Failed to open module");
  raw_fd_ostream stream {file, false};
  m.print(stream, nullptr);
  err = sys::fs::closeFile(file);
  assert(!err.value() && "Failed to close file");
  return 0;
}
