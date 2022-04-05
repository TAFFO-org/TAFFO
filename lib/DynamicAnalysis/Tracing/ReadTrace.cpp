#include "ReadTrace.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "read-trace"

cl::list<std::string> Filenames("trace_file", cl::desc("Specify filenames of trace files"), cl::OneOrMore);

//-----------------------------------------------------------------------------
// ReadTrace implementation
//-----------------------------------------------------------------------------
bool ReadTrace::runOnModule(Module &M) {
  bool InsertedAtLeastOnePrintf = false;

  auto &CTX = M.getContext();

  for (auto &filename: Filenames) {
    printf("arg: %s\n", filename.c_str());
  }

  return InsertedAtLeastOnePrintf;
}

PreservedAnalyses ReadTrace::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
