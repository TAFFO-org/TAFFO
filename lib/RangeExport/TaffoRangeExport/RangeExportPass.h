#include <iostream>
#include <fstream>

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#ifndef TAFFO_RANGEEXPORTPASS_H
#define TAFFO_RANGEEXPORTPASS_H

namespace taffo
{

class RangeExportPass: public llvm::PassInfoMixin<RangeExportPass>
{
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  void printMetadata(llvm::Function *F, llvm::Value *value, mdutils::MDInfo *meta, std::ofstream &output_file);
};

} // namespace taffo

#endif // TAFFO_RANGEEXPORTPASS_H
