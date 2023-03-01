
#include <iostream>

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include "Metadata.h"
#include "RangeExportPass.h"

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace taffo;
using namespace mdutils;

cl::opt<std::string> OutputFilename("ranges_file", cl::desc("Output csv file for variable ranges"));

namespace taffo
{

void RangeExportPass::printMetadata(Function *F,Value *value, MDInfo *meta, std::ofstream &output_file) {
  std::string funcName = "";
  if (F) {
    funcName = F->getName().str();
  }
  std::string varName = "";
  if (value) {
    varName = value->getNameOrAsOperand();
    if (auto *store = dyn_cast<StoreInst>(value)) {
      varName = store->getPointerOperand()->getName().str();
    }
  }
  if (auto *ii = dyn_cast<InputInfo>(meta)) {
    if (ii->IRange) {
      double varMin = ii->IRange->Min;
      double varMax = ii->IRange->Max;
      output_file << funcName << ",";
      output_file << varName << ",";
      output_file << varMin << ",";
      output_file << varMax << "\n";
    }
  } else if (auto *si = dyn_cast<StructInfo>(meta)) {
//    Won't handle struct for now
//    errs() << si->toString() << "\n";
  }
}

PreservedAnalyses RangeExportPass::run(Module &M, ModuleAnalysisManager &AM)
{
  MetadataManager &MDManager = MetadataManager::getMetadataManager();

  std::ofstream output_file(OutputFilename);
  output_file << "function,variable,var_min,var_max" << "\n";

  for (llvm::GlobalVariable &v : M.globals()) {
    auto meta = MDManager.retrieveMDInfo(&v);
    if (meta) {
      printMetadata(nullptr, &v, meta, output_file);
    }
  }

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &Inst : BB) {
        auto meta = MDManager.retrieveMDInfo(&Inst);
        if (meta) {
          printMetadata(&F, &Inst, meta, output_file);
        }
      }
    }
  }

  return PreservedAnalyses::all();
}

} // namespace taffo