#include "LogAnnotations.h"

#include <iostream>
#include <fstream>

#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include <llvm/Support/raw_ostream.h>

#include "TaffoUtils/Metadata.h"
#include "TaffoUtils/InputInfo.h"
#include "TaffoUtils/TypeUtils.h"

using namespace llvm;

#define DEBUG_TYPE "log-annotations"

cl::opt<std::string> LogAnnotationsFile("annot_log_file", cl::desc("Specify filename for logging annotations"), cl::Optional);

double toIntValue(double varValue) {
  if (varValue < 0) {
    varValue = floor(varValue);
  } else {
    varValue = ceil(varValue);
  }
  return varValue;
}

bool LogAnnotations::runOnModule(llvm::Module &M)
{
  bool Changed = false;
  auto &CTX = M.getContext();
  IRBuilder<> Builder(CTX);
  std::string str;
  llvm::raw_string_ostream sstm(str);

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration())
      continue;

    for (auto &BB: F.getBasicBlockList()) {
      for (auto &Inst: BB.getInstList()) {
        auto *ii = mdutils::MetadataManager::getMetadataManager().retrieveInputInfo(Inst);
        if (ii) {
          auto range = ii->IRange;
          if (range) {
            if (DILocation *Loc = Inst.getDebugLoc()) { // Here I is an LLVM instruction
              unsigned Line = Loc->getLine();
              StringRef File = Loc->getFilename();
              StringRef Dir = Loc->getDirectory();
              bool ImplicitCode = Loc->isImplicitCode();
              sstm << F.getName().str() << ";"
                   << Inst << ";"
                   << toIntValue(range->Min) << ";"
                   << toIntValue(range->Max)  << ";"
                   << Line  << ";"
                   << File.str()  << ";"
                   << Dir.str()  << ";"
                   << ImplicitCode  << ";"
                   << "\n"
                  ;
            }

          }
        }
      }
    }
  }

  for (auto &G: M.getGlobalList()) {
    auto *ii = mdutils::MetadataManager::getMetadataManager().retrieveInputInfo(G);
    if (ii) {
      auto range = ii->IRange;
      if (range) {
        sstm << "GLOBAL" << ";"
             << G.getName().str() << ";"
             << toIntValue(range->Min) << ";"
             << toIntValue(range->Max) << ";;;;;\n";
      }
    }
  }

  auto filename = LogAnnotationsFile.getValue();
  if (!filename.empty()) {
    std::ofstream annot_log_file;
    annot_log_file.open(filename);
    annot_log_file << sstm.str();
    annot_log_file.close();
  } else {
    llvm::dbgs() << sstm.str();
  }

  return Changed;
}

PreservedAnalyses LogAnnotations::run(llvm::Module &M,
                                      llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
