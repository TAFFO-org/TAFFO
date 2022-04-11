#include "ReadTrace.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <memory>

#include "TaffoUtils/Metadata.h"
#include "TaffoUtils/InputInfo.h"

using namespace llvm;

#define DEBUG_TYPE "read-trace"

cl::list<std::string> Filenames("trace_file", cl::desc("Specify filenames of trace files"), cl::ZeroOrMore);

//-----------------------------------------------------------------------------
// ReadTrace implementation
//-----------------------------------------------------------------------------
bool ReadTrace::runOnModule(Module &M) {
  bool Changed = false;

  auto &CTX = M.getContext();
  IRBuilder<> Builder(CTX);

  std::unordered_map<std::string, double> minVals, maxVals;
  std::unordered_map<std::string, mdutils::FloatType::FloatStandard> valTypes;

  parseTraceFiles(minVals, maxVals, valTypes);

  for (auto const &i: minVals) {
    std::cout << i.first << " " << "min: " << i.second
    << " max: " << maxVals[i.first]
    << " type: " << valTypes[i.first]
    << std::endl;
  }

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration())
      continue;

    for (auto &BB: F.getBasicBlockList()) {
      auto &InstList = BB.getInstList();
      auto current = InstList.getNextNode(InstList.front());
      while (current != nullptr) {
        auto &Inst = *current;
        auto next = InstList.getNextNode(*current);
        auto InstName = Inst.getName().str();
        if (!Inst.isDebugOrPseudoInst() && Inst.getType()->isFloatingPointTy() && valTypes.count(InstName) != 0) {
          if (next != nullptr) {
            Builder.SetInsertPoint(next);
          } else {
            Builder.SetInsertPoint(&BB.back());
          }
          auto instType = std::make_shared<mdutils::FloatType>(
                  valTypes.at(InstName),
                  maxVals.at(InstName));
          auto instRange = std::make_shared<mdutils::Range>(
                  minVals.at(InstName), maxVals.at(InstName));
          auto instError = std::shared_ptr<double>{};
          mdutils::InputInfo ii{instType, instRange, instError};
          mdutils::MetadataManager::setInputInfoMetadata(Inst, ii);
          Changed = true;
        }
        current = next;
      }
    }
  }

  return Changed;
}

void ReadTrace::parseTraceFiles(std::unordered_map<std::string, double>& minVals,
                                std::unordered_map<std::string, double>& maxVals,
                                std::unordered_map<std::string, mdutils::FloatType::FloatStandard>& valTypes) const {
  for (auto &filename: Filenames) {
    std::cout << "arg: " << filename << std::endl;
    std::string myText;
    std::ifstream MyReadFile(filename);
    while (getline (MyReadFile, myText)) {
      std::string parsed;
      std::stringstream ss(myText);
      getline(ss, parsed, ' ');
      if (parsed != "TAFFO_TRACE") continue;
      getline(ss, parsed, ' ');
      std::string varName = parsed;
      getline(ss, parsed, ' ');
      double varValue = std::stod(parsed);
      getline(ss, parsed, ' ');
      std::string varType = parsed;

      std::cout << "parsed var: " << varName << " ";
      std::cout << "parsed val: " << varValue << " ";
      std::cout << "parsed type: " << varType << std::endl;

      if (auto it = minVals.find(varName) != minVals.end()) {
        if (it > varValue) {
          minVals[varName] = varValue;
        }
      } else {
        minVals[varName] = varValue;
      }

      if (auto it = maxVals.find(varName) != maxVals.end()) {
        if (it < varValue) {
          maxVals[varName] = varValue;
        }
      } else {
        maxVals[varName] = varValue;
      }

      if (valTypes.find(varName) == valTypes.end()) {
        valTypes[varName] = mdutils::FloatType::getFloatStandard(varType);
      }
    }
    MyReadFile.close();
  }
}

PreservedAnalyses ReadTrace::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
