#include "ReadTrace.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "read-trace"

cl::list<std::string> Filenames("trace_file", cl::desc("Specify filenames of trace files"), cl::OneOrMore);

//-----------------------------------------------------------------------------
// ReadTrace implementation
//-----------------------------------------------------------------------------
bool ReadTrace::runOnModule(Module &M) {
  bool InsertedAtLeastOnePrintf = false;

  auto &CTX = M.getContext();
  std::unordered_map<std::string, double> minVals, maxVals;
  std::unordered_map<std::string, std::string> valTypes;

  for (auto &filename: Filenames) {
    printf("arg: %s\n", filename.c_str());
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

      if (auto it = valTypes.find(varName) == valTypes.end()) {
        valTypes[varName] = varType;
      }
    }
    MyReadFile.close();
  }

  for (auto const &i: minVals) {
    std::cout << i.first << " " << "min: " << i.second
    << " max: " << maxVals[i.first]
    << " type: " << valTypes[i.first]
    << std::endl;
  }

  return InsertedAtLeastOnePrintf;
}

PreservedAnalyses ReadTrace::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
