#include "ReadTrace.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>
#include <string>
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
  std::unordered_map<Instruction*, double> derivedMinVals, derivedMaxVals;

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
        if (!Inst.isDebugOrPseudoInst() &&
            Inst.getType()->isFloatingPointTy() &&
            valTypes.count(InstName) != 0
//            && !isa<LoadInst>(Inst)
            ) {
          auto instType = std::shared_ptr<mdutils::FloatType>{};
          auto instRange = std::make_shared<mdutils::Range>(
                  minVals.at(InstName), maxVals.at(InstName));
          auto instError = std::shared_ptr<double>{};
          mdutils::InputInfo ii{instType, instRange, instError, false, true};
          mdutils::MetadataManager::setInputInfoMetadata(Inst, ii);
          Changed = true;
        }
        current = next;
      }
    }
  }

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration())
      continue;

    for (auto &BB : F.getBasicBlockList()) {
      auto &InstList = BB.getInstList();
      auto current = InstList.getNextNode(InstList.front());
      while (current != nullptr) {
        auto &Inst = *current;
        auto next = InstList.getNextNode(*current);

        if (auto *storeInst = dyn_cast<StoreInst>(&Inst)) {
          auto *storeSrc = storeInst->getOperand(0);
          auto *storeDst = storeInst->getOperand(1);
          auto srcName = storeSrc->getName().str();
//          std::cout << "Store: "
//                    << storeSrc->getName().str()
//                    << " > "
//                    << storeDst->getName().str()
//                    << std::endl;

          auto ops = std::list<Instruction*>();
          ops.push_back(storeInst);
          ops.push_back(dyn_cast<Instruction>(storeDst));

          if (auto srcMin = minVals.find(srcName) != minVals.end()) {
            auto srcMax = maxVals.find(srcName)->second;
//            errs() << *storeInst << "\n";
//            errs() << *storeDst << "\n";
            for (auto op: storeDst->users()) {
//              errs() << *op << "\n";
              if (!Inst.isDebugOrPseudoInst()) {
                ops.push_back(dyn_cast<Instruction>(op));
              }
            }
            for (auto op: ops) {
              if (auto it = derivedMinVals.find(op) != derivedMinVals.end()) {
                if (it > srcMin) {
                  derivedMinVals[op] = srcMin;
                }
              } else {
                derivedMinVals[op] = srcMin;
              }

              if (auto it = derivedMaxVals.find(op) != derivedMaxVals.end()) {
                if (it < srcMax) {
                  derivedMaxVals[op] = srcMax;
                }
              } else {
                derivedMaxVals[op] = srcMax;
              }
            }
          }
        }
        current = next;
      }
    }
  }

  for (auto pair: derivedMinVals) {
    auto op = pair.first;
    auto minVal = pair.second;
    auto maxVal = derivedMaxVals[op];
    auto instType = std::shared_ptr<mdutils::FloatType>{};
    auto instRange = std::make_shared<mdutils::Range>(minVal, maxVal);
    auto instError = std::shared_ptr<double>{};
    mdutils::InputInfo ii{instType, instRange, instError};
    mdutils::MetadataManager::setInputInfoMetadata(*op, ii);
    Changed = true;
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

//      std::cout << "parsed var: " << varName << " ";
//      std::cout << "parsed val: " << varValue << " ";
//      std::cout << "parsed type: " << varType << std::endl;

      auto minIt = minVals.find(varName);
      if (minIt != minVals.end()) {
        if (minIt->second > varValue) {
          minVals[varName] = varValue;
        }
      } else {
        minVals[varName] = varValue;
      }

      auto maxIt = maxVals.find(varName);
      if (maxIt != maxVals.end()) {
        if (maxIt->second < varValue) {
          maxVals[varName] = varValue;
        }
      } else {
        maxVals[varName] = varValue;
      }

      assert(minVals[varName] <= maxVals[varName]);

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
