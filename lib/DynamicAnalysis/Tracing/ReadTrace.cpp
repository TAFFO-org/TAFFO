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

std::string typeName(Value& val) {
  if (isa<Argument>(val)) {
    return "Argument";
  } else if (isa<Constant>(val)) {
    return "Constant";
  } else if (isa<Instruction>(val)) {
    return "Instruction";
  } else if (isa<Operator>(val)) {
    return "Operator";
  } else {
    return "Unknown";
  }
}

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

//  parseTraceFiles(minVals, maxVals, valTypes);

  std::unordered_map<Value*, int> instToIndex;
  std::unordered_map<int, Value*> indexToInst;
  std::list<std::pair<int, int>> edges;
  int instCount = buildMemEdgesList(M, instToIndex, indexToInst, edges);
  std::map<int, std::list<int>> cc;
  connectedComponents(instCount, edges, cc);

  for (auto it = cc.begin(); it != cc.end(); it++) {
    std::list<int> l = it->second;
    for (auto x : l) {
      errs() << typeName(*indexToInst[x]) << ": ";
      errs() << *indexToInst[x] << "\n";
    }
    errs() << "-----\n";
  }

  return Changed;
}

int merge(int* parent, int x)
{
  if (parent[x] == x)
    return x;
  return merge(parent, parent[x]);
}

void ReadTrace::connectedComponents(int n, std::list<std::pair<int, int>>& edges, std::map<int, std::list<int>>& cc) {
  int parent[n];
  for (int i = 0; i < n; i++) {
    parent[i] = i;
  }
  for (auto x : edges) {
    parent[merge(parent, x.first)] = merge(parent, x.second);
  }
  for (int i = 0; i < n; i++) {
    parent[i] = merge(parent, parent[i]);
  }
  for (int i = 0; i < n; i++) {
    cc[parent[i]].push_back(i);
  }
}

int ReadTrace::buildMemEdgesList(Module &M, std::unordered_map<Value*, int>& instToIndex,
                                  std::unordered_map<int, Value*>& indexToInst,
                                  std::list<std::pair<int, int>>& edges) {
  int index = -1;

  auto getIndex = [&instToIndex, &indexToInst, &index](Value* Inst) -> int {
    auto it = instToIndex.find(Inst);
    errs() << "*** " << *Inst << "\n";
    if (it != instToIndex.end()) {
      return it->second;
    } else {
      index++;
      instToIndex[Inst] = index;
      indexToInst[index] = Inst;
      return index;
    }
  };

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration()) continue;
    for (auto &BB: F.getBasicBlockList()) {
      for (auto &Inst: BB.getInstList()) {
        if (Inst.isDebugOrPseudoInst()) continue ;
        if (isa<AllocaInst, StoreInst, LoadInst, GetElementPtrInst>(Inst)) {
          int instIndex = getIndex(&Inst);
          for (auto child: Inst.users()) {
            int childIndex = getIndex(child);
            edges.emplace_back(instIndex, childIndex);
          }

          if (auto *storeInst = dyn_cast<StoreInst>(&Inst)) {
            auto storeSrc = storeInst->getValueOperand();
            auto storeDst = storeInst->getPointerOperand();
            int srcIndex = getIndex(storeSrc);
            edges.emplace_back(instIndex, srcIndex);
            int dstIndex = getIndex(storeDst);
            edges.emplace_back(instIndex, dstIndex);
          }

          if (auto *loadInst = dyn_cast<LoadInst>(&Inst)) {
            auto loadSrc = loadInst->getPointerOperand();
            int srcIndex = getIndex(loadSrc);
            edges.emplace_back(instIndex, srcIndex);
          }

          if (auto *gepInst = dyn_cast<GetElementPtrInst>(&Inst)) {
            auto gepSrc = gepInst->getPointerOperand();
            int srcIndex = getIndex(gepSrc);
            edges.emplace_back(instIndex, srcIndex);
          }
        }
      } // instructions
    } // basic blocks
  } // functions

  return index + 1; // number of nodes in the graph
} // buildMemEdgesList

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
