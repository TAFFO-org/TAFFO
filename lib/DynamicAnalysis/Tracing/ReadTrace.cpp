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
#include "TaffoUtils/TypeUtils.h"
#include "MemoryGraph.h"
#include "ConnectedComponents.h"
#include "ExpandEqualValue.h"
#include "RangeAnalysis/TaffoVRA/VRAGlobalStore.hpp"

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

  // calculate connected components on the memory operations
  taffo::MemoryGraph graph{M};
  const std::list<std::pair<int, int>> &edges = graph.getEdges();
  int instCount = graph.getNodeCount();
  errs() << "Nodes in memory graph: " << instCount << "\n";
  taffo::ConnectedComponents ccAlg{instCount, edges};
  const std::unordered_map<int, std::list<int>> &cc = ccAlg.getResult();
  std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>> ccValues;
  std::unordered_map<int, std::pair<double, double>> ccRanges;

  for (auto &it : cc) {
    std::list<int> l = it.second;
    for (auto x : l) {
      ccValues[it.first].emplace_back(graph.getNode(x));
//      errs() << typeName(*indexToInst[x]) << ": ";
//      errs() << *indexToInst[x] << "\n";
    }
//    errs() << "-----\n";
  }

  taffo::ExpandEqualValue expand{ccValues};
  auto &expandedCCValues = expand.getResult();

  // read the trace file
  parseTraceFiles(minVals, maxVals, valTypes);

  // calculate value ranges for every component
  calculateCCRanges(expandedCCValues, minVals, maxVals, ccRanges);

  for (const auto &it : ccRanges) {
    const auto range = it.second;
    const auto l = expandedCCValues[it.first];
    for (auto &x : l) {
      errs() << typeName(*(x->value)) << ": ";
      errs() << "[" << range.first << ", " << range.second << "]: ";
      if(x->type == taffo::ValueWrapper::ValueType::ValFunCallArg) {
        auto *funCall = static_cast<taffo::FunCallArgWrapper *>(&(*x));
        if (funCall->isExternalFunc) {
          errs() << "[disabled]: ";
        }
        errs() << "[arg: " << funCall->argPos << "]: ";
      }
      errs() << *(x->value) << "\n";
    }
    errs() << "-----\n";
  }

  // assign calculated intervals to the metadata
  std::unordered_map<Value*, std::shared_ptr<DynamicValueInfo>> valuesRanges;

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration()) continue;
    for (auto &BB : F.getBasicBlockList()) {
      for (auto &Inst : BB.getInstList()) {
        if (Inst.isDebugOrPseudoInst()) continue;
        taffo::VRAGlobalStore::setConstRangeMetadata(mdutils::MetadataManager::getMetadataManager(), Inst);
        auto InstName = Inst.getName().str();
        if (minVals.count(InstName) > 0) {
          valuesRanges[&Inst] = std::make_shared<DynamicValueInfo>(
              minVals.at(InstName), maxVals.at(InstName), false);
        }
      } // instructions
    } // basic blocks
  } // functions

  for (const auto &it : ccRanges) {
    const auto range = it.second;
    const auto l = expandedCCValues[it.first];
    bool disableConversion = std::any_of(l.begin(), l.end(), [&](const auto& item){
      if(item->type == taffo::ValueWrapper::ValueType::ValFunCallArg) {
        auto *funCall = static_cast<const taffo::FunCallArgWrapper *>(&(*item));
        if (funCall->isExternalFunc) {
          return true;
        }
      }
      return false;
    });
    for (auto &value : l) {
      valuesRanges[value->value] = std::make_shared<DynamicValueInfo>(
          range.first, range.second, disableConversion);
    }
  }

  // annotate global variables
//  for (llvm::GlobalVariable &v : M.globals()) {
//    if (valuesRanges.count(&v)) {
//      mdutils::MetadataManager &MDManager = mdutils::MetadataManager::getMetadataManager();
//      const auto range = valuesRanges.at(&v);
//      // retrieve info about global var v, if any
//      if (mdutils::MDInfo *mdi = MDManager.retrieveInputInfo(*dyn_cast<GlobalObject>(&v))) {
//        auto * cpymdi(dyn_cast<mdutils::InputInfo>(mdi->clone()));
//        if(cpymdi->isFinal()) continue;
//        cpymdi->IRange =  std::make_shared<mdutils::Range>(range.first, range.second);
//        mdutils::MetadataManager::setMDInfoMetadata(&v, cpymdi);
//      } else {
//        auto instType = std::shared_ptr<mdutils::FloatType>{};
//        auto instRange = std::make_shared<mdutils::Range>(range.first, range.second);
//        auto instError = std::shared_ptr<double>{};
//        mdutils::InputInfo ii{instType, instRange, instError, true, true};
//        mdutils::MetadataManager::setInputInfoMetadata(*dyn_cast<GlobalObject>(&v), ii);
//      }
//    }
//  }

  for (const auto &it: valuesRanges) {
    auto value = it.first;
    auto range = it.second;
    if (auto *Inst = dyn_cast<Instruction>(value)) {
      auto instType = std::shared_ptr<mdutils::FloatType>{};
      auto instRange = std::make_shared<mdutils::Range>(range->min, range->max);
      auto instError = std::shared_ptr<double>{};
      mdutils::InputInfo ii{instType, instRange, instError, !range->disableConversion, true};
      mdutils::MetadataManager::setInputInfoMetadata(*Inst, ii);
//      errs() << "annotate inst:\n " << *Inst
//             << ", metadata:\n " << ii.toString()
//             << "\n";
      Changed = true;
    }
    if (auto *Arg = dyn_cast<Argument>(value)) {
      auto F = Arg->getParent();
      auto instType = std::shared_ptr<mdutils::FloatType>{};
      auto instRange = std::make_shared<mdutils::Range>(range->min , range->max);
      auto instError = std::shared_ptr<double>{};
      mdutils::InputInfo ii{instType, instRange, instError, !range->disableConversion, true};
      llvm::SmallVector<mdutils::MDInfo *> FunMD;
      mdutils::MetadataManager::getMetadataManager().retrieveArgumentInputInfo(*F, FunMD);
      if (!Arg->getType()->isStructTy()) {
        auto ArgMD = FunMD[Arg->getArgNo()];
        if (!ArgMD) {
          FunMD[Arg->getArgNo()] = new mdutils::InputInfo(ii);
        } else {
          auto *ArgII = dyn_cast<mdutils::InputInfo>(ArgMD->clone());
          *ArgII = ii;
          FunMD[Arg->getArgNo()] = ArgII;
        }
        errs() << "annotate arg:\n " << *Arg
               << ", metadata:\n " << dyn_cast<mdutils::InputInfo>(FunMD[Arg->getArgNo()])->toString()
               << "\n";
        mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, FunMD);
        Changed = true;
      }
    }

    if (auto *GlobalVal = dyn_cast<GlobalObject>(value)) {
      auto instType = std::shared_ptr<mdutils::FloatType>{};
      auto instRange = std::make_shared<mdutils::Range>(range->min, range->max);
      auto instError = std::shared_ptr<double>{};
      mdutils::InputInfo ii{instType, instRange, instError, !range->disableConversion, true};
      mdutils::MetadataManager::setInputInfoMetadata(*GlobalVal, ii);
      errs() << "annotate global:\n " << *GlobalVal
             << ", metadata:\n " << ii.toString()
             << "\n";
      Changed = true;
    }
//    if (auto *ConstVal = dyn_cast<Constant>(value)) {
//      auto instType = std::shared_ptr<mdutils::FloatType>{};
//      auto instRange = std::make_shared<mdutils::Range>(range.first, range.second);
//      auto instError = std::shared_ptr<double>{};
//      for (auto *user: ConstVal->users()) {
//        if (auto * InstUser = dyn_cast<Instruction>(user)) {
//          mdutils::InputInfo ii{instType, instRange, instError, true, true};
//          auto ResII = SmallVector<mdutils::InputInfo *>(InstUser->getNumOperands());
//          mdutils::MetadataManager::getMetadataManager().retrieveConstInfo(*InstUser, ResII);
//          int i = 0;
//          for (auto &op: InstUser->operands()) {
//            if (ConstVal == dyn_cast<Constant>(&op)) {
//              auto ArgMD = ResII[i];
//              if (!ArgMD) {
//                ResII[i] = new mdutils::InputInfo(ii);
//              } else {
//                auto *ArgII = dyn_cast<mdutils::InputInfo>(ArgMD);
//                *ArgII = ii;
//              }
//            }
//            i++;
//          }
//          mdutils::MetadataManager::setConstInfoMetadata(*InstUser, ResII);
//        }
//      }
//      Changed = true;
//    }
  }

  return Changed;
}

void ReadTrace::calculateCCRanges(const std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>>& ccValues,
                                  const std::unordered_map<std::string, double>& minVals,
                                  const std::unordered_map<std::string, double>& maxVals,
                                  std::unordered_map<int, std::pair<double, double>>& ccRanges) {
  for (const auto& it: ccValues) {
    double minV, maxV;
    bool hasValue = false;
    for (const auto &value: it.second) {
      Value *valueInst;
      if (value->type == taffo::ValueWrapper::ValueType::ValFunCallArg) {
        auto *funCallWrapper = static_cast<taffo::FunCallArgWrapper *>(&(*value));
        auto *callSite = dyn_cast<CallBase>(funCallWrapper->value);
        valueInst = callSite->getArgOperand(funCallWrapper->argPos);
      } else {
        valueInst = value->value;
      }
      auto valueName = valueInst->getName().str();
      if (minVals.count(valueName) > 0) {
        if (!hasValue) {
          hasValue = true;
          minV = minVals.at(valueName);
          maxV = maxVals.at(valueName);
        } else {
          minV = minV <= minVals.at(valueName)? minV: minVals.at(valueName);
          maxV = maxV >= maxVals.at(valueName)? maxV: maxVals.at(valueName);
        }
      }
    }
    if (hasValue) {
      ccRanges[it.first] = {minV, maxV};
    }
  }
}

std::string ReadTrace::typeName(const Value& val) {
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

bool isFPVal(Value* value) {
  if (auto *inst = dyn_cast<AllocaInst>(value)) {
    return taffo::isFloatType(inst->getAllocatedType());
  }
  if (auto *inst = dyn_cast<StoreInst>(value)) {
    return taffo::isFloatType(inst->getPointerOperandType());
  }
  if (auto *inst = dyn_cast<LoadInst>(value)) {
    return taffo::isFloatType(inst->getPointerOperandType());
  }
  if (auto *inst = dyn_cast<GetElementPtrInst>(value)) {
    return taffo::isFloatType(inst->getPointerOperandType());
  }
  if (auto *inst = dyn_cast<Argument>(value)) {
    return taffo::isFloatType(inst->getType());
  }
  if (auto *inst = dyn_cast<Constant>(value)) {
    return taffo::isFloatType(inst->getType());
  }
  return taffo::isFloatType(value->getType());
}

void ReadTrace::parseTraceFiles(std::unordered_map<std::string, double>& minVals,
                                std::unordered_map<std::string, double>& maxVals,
                                std::unordered_map<std::string, mdutils::FloatType::FloatStandard>& valTypes) const {
  for (auto &filename: Filenames) {
    errs() << "training data file: " << filename << "\n";
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