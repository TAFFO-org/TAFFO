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
  errs() << instCount << "\n";
  std::unordered_map<int, std::list<int>> cc;
  std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>> ccValues;
  std::unordered_map<int, std::pair<double, double>> ccRanges;
  connectedComponents(instCount, edges, cc);

  for (auto &it : cc) {
    std::list<int> l = it.second;
    for (auto x : l) {
      ccValues[it.first].emplace_back(graph.getNode(x));
//      errs() << typeName(*indexToInst[x]) << ": ";
//      errs() << *indexToInst[x] << "\n";
    }
//    errs() << "-----\n";
  }

  // read the trace file
  parseTraceFiles(minVals, maxVals, valTypes);

  // calculate value ranges for every component
  calculateCCRanges(ccValues, minVals, maxVals, ccRanges);

  for (const auto &it : ccRanges) {
    const auto range = it.second;
    const auto l = ccValues[it.first];
    for (auto &x : l) {
      errs() << typeName(*(x->value)) << ": ";
      errs() << "[" << range.first << ", " << range.second << "]: ";
      if(x->type == taffo::ValueWrapper::ValueType::ValFunCallArg) {
        auto *funCall = static_cast<taffo::FunCallArgWrapper *>(&(*x));
        if (funCall->isExternalFunc) {
          errs() << "[disabled]: ";
        }
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
    const auto l = ccValues[it.first];
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
      errs() << "annotate inst:\n " << *Inst
             << ", metadata:\n " << ii.toString()
             << "\n";
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

bool ReadTrace::disableConversionForExternalFun(const Value* v) {
  if (auto *arg = dyn_cast<Argument>(v)) {
    errs() << "check arg: " << *arg << "\n";
    if (arg->getType()->isPointerTy()) {
      auto *fun = arg->getParent();
      if (fun->getBasicBlockList().empty()) {
        // this is an external function, don't touch it
        return true;
      }
    }
  }
  if (auto *fun = dyn_cast<Function>(v)) {
    return fun->isVarArg();
  }
  return false;
}

void ReadTrace::calculateCCRanges(const std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>>& ccValues,
                                  const std::unordered_map<std::string, double>& minVals,
                                  const std::unordered_map<std::string, double>& maxVals,
                                  std::unordered_map<int, std::pair<double, double>>& ccRanges) {
  for (const auto& it: ccValues) {
    double minV, maxV;
    bool hasValue = false;
    for (const auto &value: it.second) {
      auto valueName = value->value->getName().str();
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

int merge(int* parent, int x)
{
  if (parent[x] == x)
    return x;
  return merge(parent, parent[x]);
}

void ReadTrace::connectedComponents(const int n, const std::list<std::pair<int, int>>& edges,
                                    std::unordered_map<int, std::list<int>>& cc) {
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


int ReadTrace::buildMemEdgesList(Module &M, std::unordered_map<Value*, int>& instToIndex,
                                  std::unordered_map<int, Value*>& indexToInst,
                                  std::list<std::pair<int, int>>& edges) {
  int index = -1;

  auto getIndex = [&instToIndex, &indexToInst, &index](Value* Inst) -> int {
    auto it = instToIndex.find(Inst);
    //errs() << "*** " << *Inst << "\n";
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
        if (auto *callInst = dyn_cast<CallInst>(&Inst)) {
          if (auto *calledFunction = callInst->getCalledFunction()) {
            errs() << "called function: " << *calledFunction << "\n";
            int i = -1;
            for (Use &actualArg: callInst->arg_operands()) {
              i++;
              if (isFPVal(actualArg)) {
                errs() << "called fun arg: " << *actualArg << "\n";
                int actualArgNodeIndex = getIndex(actualArg);
                if (calledFunction->isVarArg()) {
                  int funIndex = getIndex(calledFunction);
                  edges.emplace_back(funIndex, actualArgNodeIndex);
                } else {
                  auto *formalArg = calledFunction->getArg(i);
                  errs() << "called fun formal arg: " << *formalArg << "\n";
                  int formalArgNodeIndex = getIndex(formalArg);
                  edges.emplace_back(formalArgNodeIndex, actualArgNodeIndex);
                }
              }
            }
          }
        }
        if (isa<AllocaInst, StoreInst, LoadInst, GetElementPtrInst, ReturnInst>(Inst) && isFPVal(&Inst)) {
          int instIndex = getIndex(&Inst);
          for (auto child: Inst.users()) {
            if (isFPVal(child)) {
              int childIndex = getIndex(child);
              edges.emplace_back(instIndex, childIndex);
            }
          }

          if (auto *storeInst = dyn_cast<StoreInst>(&Inst)) {
            auto storeSrc = storeInst->getValueOperand();
            if (isFPVal(storeSrc)) {
              int srcIndex = getIndex(storeSrc);
              edges.emplace_back(instIndex, srcIndex);
            }

            auto storeDst = storeInst->getPointerOperand();
            if (isFPVal(storeDst)) {
              int dstIndex = getIndex(storeDst);
              edges.emplace_back(instIndex, dstIndex);
            }
          }

          if (auto *loadInst = dyn_cast<LoadInst>(&Inst)) {
            auto loadSrc = loadInst->getPointerOperand();
            if (isFPVal(loadSrc)) {
              int srcIndex = getIndex(loadSrc);
              edges.emplace_back(instIndex, srcIndex);
            }
          }

          if (auto *gepInst = dyn_cast<GetElementPtrInst>(&Inst)) {
            auto gepSrc = gepInst->getPointerOperand();
            if (isFPVal(gepSrc)) {
              int srcIndex = getIndex(gepSrc);
              edges.emplace_back(instIndex, srcIndex);
            }
          }

          if (auto *retInst = dyn_cast<ReturnInst>(&Inst)) {
            if (isFPVal(retInst->getReturnValue())) {
              for (auto *funUse: retInst->getFunction()->users()) {
                if (auto *callInst = dyn_cast<CallInst>(&Inst)) {
                  int srcIndex = getIndex(callInst);
                  int retValueIndex = getIndex(retInst->getReturnValue());
                  edges.emplace_back(retValueIndex, srcIndex);
                }
              }
            }
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
