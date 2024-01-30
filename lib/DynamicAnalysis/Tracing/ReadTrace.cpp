#include "ReadTrace.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/ReplaceConstant.h"

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>

#include "TaffoUtils/Metadata.h"
#include "TaffoUtils/InputInfo.h"
#include "TaffoUtils/TypeUtils.h"
#include "TaffoMathUtil.h"
#include "MemoryGraph.h"
#include "ConnectedComponents.h"
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

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration())
      continue;
    for (auto &BB : F.getBasicBlockList()) {
      for (auto &Inst : BB.getInstList()) {
        if (Inst.isDebugOrPseudoInst())
          continue;
        SmallVector<ConstantExpr *> replaceOperands = {};
        for (auto &i : Inst.operands()) {
          if (auto *constExpr = dyn_cast<ConstantExpr>(&i)) {
            replaceOperands.push_back(constExpr);
          }
        }
        for (auto *i : replaceOperands) {
          // we have to do that because LLVM sometimes just folds gep into constexpr and it messes up the analysis
          convertConstantExprsToInstructions(&Inst, i);
        }
        replaceOperands.clear();
      }
    }
  }

  // calculate connected components on the memory operations
  taffo::MemoryGraph graph{M};
  graph.print_graph();
  const std::list<std::pair<int, int>> &edges = graph.getEdges();
  int instCount = graph.getNodeCount();
  llvm::dbgs() << "Nodes in memory graph: " << instCount << "\n";
  taffo::ConnectedComponents ccAlg{instCount, edges};
  const std::unordered_map<int, std::list<int>> &cc = ccAlg.getResult();
  graph.print_connected_components(cc);

  for (auto &it : cc) {
    std::list<int> l = it.second;
    for (auto x : l) {
      ccValues[it.first].emplace_back(graph.getNode(x));
//      llvm::dbgs() << typeName(*indexToInst[x]) << ": ";
//      llvm::dbgs() << *indexToInst[x] << "\n";
    }
//    llvm::dbgs() << "-----\n";
  }

  // read the trace file
  parseTraceFiles(minVals, maxVals, valTypes);

  // calculate value ranges for every component
  calculateCCRanges(ccValues, minVals, maxVals, ccRanges);

  for (const auto &it : ccRanges) {
    const auto range = it.second;
    const auto l = ccValues[it.first];
    for (auto &x : l) {
      llvm::dbgs() << "[" << range.first << ", " << range.second << "] ";
      x->print_debug(llvm::dbgs()) << "\n";
    }
    llvm::dbgs() << "-----\n";
  }

  // assign calculated intervals to the metadata
// loose nodes
  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration()) continue;
    for (auto &BB : F.getBasicBlockList()) {
      for (auto &Inst : BB.getInstList()) {
        if (Inst.isDebugOrPseudoInst()) continue;
        auto CInfoOption = taffo::VRAGlobalStore::computeConstRangeMetadata(
            mdutils::MetadataManager::getMetadataManager(), Inst);
        if (CInfoOption.hasValue()) {
          constInfo[&Inst] = CInfoOption.value();
        }
//        taffo::VRAGlobalStore::setConstRangeMetadata(mdutils::MetadataManager::getMetadataManager(), Inst);
        auto InstName = Inst.getName().str();
        if (minVals.count(InstName) > 0) {
          auto instType = std::shared_ptr<mdutils::FloatType>{};
          auto instRange = std::make_shared<mdutils::Range>(minVals.at(InstName), maxVals.at(InstName));
          auto instError = std::shared_ptr<double>{};
          valuesInfo[&Inst] = std::make_shared<mdutils::InputInfo>(instType, instRange, instError, true, true);
        }
      }
    }
  }

//  nodes from memory graph
  for (const auto &it : ccRanges) {
    const auto range = it.second;
    const auto l = ccValues[it.first];
    bool disableConversion = std::any_of(l.begin(), l.end(), [&](const std::shared_ptr<taffo::ValueWrapper>& item){
      if(item->type == taffo::ValueWrapper::ValueType::ValFunCallArg) {
        auto *funCall = static_cast<const taffo::FunCallArgWrapper *>(&(*item));
        auto * fun = static_cast<const Argument *>(funCall->value)->getParent();
        if (funCall->isExternalFunc && !(fun && TaffoMath::isSupportedLibmFunction(fun, Fixm))) {
          return true;
        }
      } else if (auto *globalVarInst = dyn_cast<GlobalVariable>(item->value)) {
        return !globalVarInst->hasInitializer();
      }
      return false;
    });
    for (auto &valueWrapper : l) {
      if (valueWrapper->isSimpleValue() ||
          (valueWrapper->isStructElem() &&
           std::static_pointer_cast<taffo::StructElemWrapper>(valueWrapper)->isGEPFromStructToSimple())) {
        auto instType = std::shared_ptr<mdutils::FloatType>{};
        auto instRange = std::make_shared<mdutils::Range>(range.first, range.second);
        auto instError = std::shared_ptr<double>{};
        valuesInfo[valueWrapper->value] = std::make_shared<mdutils::InputInfo>(
            instType, instRange, instError, !disableConversion, true);
      } else if (valueWrapper->isStructElem()) {
        addStructInfo(valueWrapper, range, disableConversion);
      } else if (valueWrapper->isFunCallArg() || valueWrapper->isStructElemFunCall()) {
        std::shared_ptr<mdutils::MDInfo> argInfo;
        auto* Arg = dyn_cast<Argument>(valueWrapper->value);
        auto* F = Arg->getParent();
        auto fInfoP = functionsInfo.find(F);
        std::shared_ptr<llvm::SmallVector<std::shared_ptr<mdutils::MDInfo>>> fInfo;
        if (fInfoP == functionsInfo.end()) {
          fInfo = std::make_shared<llvm::SmallVector<std::shared_ptr<mdutils::MDInfo>>>(F->getFunctionType()->getNumParams());
        } else {
          fInfo = fInfoP->second;
        }
        if (valueWrapper->isStructElemFunCall()) {
          argInfo = addStructInfo(valueWrapper, range, disableConversion);
        } else {
          auto instType = std::shared_ptr<mdutils::FloatType>{};
          auto instRange = std::make_shared<mdutils::Range>(range.first, range.second);
          auto instError = std::shared_ptr<double>{};
          argInfo = std::make_shared<mdutils::InputInfo>(
              instType, instRange, instError, !disableConversion, true);
        }
        (*fInfo)[Arg->getArgNo()] = argInfo;
        functionsInfo[F] = fInfo;
      }
    }
  }

  setAllMetadata(M);

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

//  for (const auto &it: valuesRanges) {
//    auto value = it.first;
//    auto range = it.second;
//    if (auto *Inst = dyn_cast<Instruction>(value)) {
//      auto instType = std::shared_ptr<mdutils::FloatType>{};
//      auto instRange = std::make_shared<mdutils::Range>(range->min, range->max);
//      auto instError = std::shared_ptr<double>{};
//      mdutils::InputInfo ii{instType, instRange, instError, !range->disableConversion, true};
//      mdutils::MetadataManager::setInputInfoMetadata(*Inst, ii);
////      llvm::dbgs() << "annotate inst:\n " << *Inst
////             << ", metadata:\n " << ii.toString()
////             << "\n";
//      Changed = true;
//    }
//    if (auto *Arg = dyn_cast<Argument>(value)) {
//      auto F = Arg->getParent();
//      auto instType = std::shared_ptr<mdutils::FloatType>{};
//      auto instRange = std::make_shared<mdutils::Range>(range->min , range->max);
//      auto instError = std::shared_ptr<double>{};
//      mdutils::InputInfo ii{instType, instRange, instError, !range->disableConversion, true};
//      llvm::SmallVector<mdutils::MDInfo *> FunMD;
//      mdutils::MetadataManager::getMetadataManager().retrieveArgumentInputInfo(*F, FunMD);
//      if (!Arg->getType()->isStructTy()) {
//        auto ArgMD = FunMD[Arg->getArgNo()];
//        if (!ArgMD) {
//          FunMD[Arg->getArgNo()] = new mdutils::InputInfo(ii);
//        } else {
//          auto *ArgII = dyn_cast<mdutils::InputInfo>(ArgMD->clone());
//          *ArgII = ii;
//          FunMD[Arg->getArgNo()] = ArgII;
//        }
//        llvm::dbgs() << "annotate arg:\n " << *Arg
//               << ", metadata:\n " << dyn_cast<mdutils::InputInfo>(FunMD[Arg->getArgNo()])->toString()
//               << "\n";
//        mdutils::MetadataManager::setArgumentInputInfoMetadata(*F, FunMD);
//        Changed = true;
//      }
//    }

//    if (auto *GlobalVal = dyn_cast<GlobalVariable>(value)) {
//      if (GlobalVal->hasInitializer()) {
//        auto instType = std::shared_ptr<mdutils::FloatType>{};
//        auto instRange = std::make_shared<mdutils::Range>(range->min, range->max);
//        auto instError = std::shared_ptr<double>{};
//        mdutils::InputInfo ii{instType, instRange, instError, !range->disableConversion, true};
//        mdutils::MetadataManager::setInputInfoMetadata(*GlobalVal, ii);
//        llvm::dbgs() << "annotate global:\n " << *GlobalVal
//                     << ", metadata:\n " << ii.toString()
//                     << "\n";
//        Changed = true;
//      }
//    }
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
//  }

  return true;
}

std::shared_ptr<mdutils::StructInfo> ReadTrace::addStructInfo(std::shared_ptr<taffo::ValueWrapper> valueWrapper,
                                                              const std::pair<double, double> &range,
                                                              bool disableConversion)
{
  Type *structType;
  unsigned int structFieldPos;
  if (valueWrapper->isStructElem()) {
    auto *structElemWrapper = static_cast<taffo::StructElemWrapper *>(&(*valueWrapper));
    structType = structElemWrapper->structType;
    structFieldPos = structElemWrapper->argPos;
  } else {
    auto *structElemWrapper = static_cast<taffo::StructElemFunCallArgWrapper *>(&(*valueWrapper));
    structType = structElemWrapper->structType;
    structFieldPos = structElemWrapper->argPos;
    if(structElemWrapper->value->getType()->isPointerTy()) {
      llvm::dbgs() << "DISABLING STRUCT" << *structType << "\n";
      disableConversion = true;
    }
  }
  auto sInfoP = structsInfo.find(structType);
  std::shared_ptr<mdutils::StructInfo> sInfo;
  if (sInfoP == structsInfo.end()) {
    sInfo = std::make_shared<mdutils::StructInfo>(structType->getStructNumElements());
  } else {
    sInfo = sInfoP->second;
  }
  auto* fieldType = structType->getStructElementType(structFieldPos);
  if (taffo::isFloatType(fieldType)) {
          std::shared_ptr<mdutils::MDInfo> fieldInfo = sInfo->getField(structFieldPos);
          auto instType = std::shared_ptr<mdutils::FloatType>{};
          auto instRange = std::make_shared<mdutils::Range>(range.first, range.second);
          auto instError = std::shared_ptr<double>{};
          if (!fieldInfo.get()) {
            fieldInfo = std::make_shared<mdutils::InputInfo>(instType, instRange, instError, !disableConversion, true);
          } else {
            auto fieldII = std::static_pointer_cast<mdutils::InputInfo>(fieldInfo);
            fieldII->IRange->Min = std::min(instRange->Min, fieldII->IRange->Min);
            fieldII->IRange->Max = std::max(instRange->Max, fieldII->IRange->Max);
            if (disableConversion) {
              fieldII->IEnableConversion = false;
            }
          }
          sInfo->setField(structFieldPos, fieldInfo);
  }
  structsInfo[structType] = sInfo;
  return sInfo;
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
//        auto *callSite = dyn_cast<CallBase>(funCallWrapper->value);
        valueInst = funCallWrapper->value;
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
    llvm::dbgs() << "training data file: " << filename << "\n";
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
      if (varValue < 0) {
        varValue = floor(varValue);
      } else {
        varValue = ceil(varValue);
      }
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
void ReadTrace::setAllMetadata(llvm::Module &M)
{
  llvm::dbgs() << "SETTING CONST METADATA\n";
  for (const auto& item: constInfo) {
    llvm::dbgs() << *item.first << "\n";
    for (auto cInfo: item.second) {
      if (cInfo) {
        llvm::dbgs() << cInfo->toString() << "\n";
      }
    }
    mdutils::MetadataManager::setConstInfoMetadata(*dyn_cast<Instruction>(item.first), item.second);
  }
  llvm::dbgs() << "SETTING VARIABLES METADATA\n";
  for (const auto& item: valuesInfo) {
    llvm::dbgs() << *item.first << "\n";
    if (auto *value = dyn_cast<Instruction>(item.first)) {
      llvm::dbgs() << item.second->toString() << "\n";
      mdutils::MetadataManager::setInputInfoMetadata(*value, *item.second);
    } else if (auto *value = dyn_cast<GlobalObject>(item.first)) {
      llvm::dbgs() << item.second->toString() << "\n";
      mdutils::MetadataManager::setInputInfoMetadata(*value, *item.second);
    }
  }
  llvm::dbgs() << "SETTING STRUCTS METADATA\n";
  for (const auto& cc: ccValues) {
    auto wrapperList = cc.second;
    for (const auto& item: wrapperList) {
      if (item->isStructElem()) {
        auto *structWrapper = static_cast<taffo::StructElemWrapper *>(&(*item));
        auto sInfo = structsInfo[structWrapper->structType];
        structWrapper->print_debug(llvm::dbgs()) << "\n";
        if (sInfo) {
          llvm::dbgs() << sInfo->toString();
          if (auto *value = dyn_cast<Instruction>(item->value)) {
            if (structWrapper->isGEPFromStructToSimple()) {
              continue;
            }
            mdutils::MetadataManager::setStructInfoMetadata(*value, *sInfo);
          } else if (auto *value = dyn_cast<GlobalObject>(item->value)) {
            mdutils::MetadataManager::setStructInfoMetadata(*value, *sInfo);
          }
        }
      }
    }
  }
  llvm::dbgs() << "SETTING FUNCTIONS METADATA\n";
  for (const auto& item: functionsInfo) {
    auto fInfo = llvm::SmallVector<mdutils::MDInfo*>();
    llvm::dbgs() << *item.first << "\n";
    for (const auto& argInfoPtr: *item.second) {
      fInfo.push_back(argInfoPtr.get());
      if (argInfoPtr.get()) {
        llvm::dbgs() << argInfoPtr->toString();
      }
    }
    mdutils::MetadataManager::setArgumentInputInfoMetadata(*item.first, fInfo);
  }
}
