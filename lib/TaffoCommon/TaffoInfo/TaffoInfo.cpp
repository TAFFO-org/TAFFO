#include "TaffoInfo.hpp"

#include "MetadataManager.hpp"
#include "Debug/Logger.hpp"
#include "Types/TypeUtils.hpp"

#include <llvm/IR/Dominators.h>
#include <fstream>

#define DEBUG_TYPE "taffo-util"

using namespace taffo;
using namespace llvm;

TaffoInfo &TaffoInfo::getInstance() {
  static TaffoInfo instance;
  return instance;
}

void TaffoInfo::setTransparentType(Value &v, const std::shared_ptr<TransparentType> &t) {
  transparentTypes[&v] = t;
}

std::shared_ptr<TransparentType> TaffoInfo::getOrCreateTransparentType(Value &v) {
  auto iter = transparentTypes.find(&v);
  if (iter != transparentTypes.end())
    return iter->second;
  std::shared_ptr<TransparentType> type = TransparentTypeFactory::create(&v);
  LLVM_DEBUG(
    Logger &logger = Logger::getInstance();
    logger.setContextTag(logContextTag);
    logger.log("Missing transparent type for value: ", raw_ostream::Colors::YELLOW);
    logger.logValueln(&v);
    logger.log("Transparent type set to:            ", raw_ostream::Colors::YELLOW);
    logger.logln(type, raw_ostream::Colors::CYAN);
    if (type->isOpaquePointer())
      logger.logln("Warning: the newly created transparent type is opaque", raw_ostream::Colors::RED);
    logger.restorePrevContextTag();
  );
  return transparentTypes[&v] = type;
}

bool TaffoInfo::hasTransparentType(const Value &v) {
  auto iter = transparentTypes.find(&v);
  return iter != transparentTypes.end();
}

void TaffoInfo::addStartingPoint(Function &f) {
  if (!is_contained(startingPoints, &f))
  startingPoints.push_back(&f);
}

void TaffoInfo::addDefaultStartingPoint(Module &m) {
  auto main = find_if(m.functions(),
    [](Function &f) { return f.getName().equals("main"); });
  if (main != m.end())
    addStartingPoint(*main);
}

bool TaffoInfo::isStartingPoint(Function &f) const {
  return is_contained(startingPoints, &f);
}

bool TaffoInfo::hasStartingPoint(Module &m) const {
  return any_of(m.functions(), [this](auto &f) { return isStartingPoint(f); });
}

void TaffoInfo::setIndirectFunction(CallInst &call, Function &f) {
  indirectFunctions[&call] = &f;
}

Function *TaffoInfo::getIndirectFunction(const CallInst &call) const {
  auto iter = indirectFunctions.find(&call);
  return iter != indirectFunctions.end() ? iter->second : nullptr;
}

bool TaffoInfo::isIndirectFunction(const CallInst &call) const {
  return indirectFunctions.contains(&call);
}

void TaffoInfo::setOpenCLTrampoline(Function &f, Function &kernF) {
  oclTrampolines[&f] = &kernF;
}

Function *TaffoInfo::getOpenCLTrampoline(const Function &f) const {
  auto iter = oclTrampolines.find(&f);
  return iter != oclTrampolines.end() ? iter->second : nullptr;
}

bool TaffoInfo::isOpenCLTrampoline(const Function &f) const {
  return oclTrampolines.contains(&f);
}

void TaffoInfo::disableConversion(Instruction &i) {
  disabledConversion.push_back(&i);
}

bool TaffoInfo::isConversionDisabled(Instruction &i) const {
  return is_contained(disabledConversion, &i);
}

void TaffoInfo::createValueInfo(Value &v) {
  valueInfo[&v] = ValueInfoFactory::create(&v);
}

void TaffoInfo::setValueInfo(Value &v, const std::shared_ptr<ValueInfo> &vi) {
  valueInfo[&v] = vi;
}

void TaffoInfo::setValueInfo(Value &v, std::shared_ptr<ValueInfo> &&vi) {
  valueInfo[&v] = std::move(vi);
}

std::shared_ptr<ValueInfo> TaffoInfo::getValueInfo(const Value &v) const {
  auto iter = valueInfo.find(&v);
  return iter != valueInfo.end() ? iter->second : nullptr;
}

bool TaffoInfo::hasValueInfo(const Value &v) const {
  return valueInfo.contains(&v);
}

void TaffoInfo::setValueWeight(Value &v, int weight) {
  valueWeights[&v] = weight;
}

int TaffoInfo::getValueWeight(const Value &v) const {
  auto iter = valueWeights.find(&v);
  return iter != valueWeights.end() ? iter->second : -1;
}

void TaffoInfo::setTaffoFunction(Function &originalF, Function &taffoF) {
  originalFunctions[&taffoF] = &originalF;
  taffoFunctions[&originalF].insert(&taffoF);
}

Function *TaffoInfo::getOriginalFunction(Function &taffoF) const {
  auto iter = originalFunctions.find(&taffoF);
  return iter != originalFunctions.end() ? iter->second : nullptr;
}

bool TaffoInfo::hasTaffoFunctions(const Function &originalF) const {
  return taffoFunctions.contains(&originalF);
}

bool TaffoInfo::isTaffoFunction(Function &f) const {
  return originalFunctions.contains(&f);
}

void TaffoInfo::getTaffoFunctions(const Function &originalF, SmallPtrSetImpl<Function*> &taffoFunctions) const {
  auto iter = this->taffoFunctions.find(&originalF);
  if (iter != this->taffoFunctions.end())
    for (auto taffoF : iter->second)
      taffoFunctions.insert(taffoF);
}

void TaffoInfo::setMaxRecursionCount(Function &f, unsigned int maxRecursion) {
  maxRecursionCount[&f] = maxRecursion;
}

unsigned int TaffoInfo::getMaxRecursionCount(const Function &f) const {
  auto iter = maxRecursionCount.find(&f);
  return iter != maxRecursionCount.end() ? iter->second : 0;
}

void TaffoInfo::setLoopUnrollCount(Loop &l, unsigned int unrollCount) {
  loopUnrollCount[&l] = unrollCount;
}

unsigned int TaffoInfo::getLoopUnrollCount(const Loop &l) const {
  auto iter = loopUnrollCount.find(&l);
  return iter != loopUnrollCount.end() ? iter->second : 0;
}

void TaffoInfo::setError(Instruction &i, double err) {
  error[&i] = err;
}

double TaffoInfo::getError(Instruction &i) const {
  auto iter = error.find(&i);
  return iter != error.end() ? iter->second : 0.0;
}

void TaffoInfo::setCmpErrorMetadata(Instruction &i, CmpErrorInfo &compErrorInfo) {
  cmpError[&i] = std::make_unique<CmpErrorInfo>(compErrorInfo);
}

std::shared_ptr<CmpErrorInfo> TaffoInfo::getCmpError(const Instruction &i) const {
  auto iter = cmpError.find(&i);
  return iter != cmpError.end() ? iter->second : nullptr;
}

Type *TaffoInfo::getType(const std::string &typeId) const {
  auto iter = idTypeMapping.find(typeId);
  return iter != idTypeMapping.end() ? iter->second : nullptr;
}

void TaffoInfo::eraseValue(Value &v) {
  erasedValues.insert(&v);
  idValueMapping.eraseByValue(&v);
}

void TaffoInfo::eraseLoop(Loop &l) {
  erasedLoops.insert(&l);
  idLoopMapping.eraseByValue(&l);
}

void TaffoInfo::dumpToFile(const std::string &filePath, Module &m) {
  LLVM_DEBUG(
    Logger &logger = Logger::getInstance();
    logger.setContextTag(logContextTag);
    logger.logln("Dumping...");
  );

  generateTaffoIds();
  MetadataManager::setIdValueMapping(idValueMapping, m);
  MetadataManager::setIdLoopMapping(idLoopMapping, m);
  MetadataManager::setIdTypeMapping(idTypeMapping, m);
  jsonRepresentation = serialize();

  std::ofstream outFile(filePath);
  if (!outFile.is_open())
    report_fatal_error("Could not open file for writing taffo info\n");
  outFile << jsonRepresentation.dump(4);
  outFile.close();

  LLVM_DEBUG(
    Logger &logger = Logger::getInstance();
    Logger::getInstance().logln("Dumped to file " + filePath);
    logger.restorePrevContextTag();
  );
}

void TaffoInfo::initializeFromFile(const std::string &filePath, Module &m) {
  LLVM_DEBUG(
    Logger &logger = Logger::getInstance();
    logger.setContextTag(logContextTag);
    logger.logln("Initializing...");
  );

  idValueMapping = MetadataManager::getIdValueMapping(m);
  idLoopMapping = MetadataManager::getIdLoopMapping(m);
  idTypeMapping = MetadataManager::getIdTypeMapping(m);
  jsonRepresentation.clear();

  std::ifstream inFile(filePath);
  if (!inFile.is_open())
    report_fatal_error("Could not open file for reading taffo info\n");
  inFile >> jsonRepresentation;
  inFile.close();

  deserialize(jsonRepresentation);

  LLVM_DEBUG(
    Logger &logger = Logger::getInstance();
    Logger::getInstance().logln("Initialized from file " + filePath);
    logger.restorePrevContextTag();
  );
}

void TaffoInfo::generateTaffoIds() {
  // Collect all values from containers
  std::set<Value*> valueSet;

  for (const auto &[v, transparentType] : transparentTypes) {
    valueSet.insert(v);
    // Also update idTypeMapping while collecting values from transparentTypes.
    idTypeMapping[toString(transparentType->getUnwrappedType())] = transparentType->getUnwrappedType();
  }
  for (auto *f : startingPoints)
    valueSet.insert(f);
  for (auto &[call, f] : indirectFunctions) {
    valueSet.insert(call);
    valueSet.insert(f);
  }
  for (auto &[f, kernF] : oclTrampolines) {
    valueSet.insert(f);
    valueSet.insert(kernF);
  }
  for (auto *i : disabledConversion)
    valueSet.insert(i);
  for (auto &[taffoF, originalF] : originalFunctions) {
    valueSet.insert(taffoF);
    valueSet.insert(originalF);
    // No need to get values of taffoFunctions as they are the same of originalFunctions
  }
  for (auto &[v, _] : valueInfo)
    valueSet.insert(v);
  for (auto &[v, _] : valueWeights)
    valueSet.insert(v);
  for (auto &[f, _] : maxRecursionCount)
    valueSet.insert(f);
  for (auto &[inst, _] : error)
    valueSet.insert(inst);
  for (auto &[inst, _] : cmpError)
    valueSet.insert(inst);

  // Set the taffoId of each value, updating idValueMapping and idTypeMapping
  for (auto *value : valueSet) {
    if (erasedValues.contains(value))
      continue;
    generateTaffoId(value);

    std::shared_ptr<ValueInfo> valInfo = this->valueInfo[value];
    Type *type = getUnwrappedType(value);
    idTypeMapping[toString(type)] = type;
  }

  // The same for loops
  std::set<Loop*> loopSet;
  for (auto &[l, _] : loopUnrollCount)
    loopSet.insert(l);

  for (auto *l : loopSet)
    if (!erasedLoops.contains(l))
      generateTaffoId(l);
}

void TaffoInfo::generateTaffoId(Value *v) {
  if (idValueMapping.containsValue(v))
    return;
  std::string id = generateValueId(v);
  idValueMapping[id] = v;
}

void TaffoInfo::generateTaffoId(Loop *l) {
  if (idLoopMapping.containsValue(l))
    return;
  std::string id = generateLoopId(l);
  for (Loop *subLoop : l->getSubLoops())
    generateTaffoId(subLoop);
}

std::string TaffoInfo::generateValueId(const Value *v) {
  std::string idStr;
  raw_string_ostream os(idStr);

  if (const auto *f = dyn_cast<Function>(v))
    os << f->getName();
  else if (isa<GlobalValue>(v))
    os << "global";
  else if (const auto *inst = dyn_cast<Instruction>(v)) {
    if (const Function *parentF = inst->getFunction())
      os << parentF->getName() << "_";
    os << "inst";
  }
  else if (const auto *arg = dyn_cast<Argument>(v)) {
    if (const Function *parentF = arg->getParent())
      os << parentF->getName() << "_";
    os << "arg";
  }
  else if (isa<Constant>(v)) {
    os << "const";
    if (!v->getType()->isStructTy())
      os << "_" << toString(v->getType());
  }
  else
    os << "val";

  idCounter++;
  updateIdDigits();
  os << "_" << formatNumber(idDigits, idCounter);
  return os.str();
}

std::string TaffoInfo::generateLoopId(const Loop *l) {
  std::string idStr;
  raw_string_ostream os(idStr);

  if (BasicBlock *header = l->getHeader())
    if (const Function *F = header->getParent())
      os << F->getName() << "_";

  idCounter++;
  updateIdDigits();
  os << "loop_" << formatNumber(idDigits, idCounter);
  return os.str();
}

void TaffoInfo::updateIdDigits() {
  unsigned int idCounterCopy = idCounter;
  unsigned int newDigits = 0;
  do {
    newDigits++;
    idCounterCopy /= 10;
  } while(idCounterCopy);

  if (newDigits != idDigits) {
    idDigits = newDigits;

    auto updateIds = [this](auto &map, const std::string &jsonMapName) {
      std::vector<std::string> oldIds;
      for (const auto &[id, _] : map)
        oldIds.push_back(id);
      for (const auto &oldId : oldIds) {
        size_t pos = oldId.find_last_not_of("0123456789");
        std::string prefix = oldId.substr(0, pos + 1);
        std::string numericPart = oldId.substr(pos + 1);
        unsigned int number = numericPart.empty() ? 0 : std::stoul(numericPart);
        std::string newId = prefix + formatNumber(idDigits, number);
        map.updateKey(oldId, newId);
        if (jsonRepresentation.contains(jsonMapName))
          jsonRepresentation[jsonMapName].erase(oldId);
      }
    };

    updateIds(idValueMapping, "values");
    updateIds(idLoopMapping, "loops");
  }
}

json TaffoInfo::serialize() const {
  json j = jsonRepresentation;

  // Serialize idCounter
  j["idCounter"] = idCounter;

  // Serialize id-value mapping
  j["valueCount"] = idValueMapping.size();
  for (const auto &[id, value] : idValueMapping) {
    json valueJson;
    // Determine the kind of LLVM value and record a suitable representation
    if (auto *f = dyn_cast<Function>(value)) {
      valueJson["repr"] = f->getName().str();
      valueJson["kind"] = "function";
    } else if (isa<GlobalValue>(value)) {
      valueJson["repr"] = toString(value);
      valueJson["kind"] = "global";
    } else if (auto *inst = dyn_cast<Instruction>(value)) {
      valueJson["repr"] = toString(inst);
      valueJson["kind"] = "instruction";
      if (const Function *parentF = inst->getFunction())
        valueJson["function"] = parentF->getName().str();
    } else if (auto *arg = dyn_cast<Argument>(value)) {
      valueJson["repr"] = toString(arg);
      valueJson["kind"] = "argument";
      if (const Function *parentF = arg->getParent())
        valueJson["function"] = parentF->getName().str();
    } else if (isa<Constant>(value)) {
      valueJson["repr"] = toString(value);
      valueJson["kind"] = "constant";
    } else {
      valueJson["repr"] = toString(value);
      valueJson["kind"] = "value";
    }
    j["values"][id] = valueJson;
  }

  // Serialize id-loop mapping
  j["loopCount"] = idLoopMapping.size();
  for (const auto &[id, loop] : idLoopMapping) {
    json loopJson;
    if (BasicBlock *header = loop->getHeader()) {
      loopJson["header"] = toString(header);
      if (const Function *F = header->getParent())
        loopJson["function"] = F->getName().str();
    }
    j["loops"][id] = loopJson;
  }

  // Serialize deducedPointerTypes as a field in each value’s JSON entry
  for (const auto &[v, transparentType] : transparentTypes) {
    if (erasedValues.contains(v))
      continue;
    std::string id = idValueMapping.findByValue(v)->first;
    j["values"][id]["transparentType"] = transparentType->serialize();
  }

  // Serialize starting points
  j["startingPoints"] = json::array();
  for (auto* f : startingPoints) {
    if (erasedValues.contains(f))
      continue;
    std::string id = idValueMapping.findByValue(f)->first;
    j["startingPoints"].push_back(id);
  }

  // Serialize indirect functions
  j["indirectFunctions"] = json::object();
  for (auto &[call, f] : indirectFunctions) {
    if (erasedValues.contains(call) || erasedValues.contains(f))
      continue;
    std::string callId = idValueMapping.findByValue(call)->first;
    std::string funcId = idValueMapping.findByValue(f)->first;
    j["indirectFunctions"][callId] = funcId;
  }

  // Serialize OpenCL trampolines
  j["oclTrampolines"] = json::object();
  for (auto &[tramp, kernF] : oclTrampolines) {
    if (erasedValues.contains(tramp) || erasedValues.contains(kernF))
      continue;
    std::string trampId = idValueMapping.findByValue(tramp)->first;
    std::string kernFId = idValueMapping.findByValue(kernF)->first;
    j["oclTrampolines"][trampId] = kernFId;
  }

  // Serialize disabled conversions
  j["disabledConversion"] = json::array();
  for (auto* inst : disabledConversion) {
    if (erasedValues.contains(inst))
      continue;
    std::string id = idValueMapping.findByValue(inst)->first;
    j["disabledConversion"].push_back(id);
  }

  // Serialize originalFunctions
  j["originalFunctions"] = json::object();
  for (auto &[taffoF, originalF] : originalFunctions) {
    if (erasedValues.contains(taffoF) || erasedValues.contains(originalF))
      continue;
    std::string taffoId = idValueMapping.findByValue(taffoF)->first;
    std::string originalId = idValueMapping.findByValue(originalF)->first;
    j["originalFunctions"][taffoId] = originalId;
  }

  // Serialize taffoFunctions
  j["taffoFunctions"] = json::object();
  for (auto &[originalF, taffoFunctions] : taffoFunctions) {
    if (erasedValues.contains(originalF))
      continue;
    std::string originalId = idValueMapping.findByValue(originalF)->first;
    j["taffoFunctions"][originalId] = json::array();
    for (auto taffoF : taffoFunctions) {
      if (erasedValues.contains(taffoF))
        continue;
      std::string taffoId = idValueMapping.findByValue(taffoF)->first;
      j["taffoFunctions"][originalId].push_back(taffoId);
    }
  }

  // Serialize valueInfo
  for (auto &[v, vi] : valueInfo) {
    if (erasedValues.contains(v))
      continue;
    std::string id = idValueMapping.findByValue(v)->first;
    j["values"][id]["info"] = vi ? vi->serialize() : nullptr;
  }

  // Serialize valueWeights
  for (auto &[val, weight] : valueWeights) {
    if (erasedValues.contains(val))
      continue;
    std::string id = idValueMapping.findByValue(val)->first;
    j["values"][id]["weight"] = weight;
  }

  // Serialize maxRecursionCount
  j["maxRecursionCount"] = json::object();
  for (auto &[fun, count] : maxRecursionCount) {
    if (erasedValues.contains(fun))
      continue;
    std::string id = idValueMapping.findByValue(fun)->first;
    j["maxRecursionCount"][id] = count;
  }

  // Serialize loopUnrollCount
  j["loopUnrollCount"] = json::object();
  for (auto &[loop, count] : loopUnrollCount) {
    if (erasedLoops.contains(loop))
      continue;
    std::string id = idLoopMapping.findByValue(loop)->first;
    j["loopUnrollCount"][id] = count;
  }

  // Serialize error
  for (auto &[inst, errVal] : error) {
    if (erasedValues.contains(inst))
      continue;
    std::string id = idValueMapping.findByValue(inst)->first;
    j["values"][id]["error"] = errVal;
  }

  // Serialize cmpError
  for (auto &[inst, cmpErr] : cmpError) {
    if (erasedValues.contains(inst))
      continue;
    std::string id = idValueMapping.findByValue(inst)->first;
    j["values"][id]["cmpError"] = cmpErr->serialize();
  }

  return j;
}

void TaffoInfo::deserialize(const json &j) {
  // Clear existing state
  transparentTypes.clear();
  startingPoints.clear();
  indirectFunctions.clear();
  oclTrampolines.clear();
  disabledConversion.clear();
  originalFunctions.clear();
  valueInfo.clear();
  valueWeights.clear();
  maxRecursionCount.clear();
  loopUnrollCount.clear();
  error.clear();
  cmpError.clear();

  // Deserialize idCounter
  idCounter = j["idCounter"].get<unsigned int>();

  // Deserialize startingPoints
  for (const auto &id : j["startingPoints"]) {
    std::string idStr = id.get<std::string>();
    auto iter = idValueMapping.find(idStr);
    if (iter != idValueMapping.end()) {
      Value *val = iter->second;
      if (auto *f = dyn_cast<Function>(val))
        startingPoints.push_back(f);
    }
  }

  // Deserialize indirectFunctions
  for (auto &item : j["indirectFunctions"].items()) {
    const std::string &callId = item.key();
    std::string funcId = item.value().get<std::string>();
    auto callIt = idValueMapping.find(callId);
    auto funcIt = idValueMapping.find(funcId);
    if (callIt != idValueMapping.end() && funcIt != idValueMapping.end()) {
      Value *callVal = callIt->second;
      Value *funcVal = funcIt->second;
      if (auto *callInst = dyn_cast<CallInst>(callVal))
        if (auto *f = dyn_cast<Function>(funcVal))
          indirectFunctions[callInst] = f;
    }
  }

  // Deserialize oclTrampolines
  for (auto &item : j["oclTrampolines"].items()) {
    const std::string &trampId = item.key();
    std::string kernFId = item.value().get<std::string>();
    auto trampIt = idValueMapping.find(trampId);
    auto kernFIt = idValueMapping.find(kernFId);
    if (trampIt != idValueMapping.end() && kernFIt != idValueMapping.end()) {
      Value *trampVal = trampIt->second;
      Value *kernFVal = kernFIt->second;
      if (auto *trampF = dyn_cast<Function>(trampVal))
        if (auto *kernF = dyn_cast<Function>(kernFVal))
          oclTrampolines[trampF] = kernF;
    }
  }

  // Deserialize disabledConversion
  for (const auto &id : j["disabledConversion"]) {
    std::string idStr = id.get<std::string>();
    auto iter = idValueMapping.find(idStr);
    if (iter != idValueMapping.end()) {
      Value *val = iter->second;
      if (auto *inst = dyn_cast<Instruction>(val))
        disabledConversion.push_back(inst);
    }
  }

  // Deserialize originalFunctions
  for (auto &item : j["originalFunctions"].items()) {
    const std::string &taffoId = item.key();
    std::string originalId = item.value().get<std::string>();
    auto taffoIt = idValueMapping.find(taffoId);
    auto originalIt = idValueMapping.find(originalId);
    if (originalIt != idValueMapping.end() && taffoIt != idValueMapping.end()) {
      Value *taffoVal = taffoIt->second;
      Value *orignalVal = originalIt->second;
      if (auto *taffoF = dyn_cast<Function>(taffoVal))
        if (auto *originalF = dyn_cast<Function>(orignalVal))
          originalFunctions[taffoF] = originalF;
    }
  }

  // Deserialize taffoFunctions
  for (auto &item : j["taffoFunctions"].items()) {
    const std::string &originalId = item.key();
    auto originalIt = idValueMapping.find(originalId);
    if (originalIt == idValueMapping.end())
      continue;
    Value *val = originalIt->second;
    if (auto *originalF = dyn_cast<Function>(val)) {
      for (const auto &taffoIdJson : item.value()) {
        std::string taffoId = taffoIdJson.get<std::string>();
        auto taffoIt = idValueMapping.find(taffoId);
        if (taffoIt != idValueMapping.end()) {
          Value *taffoVal = taffoIt->second;
          if (auto *taffoF = dyn_cast<Function>(taffoVal))
            this->taffoFunctions[originalF].insert(taffoF);
        }
      }
    }
  }

  // Deserialize values (valueInfo, deducedPointerType, valueWeights, constantUsers, error, cmpError)
  for (auto &item : j["values"].items()) {
    const std::string &id = item.key();
    auto iter = idValueMapping.find(id);
    if (iter == idValueMapping.end())
      continue;
    Value *val = iter->second;
    json valueJson = item.value();

    // Deserialize valueInfo
    if (valueJson.contains("info") && !valueJson["info"].is_null()) {
      std::string infoKind = valueJson["info"]["kind"].get<std::string>();
      std::shared_ptr<ValueInfo> vi;
      if (infoKind == "ScalarInfo")
        vi = std::make_shared<ScalarInfo>();
      else if (infoKind == "StructInfo")
        vi = std::make_shared<StructInfo>(0);
      vi->deserialize(valueJson["info"]);
      valueInfo[val] = vi;
    }

    // Deserialize deducedPointerTypes
    if (valueJson.contains("transparentType") && !valueJson["transparentType"].is_null())
      transparentTypes[val] = TransparentTypeFactory::create(valueJson["transparentType"]);

    // Deserialize valueWeights
    if (valueJson.contains("weight") && !valueJson["weight"].is_null()) {
      valueWeights[val] = valueJson["weight"].get<int>();
    }

    // Deserialize error
    if (valueJson.contains("error") && !valueJson["error"].is_null()) {
      if (auto *inst = dyn_cast<Instruction>(val))
        error[inst] = valueJson["error"].get<double>();
    }

    // Deserialize cmpError
    if (valueJson.contains("cmpError") && !valueJson["cmpError"].is_null()) {
      if (auto *inst = dyn_cast<Instruction>(val)) {
        auto cmpErr = std::make_shared<CmpErrorInfo>(0.0, true);
        cmpErr->deserialize(valueJson["cmpError"]);
        cmpError[inst] = cmpErr;
      }
    }
  }

  // Deserialize maxRecursionCount
  for (auto &item : j["maxRecursionCount"].items()) {
    const std::string &funcId = item.key();
    auto iter = idValueMapping.find(funcId);
    if (iter != idValueMapping.end()) {
      Value *val = iter->second;
      if (auto *f = dyn_cast<Function>(val))
        maxRecursionCount[f] = item.value().get<unsigned int>();
    }
  }

  // Deserialize loopUnrollCount
  for (auto &item : j["loopUnrollCount"].items()) {
    const std::string &loopId = item.key();
    auto iter = idLoopMapping.find(loopId);
    if (iter != idLoopMapping.end()) {
      Loop *loop = iter->second;
      loopUnrollCount[loop] = item.value().get<unsigned int>();
    }
  }
}
