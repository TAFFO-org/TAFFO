#include "TaffoInfo.hpp"
#include "ValueInfo.hpp"

using namespace llvm;
using namespace tda;
using namespace taffo;

std::shared_ptr<ValueInfo> ValueInfoFactory::create(Value* value) {
  TransparentType* type = TaffoInfo::getInstance().getOrCreateTransparentType(*value);
  return create(type);
}

std::shared_ptr<ValueInfo> ValueInfoFactory::create(const TransparentType* type) {
  std::unordered_map<const TransparentType*, std::shared_ptr<StructInfo>> recursionMap;
  return create(type, recursionMap);
}

std::shared_ptr<ValueInfo>
ValueInfoFactory::create(const TransparentType* type,
                         std::unordered_map<const TransparentType*, std::shared_ptr<StructInfo>>& recursionMap) {
  auto iter = recursionMap.find(type);
  if (iter != recursionMap.end())
    return iter->second;

  if (auto* structType = dyn_cast<TransparentStructType>(type)) {
    unsigned numFields = structType->getNumFieldTypes();
    SmallVector<std::shared_ptr<ValueInfo>, 4> fields;
    auto res = std::make_shared<StructInfo>(StructInfo(numFields));
    recursionMap.insert({structType, res});
    for (unsigned i = 0; i < numFields; i++)
      res->setField(i, create(structType->getFieldType(i), recursionMap));
    return res;
  }

  return std::make_shared<ScalarInfo>();
}

json ValueInfo::serialize() const {
  json j = json::object();
  if (target.has_value())
    j["target"] = target.value();
  if (bufferId.has_value())
    j["bufferId"] = bufferId.value();
  return j;
}

void ValueInfo::deserialize(const json& j) {
  if (j.contains("target") && !j["target"].is_null())
    target = j["target"].get<std::string>();
  if (j.contains("bufferId") && !j["bufferId"].is_null())
    bufferId = j["bufferId"].get<std::string>();
}

void ValueInfo::copyFrom(const ValueInfo& other) {
  assert(getKind() == other.getKind() && "Copying from different kind of valueInfo");
  target = other.target;
  bufferId = other.bufferId;
}

ScalarInfo& ScalarInfo::operator=(const ScalarInfo& other) {
  if (this != &other) {
    copyFrom(other);
    this->numericType = other.numericType;
    this->range = other.range;
    this->error = other.error;
    this->conversionEnabled = other.conversionEnabled;
    this->final = other.final;
  }
  return *this;
}

void ScalarInfo::copyFrom(const ValueInfo& other) {
  ValueInfo::copyFrom(other);
  auto& otherScalar = cast<ScalarInfo>(other);
  numericType = otherScalar.numericType ? otherScalar.numericType->clone() : nullptr;
  range = otherScalar.range ? otherScalar.range->clone() : nullptr;
  error = otherScalar.error ? std::make_shared<double>(*otherScalar.error) : nullptr;
  conversionEnabled = otherScalar.conversionEnabled;
  final = otherScalar.final;
}

std::shared_ptr<ValueInfo> ScalarInfo::cloneImpl() const {
  auto copy = std::make_shared<ScalarInfo>();
  copy->copyFrom(*this);
  return copy;
}

std::string ScalarInfo::toString() const {
  std::stringstream ss;
  ss << "scalar(";
  bool first = true;
  if (numericType) {
    first = false;
    ss << "type(" << numericType->toString() << ")";
  }
  if (range) {
    if (!first)
      ss << " ";
    else
      first = false;
    ss << "range(" << range->min << ", " << range->max << ")";
  }
  if (error) {
    if (!first)
      ss << " ";
    ss << "error(" << *error << ")";
  }
  if (!conversionEnabled) {
    if (!first)
      ss << " ";
    ss << "disabled";
  }
  if (final) {
    if (!first)
      ss << " ";
    ss << "final";
  }
  ss << ")";
  return ss.str();
}

json ScalarInfo::serialize() const {
  json j;
  j["kind"] = "ScalarInfo";
  j.update(ValueInfo::serialize());
  if (numericType)
    j["numericType"] = numericType->serialize();
  if (range)
    j["range"] = range->serialize();
  if (error)
    j["error"] = *error;
  j["conversionEnabled"] = conversionEnabled;
  j["final"] = final;
  return j;
}

void ScalarInfo::deserialize(const json& j) {
  ValueInfo::deserialize(j);
  if (j.contains("numericType") && !j["numericType"].is_null()) {
    std::string numericTypeKind = j["numericType"]["kind"].get<std::string>();
    if (numericTypeKind == "FixedPoint") {
      numericType = std::make_shared<FixedPointInfo>(false, 0, 0);
      numericType->deserialize(j["numericType"]);
    }
    else if (numericTypeKind == "FloatingPoint") {
      numericType = std::make_shared<FloatingPointInfo>(FloatingPointInfo::Float_float, 0.0);
      numericType->deserialize(j["numericType"]);
    }
  }
  if (j.contains("range") && !j["range"].is_null()) {
    range = std::make_shared<Range>();
    range->deserialize(j["range"]);
  }
  if (j.contains("error") && !j["error"].is_null())
    error = std::make_shared<double>(j["error"].get<double>());
  conversionEnabled = j["conversionEnabled"].get<bool>();
  final = j["final"].get<bool>();
}

bool StructInfo::isConversionEnabled() const {
  SmallPtrSet<const StructInfo*, 1> visited;
  return isConversionEnabled(visited);
}

bool StructInfo::isConversionEnabled(SmallPtrSetImpl<const StructInfo*>& visited) const {
  visited.insert(this);
  for (const auto& field : Fields) {
    if (!field)
      continue;
    if (auto* si = dyn_cast<StructInfo>(field.get())) {
      if (visited.count(si) > 0)
        continue;
      if (si->isConversionEnabled(visited))
        return true;
    }
    else {
      if (field->isConversionEnabled())
        return true;
    }
  }
  return false;
}

std::shared_ptr<ValueInfo> StructInfo::resolveFromIndexList(Type* type, ArrayRef<unsigned> indices) const {
  Type* resolvedType = type;
  std::shared_ptr<ValueInfo> resolvedInfo = this->clone();
  for (unsigned idx : indices) {
    if (resolvedInfo == nullptr)
      break;
    if (resolvedType->isStructTy()) {
      resolvedType = resolvedType->getContainedType(idx);
      resolvedInfo = std::static_ptr_cast<StructInfo>(resolvedInfo)->getField(idx);
    }
    else
      resolvedType = resolvedType->getContainedType(idx);
  }
  return resolvedInfo;
}

void StructInfo::copyFrom(const ValueInfo& other) {
  ValueInfo::copyFrom(other);
  auto& otherStruct = cast<StructInfo>(other);
  assert(getNumFields() == otherStruct.getNumFields() && "Copying from structInfo with different number of fields");
  for (auto&& [field, otherField] : zip(*this, otherStruct)) {
    if (otherField)
      if (field)
        field->copyFrom(*otherField);
      else
        field = otherField->clone();
    else
      field = nullptr;
  }
}

std::shared_ptr<ValueInfo> StructInfo::cloneImpl() const {
  SmallVector<std::shared_ptr<ValueInfo>, 4> newFields;
  for (const std::shared_ptr<ValueInfo>& field : Fields)
    if (field)
      newFields.push_back(field->clone());
    else
      newFields.push_back(nullptr);
  return std::make_shared<StructInfo>(newFields);
}

std::string StructInfo::toString() const {
  std::stringstream ss;
  ss << "struct(";
  bool first = true;
  for (const std::shared_ptr<ValueInfo>& field : Fields) {
    if (!first)
      ss << ", ";
    if (field)
      ss << field->toString();
    else
      ss << "void()";
    first = false;
  }
  ss << ")";
  return ss.str();
}

json StructInfo::serialize() const {
  json j;
  j["kind"] = "StructInfo";
  j.update(ValueInfo::serialize());
  j["fields"] = json::array();
  for (const auto& field : Fields)
    if (field)
      j["fields"].push_back(field->serialize());
    else
      j["fields"].push_back(nullptr);
  return j;
}

void StructInfo::deserialize(const json& j) {
  ValueInfo::deserialize(j);
  if (!j.contains("fields") || !j["fields"].is_array())
    report_fatal_error("StructInfo::deserialize: Missing or invalid fields array");
  Fields.clear();
  for (auto& fieldJson : j["fields"]) {
    if (fieldJson.is_null()) {
      Fields.push_back(nullptr);
    }
    else {
      std::string fieldKind = fieldJson["kind"].get<std::string>();
      if (fieldKind == "ScalarInfo") {
        auto field = std::make_shared<ScalarInfo>(nullptr);
        field->deserialize(fieldJson);
        Fields.push_back(field);
      }
      else if (fieldKind == "StructInfo") {
        auto field = std::make_shared<StructInfo>(0);
        field->deserialize(fieldJson);
        Fields.push_back(field);
      }
      else {
        report_fatal_error(StringRef("StructInfo::deserialize: Unknown field kind: " + fieldKind));
      }
    }
  }
}

std::shared_ptr<ValueInfoWithRange> PointerInfo::getUnwrappedInfo() const {
  if (!pointed)
    return nullptr;
  if (std::shared_ptr<PointerInfo> pointedPointerInfo = std::dynamic_ptr_cast<PointerInfo>(pointed))
    return pointedPointerInfo->getUnwrappedInfo();
  return std::static_ptr_cast<ValueInfoWithRange>(pointed);
}

bool PointerInfo::isConversionEnabled() const {
  return true; // TODO
}

std::shared_ptr<ValueInfo> PointerInfo::cloneImpl() const {
  std::shared_ptr<PointerInfo> copy = std::make_shared<PointerInfo>(pointed);
  copy->copyFrom(*this);
  return copy;
}

std::string PointerInfo::toString() const {
  return "";                     // TODO
}

json PointerInfo::serialize() const {
  return ValueInfo::serialize(); // TODO
}

void PointerInfo::deserialize(const json& j) {
  ValueInfo::deserialize(j);     // TODO
}

bool GEPInfo::isConversionEnabled() const {
  return true;                   // TODO
}

std::shared_ptr<ValueInfo> GEPInfo::cloneImpl() const {
  std::shared_ptr<GEPInfo> copy = std::make_shared<GEPInfo>(pointed, offset);
  copy->copyFrom(*this);
  return copy;
}

std::string GEPInfo::toString() const {
  return "";                     // TODO
}

json GEPInfo::serialize() const {
  return ValueInfo::serialize(); // TODO
}

void GEPInfo::deserialize(const json& j) {
  ValueInfo::deserialize(j);     // TODO
}

json CmpErrorInfo::serialize() const {
  json j;
  j["MaxTolerance"] = MaxTolerance;
  j["MayBeWrong"] = MayBeWrong;
  return j;
}

void CmpErrorInfo::deserialize(const json& j) {
  MaxTolerance = j["MaxTolerance"].get<double>();
  MayBeWrong = j["MayBeWrong"].get<bool>();
}
