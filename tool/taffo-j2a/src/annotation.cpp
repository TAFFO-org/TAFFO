#include "annotation.hpp"
#include "taffoAnnotations.hpp"

namespace taffo {
Annotation::Annotation(const llvm::json::Object& object) {
  auto target = object.getString("target");
  auto backtracking = object.getNumber("backtracking");
  auto rangeMax = object.getNumber("rangeMax");
  auto rangeMin = object.getNumber("rangeMin");
  auto rangeMaxStr = object.getString("rangeMax");
  auto rangeMinStr = object.getString("rangeMin");
  auto typeSign = object.getString("typeSign");
  auto typeBiteSize = object.getNumber("bitsSize");
  auto fractionalPos = object.getNumber("fractionalPos");
  auto finalEnabled = object.getNumber("final");
  auto error = object.getString("error");
  auto errorNum = object.getNumber("error");
  auto disabled = object.getNumber("disabled");
  auto strc = object.getString("struct");

  std::string rmxStr;
  std::string rmnStr;
  bool maxExist(false);
  bool minExist(false);

  if (rangeMax.hasValue()) {
    rmxStr = std::to_string(rangeMax.getValue());
    maxExist = true;
  }

  if (rangeMin.hasValue()) {
    rmnStr = std::to_string(rangeMin.getValue());
    minExist = true;
  }

  if (rangeMaxStr.hasValue()) {
    rmxStr = rangeMaxStr.getValue();
    maxExist = true;
  }

  if (rangeMinStr.hasValue()) {
    rmnStr = rangeMinStr.getValue();
    minExist = true;
  }

  if (minExist != maxExist)
    llvm::errs() << "only one of range min and max was defined\n";
  else if (minExist)
    enableRange(rmnStr, rmxStr);

  if (target.hasValue())
    setTargetString(target->str());

  if (backtracking.hasValue())
    enableBacktracking(backtracking.getValue() != 0.0);

  if (error.hasValue())
    setErrorValue(error.getValue());

  if (errorNum.hasValue())
    setErrorValue(std::to_string(errorNum.getValue()));

  if (finalEnabled.hasValue())
    enableFinal(finalEnabled.getValue() != 0.0);

  if (strc.hasValue())
    setStructName(strc.getValue());

  bool sign(true);
  if (typeSign.hasValue() && typeSign.getValue() == "unsigned")
    sign = false;

  if (typeBiteSize.hasValue() != fractionalPos.hasValue()) {
    llvm::errs() << "only bits size or fractional pos was defined\n";
    return;
  }

  if (typeBiteSize.hasValue())
    setType(typeBiteSize.getValue(), fractionalPos.getValue(), sign);

  if (disabled.hasValue())
    setEnabled(disabled.getValue() == 0);
}

void Annotation::toJSON(std::ostream& stream) const {
  if (target)
    stream << "\"target\": \"" << targetName << "\",\n";

  if (backtracking)
    stream << "\"backtracking\": 1,\n";

  if (range) {
    stream << "\"rangeMax\": " << maxRange << ",\n";
    stream << "\"rangeMin\": " << minRange << ",\n";
  }

  if (disabled)
    stream << "\"disabled\": 1,\n";

  if (finalActive)
    stream << "\"final\": 1,\n";

  if (type) {
    stream << "\"typeSign\": \"" << getSignStr() << "\",\n";
    stream << "\"bitsSize\": " << std::to_string(bitsSize) << ",\n";
    stream << "\"fractionalPos\":" << std::to_string(fractionalPos) << ",\n";
  }

  if (error)
    stream << "\"error\": " << errorValue << ",\n";

  if (!strct.empty())
    stream << "\nstruct\n:" << strct << ",\n";
  stream << "\"term \": 0\n";
}
void Annotation::serialize(std::ostream& stream, const llvm::StringMap<StructAnnotation>* symbolTable) const {
  if (target)
    stream << "target('" << targetName << "') ";

  if (backtracking)
    stream << "backtracking ";

  if (isStruct() && symbolTable != nullptr) {
    auto child = symbolTable->find(strct);
    if (child != symbolTable->end())
      taffo::serialize(stream, child->getValue(), *symbolTable);
  }
  else {
    stream << "scalar(";

    if (range)
      stream << "range(" << minRange << ", " << maxRange << ") ";

    if (type)
      stream << "type(" << getSignStr() << " " << std::to_string(bitsSize) << " " << std::to_string(fractionalPos)
             << ") ";

    if (error)
      stream << "error(" << errorValue << ") ";

    if (disabled)
      stream << "disabled ";

    if (finalActive)
      stream << "final ";
    stream << ")";
  }
}
} // namespace taffo
