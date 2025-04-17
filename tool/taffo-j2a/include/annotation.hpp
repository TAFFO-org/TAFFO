#pragma once
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>

#include <variant>

namespace taffo {
class Annotation;
using StringOrAnnotation = std::variant<std::string, Annotation>;
using StructAnnotation = std::vector<StringOrAnnotation>;
class Annotation {
public:
  Annotation() = default;
  Annotation(const llvm::json::Object& object);
  bool targetEnabled() const { return target; }
  const std::string& getTargetName() const { return targetName; }

  void setTargetString(std::string trg) {
    targetName = std::move(trg);
    target = true;
  }

  void setTarget(bool targetEnabled) {
    targetName = "";
    target = targetEnabled;
  }

  bool backtrackingEnabled() const { return backtracking; }
  void enableBacktracking(bool enabled) { backtracking = enabled; }
  const std::string& getRangeMax() const { return maxRange; }
  const std::string& getRangeMin() const { return minRange; }

  void enableRange(double rMin, double rMax) {
    maxRange = std::to_string(rMax);
    minRange = std::to_string(rMin);
    range = true;
  }

  void enableRange(std::string rMin, std::string rMax) {
    maxRange = std::move(rMax);
    minRange = std::move(rMin);
    range = true;
  }

  bool rangeEnabled() const { return range; }
  void enableRange(bool enabled) { range = enabled; }

  bool typeEnabled() const { return type; }
  bool getTypeSign() const { return typeSigned; }
  int getBitSize() const { return bitsSize; }
  int getFractionalPos() const { return fractionalPos; }

  void setType(int newTypeBitSize, int newFractionalPos, bool newTypeSign = true) {
    type = true;
    typeSigned = newTypeSign;
    bitsSize = newTypeBitSize;
    fractionalPos = newFractionalPos;
  }

  void disableType() { type = false; }

  bool errorEnabled() const { return error; }
  void setErrorValue(std::string newErrorValue) {
    errorValue = newErrorValue;
    error = true;
  }

  void enableError(bool errorEnabled) { error = errorEnabled; }

  bool finalEnabled() const { return finalActive; }

  void enableFinal(bool newFinalEnabled) { finalActive = newFinalEnabled; }

  void toJSON(std::ostream& stream) const;

  void serialize(std::ostream& stream, const llvm::StringMap<StructAnnotation>* symbolTable = nullptr) const;

  void setEnabled(bool enabled) { disabled = !enabled; }

  bool isEnabled() const { return !disabled; }

  bool isStruct() const { return !strct.empty(); }

  void setStructName(std::string stuctName) { strct = std::move(stuctName); }

  const std::string& structName() const { return strct; }

  std::string getSignStr() const {
    if (typeSigned)
      return "signed";
    return "unsigned";
  }

private:
  bool target {false};
  std::string targetName {""};
  bool backtracking {false};
  bool range {false};
  std::string maxRange {"0"};
  std::string minRange {"0"};
  bool type {false};
  bool typeSigned {true};
  int bitsSize {0};
  int fractionalPos {0};
  bool error {false};
  std::string errorValue {""};
  bool disabled {false};
  bool finalActive {false};
  std::string strct {""};
};
} // namespace taffo
