#include "TaffoConvInfo.hpp"
#include "TaffoInfo/TaffoInfo.hpp"

using namespace llvm;
using namespace tda;
using namespace taffo;

ValueConvInfo* TaffoConvInfo::createValueConvInfo(Value* value, const ConversionType* oldConvType) {
  auto iter = valueConvInfo.find(value);
  assert(iter == valueConvInfo.end() && "value already has valueConvInfo!");
  TransparentType* type = TaffoInfo::getInstance().getOrCreateTransparentType(*value);
  bool isConstant = isa<Constant>(value) && !isa<GlobalVariable>(value) && !isa<Function>(value);
  std::unique_ptr<ValueConvInfo> newConversionInfo =
    oldConvType ? std::make_unique<ValueConvInfo>(oldConvType->clone(*type), isConstant)
                : std::make_unique<ValueConvInfo>(*type, isConstant);
  return (valueConvInfo[value] = std::move(newConversionInfo)).get();
}

ValueConvInfo* TaffoConvInfo::getOrCreateValueConvInfo(Value* value, const ConversionType* oldConvType, bool* isNew) {
  auto iter = valueConvInfo.find(value);
  if (iter == valueConvInfo.end()) {
    if (isNew)
      *isNew = true;
    return createValueConvInfo(value, oldConvType);
  }
  if (isNew)
    *isNew = false;
  return iter->getSecond().get();
}

ValueConvInfo* TaffoConvInfo::getValueConvInfo(Value* value) const {
  auto iter = valueConvInfo.find(value);
  assert(iter != valueConvInfo.end() && "valueConvInfo not present");
  return iter->getSecond().get();
}

bool TaffoConvInfo::hasValueConvInfo(const Value* value) const { return valueConvInfo.contains(value); }
