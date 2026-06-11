#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInitInfo.hpp"

using namespace llvm;
using namespace taffo;

ValueInitInfo& TaffoInitInfo::getValueInitInfo(const Value* value) {
  auto it = valueInitInfo.find(value);
  assert(it != valueInitInfo.end());
  return it->second;
}

ValueInitInfo& TaffoInitInfo::getOrCreateValueInitInfo(Value* value) {
  auto it = valueInitInfo.find(value);
  return it != valueInitInfo.end() ? valueInitInfo.find(value)->second : createValueInitInfo(value);
}

ValueInitInfo& TaffoInitInfo::createValueInitInfo(Value* value, unsigned rootDistance) {
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  assert(taffoInfo.hasValueInfo(*value) && "Creating a valueInitInfo of a value without valueInfo");
  ValueInitInfo newValueInitInfo = ValueInitInfoFactory::createValueInitInfo(rootDistance);
  valueInitInfo.insert({value, newValueInitInfo});
  return valueInitInfo.find(value)->second;
}

bool TaffoInitInfo::hasValueInitInfo(const Value* value) const {
  return valueInitInfo.find(value) != valueInitInfo.end();
}
