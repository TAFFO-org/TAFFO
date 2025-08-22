#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInitInfo.hpp"

using namespace llvm;
using namespace taffo;

ValueInitInfo& TaffoInitInfo::getValueInitInfo(const Value* value) {
  assert(valueInitInfo.contains(value));
  return valueInitInfo.find(value)->second;
}

ValueInitInfo& TaffoInitInfo::getOrCreateValueInitInfo(Value* value) {
  return valueInitInfo.contains(value) ? valueInitInfo.find(value)->second : createValueInitInfo(value);
}

ValueInitInfo& TaffoInitInfo::createValueInitInfo(Value* value, unsigned rootDistance, unsigned backtrackingDepth) {
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  assert(taffoInfo.hasValueInfo(*value) && "Creating a valueInitInfo of a value without valueInfo");
  ValueInitInfo newValueInitInfo = ValueInitInfoFactory::createValueInitInfo(rootDistance, backtrackingDepth);
  valueInitInfo.insert({value, newValueInitInfo});
  return valueInitInfo.find(value)->second;
}

bool TaffoInitInfo::hasValueInitInfo(const Value* value) const { return valueInitInfo.contains(value); }
