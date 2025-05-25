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
  assert(taffoInfo.hasValueInfo(*value) && "Creating a ValueInitInfo of a value without ValueInfo");
  ValueInitInfo newValueInitInfo = ValueInitInfoFactory::createValueInitInfo(rootDistance, backtrackingDepth);
  auto [_, success] = valueInitInfo.insert({value, newValueInitInfo});
  assert(success && "ValueInitInfo already exists");
  return valueInitInfo.find(value)->second;
}

bool TaffoInitInfo::hasValueInitInfo(const Value* value) const { return valueInitInfo.contains(value); }
