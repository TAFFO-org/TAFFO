#include "TaffoInitInfo.hpp"
#include "TaffoInfo/TaffoInfo.hpp"

using namespace llvm;
using namespace taffo;

ValueInitInfo &TaffoInitInfo::getValueInitInfo(llvm::Value *value) {
  assert(valueInitInfo.contains(value));
  return valueInitInfo[value];
}

ValueInitInfo &TaffoInitInfo::createValueInitInfo(llvm::Value *value, unsigned int rootDistance, unsigned int backtrackingDepthLeft  ) {
    auto &taffoInfo = TaffoInfo::getInstance();
    assert(taffoInfo.hasValueInfo(*value) && "creating a ValueInitInfo of a a value without ValueInfo");
    ValueInfo * valueInfo = &*taffoInfo.getValueInfo(*value);
    valueInitInfo.insert({value, {valueInfo, rootDistance, backtrackingDepthLeft}});
    return valueInitInfo[value];
}
