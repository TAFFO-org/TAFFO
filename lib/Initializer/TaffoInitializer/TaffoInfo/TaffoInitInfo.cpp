#include "TaffoInitInfo.hpp"

using namespace llvm;
using namespace taffo;

ValueInitInfo *TaffoInitInfo::getValueInitInfo(llvm::Value *value) {
  assert(valueInitInfo.contains(value));
  return valueInitInfo.at(value);
}

ValueInitInfo *TaffoInitInfo::createValueInitInfo(llvm::Value *value) {

}
