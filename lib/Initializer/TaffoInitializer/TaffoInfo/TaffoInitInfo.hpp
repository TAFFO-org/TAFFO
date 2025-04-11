#ifndef TAFFO_TAFFO_INIT_INFO_HPP
#define TAFFO_TAFFO_INIT_INFO_HPP

#include "ValueInitInfo.hpp"

namespace taffo {

class TaffoInitInfo {
public:
  ValueInitInfo *getValueInitInfo(llvm::Value *value);
  ValueInitInfo *createValueInitInfo(llvm::Value *value);

private:
  llvm::DenseMap<llvm::Value*, ValueInitInfo*> valueInitInfo;
};

}

#endif // TAFFO_TAFFO_INIT_INFO_HPP
