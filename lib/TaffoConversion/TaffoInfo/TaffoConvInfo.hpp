#pragma once

#include "ValueConvInfo.hpp"

namespace taffo {

class ConversionPass;

class TaffoConvInfo {
  friend class ConversionPass;

  llvm::DenseMap<llvm::Value*, std::unique_ptr<ValueConvInfo>> valueConvInfo;

  ValueConvInfo* createValueConvInfo(llvm::Value* value, const ConversionType* oldConvType = nullptr);
  ValueConvInfo* getValueConvInfo(llvm::Value* value) const;
  ValueConvInfo*
  getOrCreateValueConvInfo(llvm::Value* value, const ConversionType* oldConvType = nullptr, bool* isNew = nullptr);
  bool hasValueConvInfo(const llvm::Value* value) const;

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getOrCreateCurrentType(llvm::Value* value) {
    ValueConvInfo* valueConvInfo = hasValueConvInfo(value) ? getValueConvInfo(value) : createValueConvInfo(value);
    return valueConvInfo->getCurrentType<T>();
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getCurrentType(llvm::Value* value) {
    return getValueConvInfo(value)->getCurrentType<T>();
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getNewType(llvm::Value* value) const {
    return getValueConvInfo(value)->getNewType<T>();
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getNewOrOldType(llvm::Value* value) const {
    return getValueConvInfo(value)->getNewOrOldType<T>();
  }
};

} // namespace taffo
