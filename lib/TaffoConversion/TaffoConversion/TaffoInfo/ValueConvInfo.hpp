#pragma once

#include "../Types/ConversionType.hpp"
#include "TransparentType.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Value.h>

namespace taffo {

struct ValueConvInfo : tda::Printable {
  bool isConverted = false;
  bool isConversionDisabled = true;
  bool isArgumentPlaceholder = false;
  bool isBacktrackingNode = false;
  bool isRoot = false;
  llvm::SmallPtrSet<llvm::Value*, 5> roots;

  ValueConvInfo(std::unique_ptr<ConversionType> type)
  : oldType(std::move(type)) {}

  ValueConvInfo(const tda::TransparentType& type)
  : oldType(ConversionTypeFactory::create(type)) {}

  ValueConvInfo& operator=(const ValueConvInfo& other) {
    isConverted = other.isConverted;
    isConversionDisabled = other.isConversionDisabled;
    isArgumentPlaceholder = other.isArgumentPlaceholder;
    isBacktrackingNode = other.isBacktrackingNode;
    isRoot = other.isRoot, roots = other.roots;
    oldType = other.oldType->clone();
    newType = other.newType ? other.newType->clone() : nullptr;
    return *this;
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getOldType() const {
    return static_cast<T*>(oldType.get());
  }

  void setNewType(std::unique_ptr<ConversionType> type) { this->newType = std::move(type); }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getNewType() const {
    return static_cast<T*>(newType ? newType.get() : oldType.get());
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getCurrentType() const {
    return static_cast<T*>(isConverted ? newType.get() : oldType.get());
  }

  std::string toString() const override;

private:
  std::unique_ptr<ConversionType> oldType;
  std::unique_ptr<ConversionType> newType;
};

} // namespace taffo
