#pragma once

#include "../Types/ConversionType.hpp"
#include "TransparentType.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Value.h>

namespace taffo {

struct ValueConvInfo : tda::Printable {
  bool isArgumentPlaceholder = false;
  bool isBacktrackingNode = false;
  bool isRoot = false;
  llvm::SmallPtrSet<llvm::Value*, 5> roots;

  ValueConvInfo(std::unique_ptr<ConversionType> type, bool isConstant)
  : oldType(std::move(type)), constant(isConstant), conversionDisabled(!isConstant) {}

  ValueConvInfo(const tda::TransparentType& type, bool isConstant)
  : oldType(ConversionTypeFactory::create(type)), constant(isConstant), conversionDisabled(!isConstant) {}

  ValueConvInfo& operator=(const ValueConvInfo& other) {
    isArgumentPlaceholder = other.isArgumentPlaceholder;
    isBacktrackingNode = other.isBacktrackingNode;
    isRoot = other.isRoot, roots = other.roots;
    oldType = other.oldType->clone();
    newType = other.newType ? other.newType->clone() : nullptr;
    constant = other.constant;
    conversionDisabled = other.conversionDisabled;
    isConverted = other.isConverted;
    return *this;
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getOldType() const {
    return static_cast<T*>(oldType.get());
  }

  void setNewType(std::unique_ptr<ConversionType> type) {
    assert(!constant && "Cannot set newType for a constant because they are uniqued");
    this->newType = std::move(type);
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getNewType() const {
    assert(!constant && "Cannot get newType for a constant because they are uniqued");
    return static_cast<T*>(newType ? newType.get() : oldType.get());
  }

  template <std::derived_from<ConversionType> T = ConversionType>
  T* getCurrentType() const {
    return static_cast<T*>(isConverted ? newType.get() : oldType.get());
  }

  bool isConstant() const { return constant; }

  bool isConversionDisabled() const { return conversionDisabled; }
  void enableConversion() { conversionDisabled = false; }

  void setConverted() {
    assert(!constant && "Cannot set converted for a constant because they are uniqued");
    isConverted = true;
  }

  std::string toString() const override;

private:
  std::unique_ptr<ConversionType> oldType;
  std::unique_ptr<ConversionType> newType;
  bool constant;
  bool conversionDisabled;
  bool isConverted = false;
};

} // namespace taffo
