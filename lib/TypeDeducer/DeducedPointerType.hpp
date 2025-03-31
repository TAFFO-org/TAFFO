#ifndef TAFFO_DEDUCED_POINTER_TYPE_HPP
#define TAFFO_DEDUCED_POINTER_TYPE_HPP

#include "SerializationUtils.hpp"

namespace taffo {

struct DeducedPointerType : Serializable, Printable {
  llvm::Type *unwrappedType = nullptr;
  unsigned int indirections = 0;

  DeducedPointerType() = default;
  DeducedPointerType(llvm::Type *unwrappedType, unsigned int indirections)
  : unwrappedType(unwrappedType), indirections(indirections) {}

  bool isAmbiguous() const { return unwrappedType == nullptr; }
  bool isOpaque() const { return !isAmbiguous() && unwrappedType->isPointerTy(); }

  bool operator==(const DeducedPointerType &other) const { return unwrappedType == other.unwrappedType && indirections == other.indirections; }
  bool operator!=(const DeducedPointerType &other) const { return !(*this == other); }
  bool operator<(const DeducedPointerType &other) const { return unwrappedType < other.unwrappedType || (unwrappedType == other.unwrappedType && indirections < other.indirections); }
  bool operator>(const DeducedPointerType &other) const { return other < *this; }

  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;
};

} // namespace taffo

#endif // TAFFO_DEDUCED_POINTER_TYPE_HPP
