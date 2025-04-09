#ifndef TAFFO_TRANSPARENT_TYPE_HPP
#define TAFFO_TRANSPARENT_TYPE_HPP

#include "SerializationUtils.hpp"
#include "PtrCasts.hpp"

#include <llvm/IR/DerivedTypes.h>

namespace flttofix {

class FixedPointScalarType;
class FixedPointStructType;

} // namespace flttofix

namespace taffo {

class TransparentType;

class TransparentTypeFactory {
private:
  friend class TransparentType;
  friend class TransparentStructType;
  friend class TypeDeducerPass;
  friend class TaffoInfo;

  static std::shared_ptr<TransparentType> create(const llvm::Value *value);
  static std::shared_ptr<TransparentType> create(llvm::Type *unwrappedType, unsigned int indirections);
  static std::shared_ptr<TransparentType> create(json j);
};

class TransparentType : public Serializable, public Printable {
public:
  friend class TransparentTypeFactory;
  friend class TypeDeducerPass;
  friend class flttofix::FixedPointScalarType;

  enum TransparentTypeKind {
    K_Scalar,
    K_Struct
  };

  static bool classof(const TransparentType *type) { return type->getKind() == K_Scalar; }

  bool isValid() const { return unwrappedType; }
  llvm::Type *getUnwrappedType() const { return unwrappedType; }
  unsigned int getIndirections() const { return indirections; }
  bool isStructType() const { return unwrappedType->isStructTy(); }
  bool isFloatType() const { return unwrappedType->isFloatTy(); }
  bool isPointerType() const { return indirections > 0 || isOpaquePointer(); }
  virtual bool isOpaquePointer() const { return unwrappedType->isPointerTy(); }
  virtual int compareTransparency(const TransparentType &other) const;
  std::shared_ptr<TransparentType> getPointerElementType() const;
  llvm::Type* toLLVMType() const;
  virtual TransparentTypeKind getKind() const { return K_Scalar; }

  virtual bool operator==(const TransparentType &other) const;
  bool operator!=(const TransparentType &other) const { return !(*this == other); }

  virtual std::shared_ptr<TransparentType> clone() const;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;
  
protected:
  llvm::Type *unwrappedType = nullptr;
  unsigned int indirections = 0;

  TransparentType() = default;
  TransparentType(const TransparentType &other) = default;

  TransparentType(llvm::Type *unwrappedType, unsigned int indirections)
  : unwrappedType(unwrappedType), indirections(indirections) {}

  void incrementIndirections(int increment);
};

class TransparentStructType : public TransparentType {
private:
  friend class TransparentTypeFactory;
  friend class TypeDeducerPass;
  friend class flttofix::FixedPointStructType;

  llvm::SmallVector<std::shared_ptr<TransparentType>, 2> fieldTypes;

  TransparentStructType() = default;
  TransparentStructType(const TransparentStructType &other) = default;

  TransparentStructType(llvm::StructType *unwrappedType, unsigned int indirections)
  : TransparentType(unwrappedType, indirections) {
    for (llvm::Type *fieldType : unwrappedType->elements())
      fieldTypes.push_back(TransparentTypeFactory::create(fieldType, 0));
  }

  void setFieldType(unsigned int i, std::shared_ptr<TransparentType> fieldType) { fieldTypes[i] = fieldType; }

public:
  static bool classof(const TransparentType *type) { return type->getKind() == K_Struct; }

  auto begin() { return fieldTypes.begin(); }
  auto end() { return fieldTypes.end(); }
  auto begin() const { return fieldTypes.begin(); }
  auto end() const { return fieldTypes.end(); }

  bool isOpaquePointer() const override;
  int compareTransparency(const TransparentType &other) const override;
  std::shared_ptr<TransparentType> getFieldType(unsigned int i) const { return fieldTypes[i]; }
  unsigned int getNumFieldTypes() const { return fieldTypes.size(); }
  TransparentTypeKind getKind() const override { return K_Struct; }

  bool operator==(const TransparentType &other) const override;

  std::shared_ptr<TransparentType> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;
};

} // namespace taffo

namespace std {

template <>
struct std::hash<std::shared_ptr<taffo::TransparentType>> {
  std::size_t operator()(const std::shared_ptr<taffo::TransparentType> &ptr) const {
    if (!ptr)
      return 0;

    std::size_t combined = 0;
    auto combine = [](std::size_t seed, std::size_t value) {
      return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    };

    combined = combine(combined, std::hash<llvm::Type *>()(ptr->getUnwrappedType()));
    combined = combine(combined, std::hash<unsigned int>()(ptr->getIndirections()));

    if (auto structPtr = std::dynamic_ptr_cast<taffo::TransparentStructType>(ptr))
      for (const auto &field : *structPtr)
        combined = combine(combined, std::hash<std::shared_ptr<taffo::TransparentType>>()(field));

    return combined;
  }
};

template <>
struct std::equal_to<std::shared_ptr<taffo::TransparentType>> {
  bool operator()(const std::shared_ptr<taffo::TransparentType> &lhs,
                  const std::shared_ptr<taffo::TransparentType> &rhs) const {
    if (lhs == rhs)
      return true;
    if (!lhs || !rhs)
      return false;
    return *lhs == *rhs;
  }
};

} // namespace std

#endif // TAFFO_TRANSPARENT_TYPE_HPP
