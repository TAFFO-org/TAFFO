#pragma once

#include "PtrCasts.hpp"
#include "SerializationUtils.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/DerivedTypes.h>

#include <memory>

namespace taffo {

class FixedPointScalarType;
class FixedPointStructType;

} // namespace taffo

namespace taffo {

class TransparentType;

class TransparentTypeFactory {
public:
  static std::shared_ptr<TransparentType> create(llvm::Type* type);

private:
  friend class TransparentType;
  friend class TransparentArrayType;
  friend class TransparentStructType;
  friend class TypeDeducerPass;
  friend class TaffoInfo;

  static std::shared_ptr<TransparentType> create(const llvm::Value* value);
  static std::shared_ptr<TransparentType> create(llvm::Type* unwrappedType, unsigned int indirections);
  static std::shared_ptr<TransparentType> create(const json& j);
};

class TransparentType : public Serializable,
                        public Printable {
public:
  friend class TransparentTypeFactory;
  friend class TypeDeducerPass;
  friend class taffo::FixedPointScalarType;

  enum TransparentTypeKind {
    K_Scalar,
    K_Array,
    K_Struct
  };

  static bool classof(const TransparentType* type) { return type->getKind() == K_Scalar; }

  bool isValid() const { return unwrappedType; }
  llvm::Type* getUnwrappedType() const { return unwrappedType; }
  unsigned int getIndirections() const { return indirections; }
  std::shared_ptr<TransparentType> getPointedType() const;
  virtual llvm::SmallPtrSet<llvm::Type*, 4> getContainedTypes() const { return {unwrappedType}; }
  bool isArrayType() const { return unwrappedType->isArrayTy() || unwrappedType->isVectorTy(); }
  bool isStructType() const { return unwrappedType->isStructTy(); }
  bool isFloatingPointType() const { return unwrappedType->isFloatingPointTy(); }
  bool isIntegerType() const { return unwrappedType->isIntegerTy(); }
  virtual bool containsFloatingPointType() const { return unwrappedType->isFloatingPointTy(); }
  bool isPointerType() const { return indirections > 0 || isOpaquePointer(); }
  virtual bool isOpaquePointer() const { return unwrappedType->isPointerTy(); }
  virtual int compareTransparency(const TransparentType& other) const;
  llvm::Type* toLLVMType() const;
  virtual TransparentTypeKind getKind() const { return K_Scalar; }

  virtual bool operator==(const TransparentType& other) const;
  bool operator!=(const TransparentType& other) const { return !(*this == other); }

  virtual std::shared_ptr<TransparentType> clone() const;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

protected:
  llvm::Type* unwrappedType = nullptr;
  unsigned int indirections = 0;

  TransparentType() = default;
  TransparentType(const TransparentType& other) = default;

  TransparentType(llvm::Type* unwrappedType, unsigned int indirections)
  : unwrappedType(unwrappedType), indirections(indirections) {}

  void incrementIndirections(int increment);
};

class TransparentArrayType : public TransparentType {
public:
  friend class TransparentTypeFactory;
  friend class TypeDeducerPass;

  static bool classof(const TransparentType* type) { return type->getKind() == K_Array; }

  bool isOpaquePointer() const override;
  bool containsFloatingPointType() const override { return getArrayElementType()->containsFloatingPointType(); }
  int compareTransparency(const TransparentType& other) const override;
  std::shared_ptr<TransparentType> getArrayElementType() const { return elementType; }
  llvm::SmallPtrSet<llvm::Type*, 4> getContainedTypes() const override;
  std::shared_ptr<TransparentType> setArrayElementType(const std::shared_ptr<TransparentType>& elementType) {
    return this->elementType = elementType;
  }
  TransparentTypeKind getKind() const override { return K_Array; }

  bool operator==(const TransparentType& other) const override;

  std::shared_ptr<TransparentType> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

private:
  std::shared_ptr<TransparentType> elementType;

  TransparentArrayType() = default;

  TransparentArrayType(const TransparentArrayType& other)
  : TransparentType(other), elementType(other.elementType->clone()) {}

  TransparentArrayType(llvm::ArrayType* arrayType, unsigned int indirections)
  : TransparentType(arrayType, indirections) {
    elementType = TransparentTypeFactory::create(arrayType->getElementType(), 0);
  }

  TransparentArrayType(llvm::VectorType* vecType, unsigned int indirections)
  : TransparentType(vecType, indirections) {
    elementType = TransparentTypeFactory::create(vecType->getElementType(), 0);
  }
};

class TransparentStructType : public TransparentType {
public:
  friend class TransparentTypeFactory;
  friend class TypeDeducerPass;
  friend class taffo::FixedPointStructType;

  static bool classof(const TransparentType* type) { return type->getKind() == K_Struct; }

  auto begin() { return fieldTypes.begin(); }
  auto end() { return fieldTypes.end(); }
  auto begin() const { return fieldTypes.begin(); }
  auto end() const { return fieldTypes.end(); }

  bool isOpaquePointer() const override;
  bool containsFloatingPointType() const override;
  int compareTransparency(const TransparentType& other) const override;
  std::shared_ptr<TransparentType> getFieldType(unsigned int i) const { return fieldTypes[i]; }
  unsigned int getNumFieldTypes() const { return fieldTypes.size(); }
  llvm::SmallPtrSet<llvm::Type*, 4> getContainedTypes() const override;
  TransparentTypeKind getKind() const override { return K_Struct; }

  bool operator==(const TransparentType& other) const override;

  std::shared_ptr<TransparentType> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

private:
  llvm::SmallVector<std::shared_ptr<TransparentType>, 2> fieldTypes;

  TransparentStructType() = default;

  TransparentStructType(const TransparentStructType& other)
  : TransparentType(other) {
    for (auto field : other.fieldTypes)
      fieldTypes.push_back(field->clone());
  }

  TransparentStructType(llvm::StructType* unwrappedType, unsigned int indirections)
  : TransparentType(unwrappedType, indirections) {
    for (llvm::Type* fieldType : unwrappedType->elements())
      fieldTypes.push_back(TransparentTypeFactory::create(fieldType, 0));
  }

  void setFieldType(unsigned int i, std::shared_ptr<TransparentType> fieldType) { fieldTypes[i] = fieldType; }
};

} // namespace taffo

namespace std {

template <>
struct hash<shared_ptr<taffo::TransparentType>> {
  size_t operator()(const shared_ptr<taffo::TransparentType>& ptr) const {
    if (!ptr)
      return 0;

    size_t combined = 0;
    auto combine = [](size_t seed, size_t value) { return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2)); };

    combined = combine(combined, hash<llvm::Type*>()(ptr->getUnwrappedType()));
    combined = combine(combined, hash<unsigned int>()(ptr->getIndirections()));

    if (auto arrayPtr = dynamic_ptr_cast<taffo::TransparentArrayType>(ptr))
      combined = combine(combined, hash()(arrayPtr->getArrayElementType()));
    else if (auto structPtr = dynamic_ptr_cast<taffo::TransparentStructType>(ptr))
      for (const auto& field : *structPtr)
        combined = combine(combined, hash()(field));

    return combined;
  }
};

template <>
struct equal_to<shared_ptr<taffo::TransparentType>> {
  bool operator()(const shared_ptr<taffo::TransparentType>& lhs, const shared_ptr<taffo::TransparentType>& rhs) const {
    if (lhs == rhs)
      return true;
    if (!lhs || !rhs)
      return false;
    return *lhs == *rhs;
  }
};

} // namespace std
