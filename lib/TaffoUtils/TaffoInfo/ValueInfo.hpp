#ifndef TAFFO_VALUE_INFO_HPP
#define TAFFO_VALUE_INFO_HPP

#include "NumericInfo.hpp"
#include "RangeInfo.hpp"
#include "SerializationUtils.hpp"

#include <llvm/IR/DerivedTypes.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Instructions.h>

namespace taffo {

class TaffoInitializer; //TODO remove
class AnnotationParser;

class ValueInfo : public Serializable, public Printable {
public:
  friend class TaffoInitializer; //TODO remove
  friend class AnnotationParser;

  enum ValueInfoKind {
    K_Scalar,
    K_Struct,
    K_Pointer,
    K_GetElementPointer
  };

  ValueInfo(ValueInfoKind kind, llvm::Type *unwrappedType = nullptr)
  : kind(kind), unwrappedType(unwrappedType) {}

  virtual ~ValueInfo() = default;

  ValueInfoKind getKind() const { return kind; }

  virtual void setUnwrappedType(llvm::Type *t) { unwrappedType = t; }
  virtual llvm::Type *getUnwrappedType() const { return unwrappedType; }

  std::optional<std::string> getTarget() { return target; }
  std::optional<std::string> getBufferId() { return bufferId; }

  virtual bool isConversionEnabled() const = 0;

  virtual std::shared_ptr<ValueInfo> clone() const = 0;
  json serialize() const override;
  void deserialize(const json &j) override;

protected:
  void copyFrom(const ValueInfo &other);

private:
  const ValueInfoKind kind;
  llvm::Type *unwrappedType;
  std::optional<std::string> target;
  std::optional<std::string> bufferId;
};

class ValueInfoWithRange : public ValueInfo {
public:
  static bool classof(const ValueInfo *valueInfo) {
    return valueInfo->getKind() == K_Scalar || valueInfo->getKind() == K_Struct;
  }

  ValueInfoWithRange(ValueInfoKind kind, llvm::Type *type = nullptr) : ValueInfo(kind, type) {}
};

/// Structure containing pointers to Type, Range, and initial Error
/// of an LLVM Value.
struct ScalarInfo : public ValueInfoWithRange {
  static bool classof(const ValueInfo *valueInfo) { return valueInfo->getKind() == K_Scalar; }

  std::shared_ptr<NumericType> numericType;
  std::shared_ptr<Range> range;
  std::shared_ptr<double> error;
  bool conversionEnabled;
  bool final;

  ScalarInfo(llvm::Type *type,
             std::shared_ptr<NumericType> numericType = nullptr,
             std::shared_ptr<Range> range = nullptr,
             std::shared_ptr<double> error = nullptr,
             bool conversionEnabled = false,
             bool final = false)
  : ValueInfoWithRange(K_Scalar, type),
  numericType(numericType), range(range), error(error),
  conversionEnabled(conversionEnabled), final(final) {}

  bool isConversionEnabled() const override { return conversionEnabled; };
  bool isFinal() const { return final; }

  ScalarInfo &operator=(const ScalarInfo &other);

  std::shared_ptr<ValueInfo> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;
};

class StructInfo : public ValueInfoWithRange {
private:
  using FieldsType = llvm::SmallVector<std::shared_ptr<ValueInfo>, 4U>;

public:
  using iterator = FieldsType::iterator;
  using const_iterator = FieldsType::const_iterator;
  using size_type = FieldsType::size_type;

  /** Builds a StructInfo with the recursive structure of the specified
   *  LLVM Type. All non-struct struct members are set to nullptr.
   *  @returns Either a StructInfo, or nullptr if the type does not
   *    contain any structure. */
  static std::shared_ptr<StructInfo> constructFromLLVMType(
      llvm::Type *type, llvm::SmallDenseMap<llvm::Type*, std::shared_ptr<StructInfo>> *recursionMap = nullptr);

  static bool classof(const ValueInfo *valueInfo) { return valueInfo->getKind() == K_Struct; }

  StructInfo(llvm::StructType *type, unsigned int numFields)
  : ValueInfoWithRange(K_Struct, type), Fields(numFields, nullptr) {}

  StructInfo(llvm::StructType *type, const llvm::ArrayRef<std::shared_ptr<ValueInfo>> SInfos)
  : ValueInfoWithRange(K_Struct, type), Fields(SInfos.begin(), SInfos.end()) {}

  iterator begin() { return Fields.begin(); }
  iterator end() { return Fields.end(); }
  const_iterator begin() const { return Fields.begin(); }
  const_iterator end() const { return Fields.end(); }

  size_type numFields() const { return Fields.size(); }
  std::shared_ptr<ValueInfo> getField(size_type i) { return Fields[i]; }
  void setField(size_type i, std::shared_ptr<ValueInfo> field) { Fields[i] = std::move(field); }

  bool isConversionEnabled() const override;

  llvm::StructType* getUnwrappedType() const override {
    return llvm::cast<llvm::StructType>(ValueInfo::getUnwrappedType());
  }

  std::shared_ptr<ValueInfo> resolveFromIndexList(llvm::Type *type, llvm::ArrayRef<unsigned> indices);

  std::shared_ptr<ValueInfo> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

private:
  FieldsType Fields;

  bool isConversionEnabled(llvm::SmallPtrSetImpl<const StructInfo*> &visited) const;
};

class PointerInfo : public ValueInfo {
public:
  static bool classof(const ValueInfo *valueInfo) {
    return valueInfo->getKind() == K_Pointer || valueInfo->getKind() == K_GetElementPointer;
  }

  PointerInfo(const std::shared_ptr<ValueInfo> &pointed, llvm::Type *type = nullptr)
  : ValueInfo(K_Pointer, type), pointed(pointed) {}

  void setPointed(const std::shared_ptr<ValueInfo> &p) { pointed = p; }
  std::shared_ptr<ValueInfo> getPointed() const { return pointed; }

  std::shared_ptr<ValueInfoWithRange> getUnwrappedInfo() const;
  bool isConversionEnabled() const override;

  std::shared_ptr<ValueInfo> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

protected:
  std::shared_ptr<ValueInfo> pointed;

  PointerInfo(ValueInfoKind kind, const std::shared_ptr<ValueInfo> &pointed, llvm::Type *type = nullptr)
  : ValueInfo(kind, type), pointed(pointed) {}
};

class GEPInfo : public PointerInfo {
public:
  static bool classof(const ValueInfo *valueInfo) { return valueInfo->getKind() == K_GetElementPointer; }

  GEPInfo(const std::shared_ptr<ValueInfo> &pointed, llvm::Type *type = nullptr)
  : PointerInfo(K_GetElementPointer, pointed, type) {}

  GEPInfo(const std::shared_ptr<ValueInfo> &pointed, const llvm::ArrayRef<unsigned> offset, llvm::Type *type = nullptr)
  : PointerInfo(K_GetElementPointer, pointed, type), offset(offset.begin(), offset.end()) {}

  llvm::ArrayRef<unsigned> getOffset() const { return offset; }

  bool isConversionEnabled() const override;

  std::shared_ptr<ValueInfo> clone() const override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

private:
  llvm::SmallVector<unsigned, 1> offset;
};

/// Struct containing info about a possible comparison error.
struct CmpErrorInfo : public Serializable {
public:
  double MaxTolerance; ///< Maximum error tolerance for this comparison.
  bool MayBeWrong;     ///< True if this comparison may be wrong due to propagated errors.

  CmpErrorInfo(double MaxTolerance, bool MayBeWrong = true)
      : MaxTolerance(MaxTolerance), MayBeWrong(MayBeWrong) {}

  json serialize() const override;
  void deserialize(const json &j) override;
};

} // namespace taffo

#endif // TAFFO_VALUE_INFO_HPP
