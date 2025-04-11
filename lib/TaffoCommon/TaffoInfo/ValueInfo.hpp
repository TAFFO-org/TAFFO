#ifndef TAFFO_VALUE_INFO_HPP
#define TAFFO_VALUE_INFO_HPP

#include "NumericInfo.hpp"
#include "RangeInfo.hpp"
#include "Types/TransparentType.hpp"
#include "SerializationUtils.hpp"
#include "PtrCasts.hpp"

#include <llvm/IR/Instructions.h>

namespace taffo {

class TaffoInitializerPass; //TODO remove
class AnnotationParser;

class ValueInfo : public Serializable, public Printable {
public:
  friend class TaffoInitializerPass; //TODO remove
  friend class AnnotationParser;

  enum ValueInfoKind {
    K_Scalar,
    K_Struct,
    K_Pointer,
    K_GetElementPointer
  };

  ValueInfo(ValueInfoKind kind) : kind(kind) {}

  virtual ~ValueInfo() = default;

  ValueInfoKind getKind() const { return kind; }

  std::optional<std::string> getTarget() { return target; }
  std::optional<std::string> getBufferId() { return bufferId; }

  virtual bool isConversionEnabled() const = 0;

  template<typename ValueInfoT = ValueInfo>
  std::shared_ptr<ValueInfoT> clone() const {
    return std::dynamic_ptr_cast<ValueInfoT>(cloneImpl());
  }

  json serialize() const override;
  void deserialize(const json &j) override;

protected:
  void copyFrom(const ValueInfo &other);
  virtual std::shared_ptr<ValueInfo> cloneImpl() const = 0;

private:
  const ValueInfoKind kind;
  std::optional<std::string> target;
  std::optional<std::string> bufferId;
};

class ValueInfoWithRange : public ValueInfo {
public:
  static bool classof(const ValueInfo *valueInfo) {
    return valueInfo->getKind() == K_Scalar || valueInfo->getKind() == K_Struct;
  }

  ValueInfoWithRange(ValueInfoKind kind) : ValueInfo(kind) {}
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

  ScalarInfo(std::shared_ptr<NumericType> numericType = nullptr,
             std::shared_ptr<Range> range = nullptr,
             std::shared_ptr<double> error = nullptr,
             bool conversionEnabled = false,
             bool final = false)
  : ValueInfoWithRange(K_Scalar),
  numericType(numericType), range(range), error(error),
  conversionEnabled(conversionEnabled), final(final) {}

  bool isConversionEnabled() const override { return conversionEnabled; };
  bool isFinal() const { return final; }

  ScalarInfo &operator=(const ScalarInfo &other);

  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

private:
  std::shared_ptr<ValueInfo> cloneImpl() const override;
};

class StructInfo : public ValueInfoWithRange {
private:
  using FieldsType = llvm::SmallVector<std::shared_ptr<ValueInfo>, 4>;

public:
  using iterator = FieldsType::iterator;
  using const_iterator = FieldsType::const_iterator;

  /**
   * Builds a StructInfo with the recursive structure of the specified
   * TransparentType. All non-struct struct members are set to nullptr.
   */
  static std::shared_ptr<StructInfo> createFromTransparentType(const std::shared_ptr<TransparentStructType> &structType);

  static bool classof(const ValueInfo *valueInfo) { return valueInfo->getKind() == K_Struct; }

  StructInfo(unsigned int numFields)
  : ValueInfoWithRange(K_Struct), Fields(numFields, nullptr) {}

  StructInfo(const llvm::ArrayRef<std::shared_ptr<ValueInfo>> SInfos)
  : ValueInfoWithRange(K_Struct), Fields(SInfos.begin(), SInfos.end()) {}

  iterator begin() { return Fields.begin(); }
  iterator end() { return Fields.end(); }
  const_iterator begin() const { return Fields.begin(); }
  const_iterator end() const { return Fields.end(); }

  unsigned int numFields() const { return Fields.size(); }
  std::shared_ptr<ValueInfo> getField(unsigned int i) { return Fields[i]; }
  void setField(unsigned int i, std::shared_ptr<ValueInfo> field) { Fields[i] = std::move(field); }

  bool isConversionEnabled() const override;

  std::shared_ptr<ValueInfo> resolveFromIndexList(llvm::Type *type, llvm::ArrayRef<unsigned> indices);

  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

private:
  FieldsType Fields;

  static std::shared_ptr<StructInfo> createFromTransparentType(
    const std::shared_ptr<TransparentType> &type,
    std::unordered_map<std::shared_ptr<TransparentType>, std::shared_ptr<StructInfo>> &recursionMap);

  bool isConversionEnabled(llvm::SmallPtrSetImpl<const StructInfo*> &visited) const;

  std::shared_ptr<ValueInfo> cloneImpl() const override;
};

class PointerInfo : public ValueInfo {
public:
  static bool classof(const ValueInfo *valueInfo) {
    return valueInfo->getKind() == K_Pointer || valueInfo->getKind() == K_GetElementPointer;
  }

  PointerInfo(const std::shared_ptr<ValueInfo> &pointed)
  : ValueInfo(K_Pointer), pointed(pointed) {}

  void setPointed(const std::shared_ptr<ValueInfo> &p) { pointed = p; }
  std::shared_ptr<ValueInfo> getPointed() const { return pointed; }

  std::shared_ptr<ValueInfoWithRange> getUnwrappedInfo() const;
  bool isConversionEnabled() const override;

  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

protected:
  std::shared_ptr<ValueInfo> pointed;

  PointerInfo(ValueInfoKind kind, const std::shared_ptr<ValueInfo> &pointed)
  : ValueInfo(kind), pointed(pointed) {}

  std::shared_ptr<ValueInfo> cloneImpl() const override;
};

class GEPInfo : public PointerInfo {
public:
  static bool classof(const ValueInfo *valueInfo) { return valueInfo->getKind() == K_GetElementPointer; }

  GEPInfo(const std::shared_ptr<ValueInfo> &pointed)
  : PointerInfo(K_GetElementPointer, pointed) {}

  GEPInfo(const std::shared_ptr<ValueInfo> &pointed, const llvm::ArrayRef<unsigned> offset)
  : PointerInfo(K_GetElementPointer, pointed), offset(offset.begin(), offset.end()) {}

  llvm::ArrayRef<unsigned> getOffset() const { return offset; }

  bool isConversionEnabled() const override;

  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json &j) override;

private:
  llvm::SmallVector<unsigned, 1> offset;

  std::shared_ptr<ValueInfo> cloneImpl() const override;
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
