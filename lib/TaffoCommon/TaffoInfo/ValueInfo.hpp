#pragma once

#include "NumericInfo.hpp"
#include "PtrCasts.hpp"
#include "RangeInfo.hpp"
#include "SerializationUtils.hpp"
#include "Types/TransparentType.hpp"

#include <llvm/IR/Instructions.h>

namespace taffo {

class ValueInfo;
class StructInfo;
class TaffoInfo;

class ValueInfoFactory {
private: // TODO Make all of this private and accessible by TaffoInfo only

public:
  friend class TaffoInfo;

  static std::shared_ptr<ValueInfo> create(llvm::Value* value);

  static std::shared_ptr<ValueInfo> create(const std::shared_ptr<TransparentType>& type);

  static std::shared_ptr<ValueInfo>
  create(const std::shared_ptr<TransparentType>& type,
         std::unordered_map<std::shared_ptr<TransparentType>, std::shared_ptr<StructInfo>>& recursionMap);
};

class InitializerPass; // TODO remove
class AnnotationParser;

class ValueInfo : public Serializable,
                  public Printable {
public:
  friend class InitializerPass; // TODO remove
  friend class AnnotationParser;

  enum ValueInfoKind {
    K_Scalar,
    K_Struct,
    K_Pointer,
    K_GetElementPointer
  };

  virtual ~ValueInfo() = default;

  std::optional<std::string> getTarget() { return target; }
  std::optional<std::string> getBufferId() { return bufferId; }

  virtual ValueInfoKind getKind() const = 0;
  virtual bool isConversionEnabled() const = 0;

  template <typename ValueInfoT = ValueInfo>
  std::shared_ptr<ValueInfoT> clone() const {
    return std::dynamic_ptr_cast<ValueInfoT>(cloneImpl());
  }

  virtual void copyFrom(const ValueInfo& other);
  json serialize() const override;
  void deserialize(const json& j) override;

protected:
  virtual std::shared_ptr<ValueInfo> cloneImpl() const = 0;

private:
  std::optional<std::string> target;
  std::optional<std::string> bufferId;
};

class ValueInfoWithRange : public ValueInfo {
public:
  static bool classof(const ValueInfo* valueInfo) {
    return valueInfo->getKind() == K_Scalar || valueInfo->getKind() == K_Struct;
  }
};

class ScalarInfo : public ValueInfoWithRange {
public:
  static bool classof(const ValueInfo* valueInfo) { return valueInfo->getKind() == K_Scalar; }

  std::shared_ptr<NumericTypeInfo> numericType;
  std::shared_ptr<Range> range;
  std::shared_ptr<double> error;
  bool conversionEnabled;
  bool final;

  ScalarInfo(std::shared_ptr<NumericTypeInfo> numericType = nullptr,
             std::shared_ptr<Range> range = nullptr,
             std::shared_ptr<double> error = nullptr,
             bool conversionEnabled = false,
             bool final = false)
  : numericType(numericType), range(range), error(error), conversionEnabled(conversionEnabled), final(final) {}

  ValueInfoKind getKind() const override { return K_Scalar; }
  bool isConversionEnabled() const override { return conversionEnabled; }
  bool isFinal() const { return final; }

  ScalarInfo& operator=(const ScalarInfo& other);

  void copyFrom(const ValueInfo& other) override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

private:
  std::shared_ptr<ValueInfo> cloneImpl() const override;
};

class StructInfo : public ValueInfoWithRange {
public:
  static bool classof(const ValueInfo* valueInfo) { return valueInfo->getKind() == K_Struct; }

  StructInfo(unsigned numFields)
  : Fields(numFields, nullptr) {}

  StructInfo(const llvm::ArrayRef<std::shared_ptr<ValueInfo>> SInfos)
  : Fields(SInfos.begin(), SInfos.end()) {}

  auto begin() { return Fields.begin(); }
  auto end() { return Fields.end(); }
  auto begin() const { return Fields.begin(); }
  auto end() const { return Fields.end(); }

  unsigned getNumFields() const { return Fields.size(); }
  std::shared_ptr<ValueInfo> getField(unsigned i) { return Fields[i]; }
  void setField(unsigned i, std::shared_ptr<ValueInfo> field) { Fields[i] = std::move(field); }

  ValueInfoKind getKind() const override { return K_Struct; }
  bool isConversionEnabled() const override;

  std::shared_ptr<ValueInfo> resolveFromIndexList(llvm::Type* type, llvm::ArrayRef<unsigned> indices) const;

  void copyFrom(const ValueInfo& other) override;
  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

private:
  llvm::SmallVector<std::shared_ptr<ValueInfo>, 4> Fields;

  bool isConversionEnabled(llvm::SmallPtrSetImpl<const StructInfo*>& visited) const;

  std::shared_ptr<ValueInfo> cloneImpl() const override;
};

class PointerInfo : public ValueInfo {
public:
  static bool classof(const ValueInfo* valueInfo) {
    return valueInfo->getKind() == K_Pointer || valueInfo->getKind() == K_GetElementPointer;
  }

  PointerInfo(const std::shared_ptr<ValueInfo>& pointed)
  : pointed(pointed) {}

  void setPointed(const std::shared_ptr<ValueInfo>& p) { pointed = p; }
  std::shared_ptr<ValueInfo> getPointed() const { return pointed; }
  std::shared_ptr<ValueInfoWithRange> getUnwrappedInfo() const;

  ValueInfoKind getKind() const override { return K_Pointer; }
  bool isConversionEnabled() const override;

  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

protected:
  std::shared_ptr<ValueInfo> pointed;

  std::shared_ptr<ValueInfo> cloneImpl() const override;
};

class GEPInfo : public PointerInfo {
public:
  static bool classof(const ValueInfo* valueInfo) { return valueInfo->getKind() == K_GetElementPointer; }

  GEPInfo(const std::shared_ptr<ValueInfo>& pointed)
  : PointerInfo(pointed) {}

  GEPInfo(const std::shared_ptr<ValueInfo>& pointed, const llvm::ArrayRef<unsigned> offset)
  : PointerInfo(pointed), offset(offset.begin(), offset.end()) {}

  llvm::ArrayRef<unsigned> getOffset() const { return offset; }

  ValueInfoKind getKind() const override { return K_GetElementPointer; }
  bool isConversionEnabled() const override;

  std::string toString() const override;
  json serialize() const override;
  void deserialize(const json& j) override;

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
  void deserialize(const json& j) override;
};

} // namespace taffo
