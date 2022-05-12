#ifndef TAFFO_TRACINGUTILS_H
#define TAFFO_TRACINGUTILS_H

#include <list>
#include <llvm/IR/Instructions.h>
#include <memory>
#include <unordered_map>

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

namespace taffo
{

struct TracingUtils
{
  static bool isMallocLike(const llvm::Function *F);

  static bool isMallocLike(const llvm::Value *Inst);

  static bool isExternalCallWithPointer(const llvm::CallInst *callInst, int argNo);

  static bool isSafeExternalFunction(const llvm::Function *F);
};

class ValueWrapper
{
public:
  enum class ValueType {
    ValInst,
    ValFunCallArg
  };

protected:
  ValueWrapper(ValueType T, llvm::Value *V) : type{T}, value{V} {}

public:
  const ValueType type;
  llvm::Value *value;
  virtual bool operator==(const ValueWrapper &other) const
  {
    return type == other.type && value == other.value;
  }

  static std::shared_ptr<ValueWrapper> wrapValue(llvm::Value *V);
  static std::shared_ptr<ValueWrapper> wrapValueUse(llvm::Use *V);
};

class InstWrapper : public ValueWrapper
{
public:
  explicit InstWrapper(llvm::Value *V) : ValueWrapper{ValueType::ValInst, V} {}
};

class FunCallArgWrapper : public ValueWrapper
{
public:
  FunCallArgWrapper(llvm::Value *V, int ArgPos, bool external)
      : ValueWrapper{ValueType::ValFunCallArg, V}, argPos{ArgPos}, isExternalFunc{external} {}
  const int argPos;
  const bool isExternalFunc;
  bool operator==(const ValueWrapper &other) const override
  {
    if (ValueWrapper::operator==(other)) {
      return argPos == static_cast<const FunCallArgWrapper *>(&other)->argPos;
    }
    return false;
  }
};

} // namespace taffo

namespace std {
template <>
struct hash<std::shared_ptr<taffo::ValueWrapper>>
{
  std::size_t operator()(const std::shared_ptr<taffo::ValueWrapper>& k) const
  {
    using std::size_t;
    using std::hash;
    using std::string;

    return hash<size_t>()((size_t)k->value);
  }
};

template <>
struct equal_to<std::shared_ptr<taffo::ValueWrapper>>
{
  bool operator()(const std::shared_ptr<taffo::ValueWrapper>& a, const std::shared_ptr<taffo::ValueWrapper>& b) const
  {
    return (*a == *b) && (*b == *a);
  }
};
} // namespace std

#endif // TAFFO_TRACINGUTILS_H
