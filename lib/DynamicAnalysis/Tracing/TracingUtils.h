#ifndef TAFFO_TRACINGUTILS_H
#define TAFFO_TRACINGUTILS_H

#include <list>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>
#include <memory>
#include <unordered_map>

#include "TypeUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

namespace taffo
{

struct TracingUtils
{
  static bool isMallocLike(const llvm::Function *F);

  static bool isMallocLike(const llvm::Value *Inst);

  static bool isExternalCallWithPointer(const llvm::Function *fun, unsigned int argNo);

  static bool isSafeExternalFunction(const llvm::Function *F);

};

class ValueWrapper
{
public:
  enum class ValueType {
    ValInst = 0,
    ValFunCallArg,
    ValStructElem,
    ValStructElemFunCall
  };

  // since C++11
  const char * const ValueTypes[4] =
  {
      "ValInst",
      "ValFunCallArg",
      "ValStructElem",
      "ValStructElemFunCall"
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

  virtual llvm::raw_ostream& print_debug(llvm::raw_ostream &dbg) const
  {
    return dbg << ValueTypes[(int)type] << ": " << *value;
  }

  virtual llvm::iterator_range<llvm::Value::use_iterator> uses() const
  {
    return value->uses();
  }

  bool isStructElem() const {
    return type == ValueType::ValStructElem;
  }

  bool isStructElemFunCall() const {
    return type == ValueType::ValStructElemFunCall;
  }

  bool isFunCallArg() const {
    return type == ValueType::ValFunCallArg || type == ValueType::ValStructElemFunCall;
  }

  static std::shared_ptr<ValueWrapper> wrapValue(llvm::Value *V);
  static std::shared_ptr<ValueWrapper> wrapFunCallArg(llvm::Function *fun, unsigned int argNo);
  static std::shared_ptr<ValueWrapper> wrapStructElem(llvm::Value *V, unsigned int ArgPos);
  static std::shared_ptr<ValueWrapper> wrapStructElemFunCallArg(llvm::Function *fun, unsigned int ArgPos, unsigned int FunArgPos);
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

  llvm::raw_ostream& print_debug(llvm::raw_ostream &dbg) const override
  {
    std::string s = "";
    if (isExternalFunc) {
      s = " [disabled]: ";
    }
    return dbg << ValueTypes[(int)type]
               << s
               << ", arg<" << argPos << ">: "
               << *value;
  }
};

class StructElemWrapper : public ValueWrapper
{
public:
  StructElemWrapper(llvm::Value *V, unsigned int ArgPos)
      : ValueWrapper{ValueType::ValStructElem, V}, argPos{ArgPos}{}
  const unsigned int argPos;
  bool operator==(const ValueWrapper &other) const override
  {
    if (ValueWrapper::operator==(other)) {
      return argPos == static_cast<const StructElemWrapper *>(&other)->argPos;
    }
    return false;
  }

  llvm::Type* argType() {
    auto* structType = fullyUnwrapPointerOrArrayType(value->getType());
    auto t = structType->getStructElementType(argPos);
    return t;
  }

  llvm::raw_ostream& print_debug(llvm::raw_ostream &dbg) const override
  {
    return dbg << ValueTypes[(int)type] << ", field<" << argPos << ">: " << *value;
  }
};

class StructElemFunCallArgWrapper : public ValueWrapper
{
public:
  StructElemFunCallArgWrapper(llvm::Value *V, unsigned int ArgPos, int FunArgPos, bool FunExternal)
      : ValueWrapper{ValueType::ValStructElemFunCall, V}, argPos{ArgPos}, funArgPos{FunArgPos}, funExternal{FunExternal} {}
  const unsigned int argPos;
  const int funArgPos;
  const bool funExternal;
  bool operator==(const ValueWrapper &other) const override
  {
    if (ValueWrapper::operator==(other)) {
      const StructElemFunCallArgWrapper *castedWrapper = static_cast<const StructElemFunCallArgWrapper *>(&other);
      return (argPos == castedWrapper->argPos) &&
             (funArgPos == castedWrapper->funArgPos)  &&
             (funExternal == castedWrapper->funExternal);
    }
    return false;
  }

  llvm::Type* argType() {
    auto* structType = fullyUnwrapPointerOrArrayType(value->getType());
    auto t = structType->getStructElementType(argPos);
    return t;
  }

  llvm::raw_ostream& print_debug(llvm::raw_ostream &dbg) const override
  {
    std::string s = "";
    if (funExternal) {
      s = " [disabled]: ";
    }
    return dbg << ValueTypes[(int)type]
               << s
               << ", arg<" << funArgPos << "> "
               << ", field<" << argPos << ">: "
               << *value;
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
