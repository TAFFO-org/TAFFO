#ifndef TAFFO_MEMORYGRAPH_H
#define TAFFO_MEMORYGRAPH_H

#include <unordered_map>
#include <list>
#include <memory>

#include "llvm/IR/Value.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"

namespace taffo
{

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
};

class InstWrapper : public ValueWrapper
{
public:
  InstWrapper(llvm::Value *V) : ValueWrapper{ValueType::ValInst, V} {}
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

    return hash<int>()((size_t)k->value);
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

namespace taffo {

class MemoryGraph
{
public:
  MemoryGraph(llvm::Module &Module);

  const std::list<std::pair<int, int>>& getEdges() {
    return edges;
  }

  int getNodeCount() {
    return index + 1;
  }

  std::shared_ptr<ValueWrapper> getNode(int i) {
    return indexToInst.at(i);
  }

private:
  llvm::Module &M;

  std::unordered_map<std::shared_ptr<ValueWrapper>, int,
                     std::hash<std::shared_ptr<taffo::ValueWrapper>>,
                     std::equal_to<std::shared_ptr<taffo::ValueWrapper>>> instToIndex;
  std::unordered_map<int, std::shared_ptr<ValueWrapper>> indexToInst;
  std::list<std::pair<int, int>> edges;
  std::list<llvm::Value*> queue;
  std::unordered_map<llvm::Value*, bool> visited;
  int index = -1;

  int assignOrGetIndex(std::shared_ptr<ValueWrapper> Inst);
  void seedRoots();
  void makeGraph();
  void addToGraph(std::shared_ptr<ValueWrapper> src, std::shared_ptr<ValueWrapper> dst);
  void queuePush(llvm::Value* V);
  llvm::Value* queuePop();
  void markVisited(llvm::Value* V);
  bool isVisited(llvm::Value* V);
  bool isMallocLike(const llvm::Function *F) const;
  bool isMallocLike(const llvm::Value *Inst) const;
  bool isExternalCallWithPointer(const llvm::CallInst *V, int argNo) const;
  void handleAllocaInst(llvm::AllocaInst* allocaInst);
  void handleStoreInst(llvm::StoreInst* storeInst);
  void handleLoadInst(llvm::LoadInst* loadInst);
  void handleGEPInst(llvm::GetElementPtrInst* gepInst);
  void addUsesToGraph(llvm::Value* V);
  std::shared_ptr<ValueWrapper> wrapValue(llvm::Value *V);
  std::shared_ptr<ValueWrapper> wrapValueUse(llvm::Use *V);
};

} // namespace taffo

#endif // TAFFO_MEMORYGRAPH_H
