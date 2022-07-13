#ifndef TAFFO_VRA_RANGE_NODE_HPP
#define TAFFO_VRA_RANGE_NODE_HPP

#include "Range.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <memory>

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

class VRANode
{
public:
  enum VRANodeKind { VRAPtrNodeK,
                     VRAGEPNodeK,
                     VRAStructNodeK,
                     VRAScalarNodeK };
  VRANodeKind getKind() const { return Kind; }

protected:
  VRANode(VRANodeKind K) : Kind(K) {}

private:
  const VRANodeKind Kind;
};
using NodePtrT = std::shared_ptr<VRANode>;

class VRAPtrNode : public VRANode
{
public:
  VRAPtrNode()
      : VRANode(VRAPtrNodeK), Parent(nullptr) {}

  VRAPtrNode(NodePtrT P)
      : VRANode(VRAPtrNodeK), Parent(P) {}

  NodePtrT getParent() const { return Parent; }
  void setParent(NodePtrT P) { Parent = P; }

  static bool classof(const VRANode *N)
  {
    return N->getKind() >= VRAPtrNodeK && N->getKind() <= VRAGEPNodeK;
  }

protected:
  NodePtrT Parent;

  VRAPtrNode(VRANodeKind K, NodePtrT P)
      : VRANode(K), Parent(P) {}
};

class VRAGEPNode : public VRAPtrNode
{
public:
  VRAGEPNode(NodePtrT Parent, llvm::ArrayRef<unsigned> Offset)
      : VRAPtrNode(VRAGEPNodeK, Parent), ParentOffset(Offset.begin(), Offset.end()) {}

  const llvm::ArrayRef<unsigned> getOffset() const { return ParentOffset; }

  static bool classof(const VRANode *N)
  {
    return N->getKind() == VRAGEPNodeK;
  }

protected:
  llvm::SmallVector<unsigned, 1U> ParentOffset;
};

class VRARangeNode : public VRANode
{
public:
  static bool classof(const VRANode *N)
  {
    return N->getKind() >= VRAStructNodeK && N->getKind() <= VRAScalarNodeK;
  }

protected:
  VRARangeNode(VRANodeKind K) : VRANode(K) {}
};
using RangeNodePtrT = std::shared_ptr<VRARangeNode>;

class VRAStructNode : public VRARangeNode
{
public:
  VRAStructNode()
      : VRARangeNode(VRAStructNodeK), Fields() {}

  VRAStructNode(llvm::ArrayRef<NodePtrT> Fields)
      : VRARangeNode(VRAStructNodeK), Fields(Fields.begin(), Fields.end()) {}

  const llvm::ArrayRef<NodePtrT> fields() const { return Fields; }
  unsigned getNumFields() const { return Fields.size(); }
  NodePtrT getNodeAt(unsigned Idx) const
  {
    return (Idx < Fields.size()) ? Fields[Idx] : nullptr;
  }

  void setNodeAt(unsigned Idx, NodePtrT Node)
  {
    if (Idx >= Fields.size())
      Fields.resize(Idx + 1U, nullptr);
    Fields[Idx] = Node;
  }

  static bool classof(const VRANode *N)
  {
    return N->getKind() == VRAStructNodeK;
  }

protected:
  llvm::SmallVector<NodePtrT, 4U> Fields;
};

class VRAScalarNode : public VRARangeNode
{
public:
  VRAScalarNode(const range_ptr_t Range)
      : VRARangeNode(VRAScalarNodeK), Range(Range) {}

  range_ptr_t getRange() const { return Range; }
  void setRange(range_ptr_t R) { Range = R; }
  bool isFinal() const { return Range && Range->isFinal(); }

  static bool classof(const VRANode *N)
  {
    return N->getKind() == VRAScalarNodeK;
  }

protected:
  range_ptr_t Range;
};

} // namespace taffo

#undef DEBUG_TYPE

#endif /* end of include guard: TAFFO_VRA_RANGE_NODE_HPP */
