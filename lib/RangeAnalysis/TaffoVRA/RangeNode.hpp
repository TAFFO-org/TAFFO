#ifndef TAFFO_VRA_RANGE_NODE_HPP
#define TAFFO_VRA_RANGE_NODE_HPP

#include "Range.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"

#include <llvm/IR/DerivedTypes.h>
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

  llvm::Type *getType() const { return Type; }
  unsigned int getSizeInBits() const { return SizeInBits; }
  VRANodeKind getKind() const { return Kind; }

protected:
  llvm::Type *Type;
  unsigned int SizeInBits;

  VRANode(VRANodeKind K, llvm::Type *Type, unsigned int SizeInBits)
      : Type(Type), SizeInBits(SizeInBits), Kind(K)
  {
    assert(Type);
  }

private:
  const VRANodeKind Kind;
};
using NodePtrT = std::shared_ptr<VRANode>;

class VRAPtrNode : public VRANode
{
public:
  VRAPtrNode(llvm::Type *Type, unsigned int SizeInBits)
      : VRANode(VRAPtrNodeK, Type, SizeInBits), Parent(nullptr) {}

  VRAPtrNode(llvm::Type *Type, unsigned int SizeInBits, NodePtrT Parent)
      : VRANode(VRAPtrNodeK, Type, SizeInBits), Parent(Parent) {}

  NodePtrT getParent() const { return Parent; }
  void setParent(NodePtrT P) { Parent = P; }

  static bool classof(const VRANode *N)
  {
    return N->getKind() >= VRAPtrNodeK && N->getKind() <= VRAGEPNodeK;
  }

protected:
  NodePtrT Parent;

  VRAPtrNode(VRANodeKind K, llvm::Type *Type, unsigned int SizeInBits, NodePtrT P)
      : VRANode(K, Type, SizeInBits), Parent(P) {}
};

class VRAGEPNode : public VRAPtrNode
{
public:
  VRAGEPNode(llvm::Type *Type, unsigned int SizeInBits, NodePtrT Parent, llvm::ArrayRef<unsigned> Offset)
      : VRAPtrNode(VRAGEPNodeK, Type, SizeInBits, Parent), ParentOffset(Offset.begin(), Offset.end()) {}

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
  VRARangeNode(VRANodeKind K, llvm::Type *Type, unsigned int SizeInBits)
      : VRANode(K, Type, SizeInBits) {}
};
using RangeNodePtrT = std::shared_ptr<VRARangeNode>;

class VRAStructNode : public VRARangeNode
{
public:
  VRAStructNode(llvm::Type *Type, unsigned int SizeInBits)
      : VRARangeNode(VRAStructNodeK, Type, SizeInBits), Fields()
  {
    auto ST = getStructType();
    Fields.reserve(ST->getNumElements());
    for (unsigned int i = 0; i < ST->getNumElements(); i++)
      Fields.push_back(nullptr);
  }

  VRAStructNode(llvm::Type *Type, unsigned int SizeInBits, llvm::ArrayRef<NodePtrT> Fields)
      : VRARangeNode(VRAStructNodeK, Type, SizeInBits), Fields(Fields.begin(), Fields.end())
  {
    getStructType(); //Just to trigger the type assertion
  }

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
    Fields[Idx] = std::move(Node);
  }

  static bool classof(const VRANode *N)
  {
    return N->getKind() == VRAStructNodeK;
  }

  llvm::StructType *getStructType() const;

protected:
  llvm::SmallVector<NodePtrT, 4U> Fields;
};

class VRAScalarNode : public VRARangeNode
{
public:
  VRAScalarNode(llvm::Type *Type, unsigned int SizeInBits, const range_ptr_t Range)
      : VRARangeNode(VRAScalarNodeK, Type, SizeInBits), Range(Range) {}

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
