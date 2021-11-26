#include "VRAStore.hpp"

#include "llvm/Support/Debug.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "RangeOperations.hpp"

using namespace llvm;
using namespace taffo;

void
VRAStore::convexMerge(const VRAStore &Other) {
  for (auto &OValueRangeNode : Other.DerivedRanges) {
    const llvm::Value *V = OValueRangeNode.first;
    NodePtrT ThisNode = this->getNode(V);
    if (ThisNode) {
      if (std::isa_ptr<VRAStructNode>(ThisNode))
        assignStructNode(ThisNode, OValueRangeNode.second);
      else if (NodePtrT Union = assignScalarRange(ThisNode, OValueRangeNode.second)) {
        DerivedRanges[V] = Union;
      }
    } else {
      DerivedRanges[V] = OValueRangeNode.second;
    }
  }
}

const range_ptr_t
VRAStore::fetchRange(const llvm::Value *V) {
  if (const std::shared_ptr<VRAScalarNode> Ret =
      std::dynamic_ptr_cast_or_null<VRAScalarNode>(fetchRangeNode(V)))
    return Ret->getRange();
  return nullptr;
}

const RangeNodePtrT
VRAStore::fetchRangeNode(const llvm::Value* v) {
  if (const NodePtrT Node = getNode(v)) {
    if (std::shared_ptr<VRAScalarNode> Scalar =
        std::dynamic_ptr_cast<VRAScalarNode>(Node)) {
      return Scalar;
    } else if (v->getType()->isPointerTy()) {
      return fetchRange(Node);
    }
  }

  // no info available
  return nullptr;
}

void
VRAStore::saveValueRange(const llvm::Value* v,
                         const range_ptr_t Range) {
  if (!Range)
    return;
  // TODO: make specialized version of this to avoid creating useless node
  saveValueRange(v, std::make_shared<VRAScalarNode>(Range));
}

void
VRAStore::saveValueRange(const llvm::Value* v,
                         const RangeNodePtrT Range) {
  assert(v && "Trying to save range for null value.");
  if (!Range)
    return;

  if (NodePtrT Union = assignScalarRange(getNode(v), Range)) {
    DerivedRanges[v] = Union;
    return;
  }
  DerivedRanges[v] = Range;
}

NodePtrT
VRAStore::getNode(const llvm::Value* v) {
  assert(v && "Trying to get node for null value.");
  const auto it = DerivedRanges.find(v);
  if (it != DerivedRanges.end()) {
    return it->second;
  }
  return nullptr;
}

void
VRAStore::setNode(const llvm::Value* V, NodePtrT Node) {
  DerivedRanges[V] = Node;
}

NodePtrT
VRAStore::loadNode(const NodePtrT Node) const {
  llvm::SmallVector<unsigned, 1U> Offset;
  return loadNode(Node, Offset);
}

NodePtrT
VRAStore::loadNode(const NodePtrT Node,
                   llvm::SmallVectorImpl<unsigned>& Offset) const {
  if (!Node) return nullptr;
  switch (Node->getKind()) {
    case VRANode::VRAScalarNodeK:
      return Node;
    case VRANode::VRAStructNodeK:
      if (Offset.empty()) {
        return Node;
      } else {
        std::shared_ptr<VRAStructNode> StructNode =
          std::static_ptr_cast<VRAStructNode>(Node);
        NodePtrT Field = StructNode->getNodeAt(Offset.back());
        Offset.pop_back();
        if (Offset.empty())
          return Field;
        else
          return loadNode(Field, Offset);
      }
    case VRANode::VRAGEPNodeK: {
      std::shared_ptr<VRAGEPNode> GEPNode =
        std::static_ptr_cast<VRAGEPNode>(Node);
      const llvm::ArrayRef<unsigned> GEPOffset = GEPNode->getOffset();
      Offset.append(GEPOffset.begin(), GEPOffset.end());
      return loadNode(GEPNode->getParent(), Offset);
    }
    case VRANode::VRAPtrNodeK: {
      std::shared_ptr<VRAPtrNode> PtrNode =
        std::static_ptr_cast<VRAPtrNode>(Node);
      return PtrNode->getParent();
    }
    default:
      llvm_unreachable("Unhandled node type.");
  }
}

std::shared_ptr<VRAScalarNode>
VRAStore::assignScalarRange(NodePtrT Dst, const NodePtrT Src) const {
  std::shared_ptr<VRAScalarNode> ScalarDst =
    std::dynamic_ptr_cast_or_null<VRAScalarNode>(Dst);
  const std::shared_ptr<VRAScalarNode> ScalarSrc =
    std::dynamic_ptr_cast_or_null<VRAScalarNode>(Src);
  if (!(ScalarDst && ScalarSrc))
    return nullptr;

  if (ScalarDst->isFinal())
    return ScalarDst;

  range_ptr_t Union = getUnionRange(ScalarDst->getRange(), ScalarSrc->getRange());
  return std::make_shared<VRAScalarNode>(Union);
}

void
VRAStore::assignStructNode(NodePtrT Dst, const NodePtrT Src) const {
  std::shared_ptr<VRAStructNode> StructDst =
    std::dynamic_ptr_cast_or_null<VRAStructNode>(Dst);
  const std::shared_ptr<VRAStructNode> StructSrc =
    std::dynamic_ptr_cast_or_null<VRAStructNode>(Src);
  if (!(StructDst && StructSrc))
    return;

  const llvm::ArrayRef<NodePtrT> SrcFields = StructSrc->fields();
  for (unsigned Idx = 0; Idx < SrcFields.size(); ++Idx) {
    NodePtrT SrcField = SrcFields[Idx];
    if (!SrcField)
      continue;
    NodePtrT DstField = StructDst->getNodeAt(Idx);
    if (!DstField) {
      StructDst->setNodeAt(Idx, SrcField);
    } else {
      if (std::isa_ptr<VRAStructNode>(DstField)) {
        assignStructNode(DstField, SrcField);
      } else if (NodePtrT Union = assignScalarRange(DstField, SrcField)) {
        StructDst->setNodeAt(Idx, Union);
      }
    }
  }
}

void
VRAStore::storeNode(NodePtrT Dst, const NodePtrT Src) {
  llvm::SmallVector<unsigned, 1U> Offset;
  storeNode(Dst, Src, Offset);
}

void
VRAStore::storeNode(NodePtrT Dst, const NodePtrT Src,
                    llvm::SmallVectorImpl<unsigned>& Offset) {
  if (!(Dst && Src)) return;
  NodePtrT Pointed = nullptr;
  switch (Dst->getKind()) {
    case VRANode::VRAGEPNodeK: {
      std::shared_ptr<VRAGEPNode> GEPNode =
        std::static_ptr_cast<VRAGEPNode>(Dst);
      const llvm::ArrayRef<unsigned> GEPOffset = GEPNode->getOffset();
      Offset.append(GEPOffset.begin(), GEPOffset.end());
      storeNode(GEPNode->getParent(), Src, Offset);
      break;
    }
    case VRANode::VRAStructNodeK: {
        std::shared_ptr<VRAStructNode> StructDst =
          std::static_ptr_cast<VRAStructNode>(Dst);
      if (Offset.empty()) {
        assignStructNode(StructDst, Src);
      } else if (Offset.size() == 1U) {
        unsigned Idx = Offset.front();
        NodePtrT Union = assignScalarRange(StructDst->getNodeAt(Idx), Src);
        if (Union) {
          StructDst->setNodeAt(Idx, Union);
        } else {
          StructDst->setNodeAt(Idx, Src);
        }
      } else {
        NodePtrT Field = StructDst->getNodeAt(Offset.back());
        if (!Field) {
          Field = std::make_shared<VRAStructNode>();
          StructDst->setNodeAt(Offset.back(), Field);
        }
        Offset.pop_back();
        storeNode(Field, Src, Offset);
      }
      break;
    }
    case VRANode::VRAPtrNodeK: {
      std::shared_ptr<VRAPtrNode> PtrDst =
        std::static_ptr_cast<VRAPtrNode>(Dst);
      NodePtrT Union = assignScalarRange(PtrDst->getParent(), Src);
      if (Union) {
        PtrDst->setParent(Union);
      } else {
        PtrDst->setParent(Src);
      }
      break;
    }
    default:
      LLVM_DEBUG(dbgs() << "WARNING: trying to store into a non-pointer node, aborted.\n");
  }
}

RangeNodePtrT
VRAStore::fetchRange(const NodePtrT Node) const {
  llvm::SmallVector<unsigned, 1U> Offset;
  return fetchRange(Node, Offset);
}

RangeNodePtrT
VRAStore::fetchRange(const NodePtrT Node,
                     llvm::SmallVectorImpl<unsigned>& Offset) const {
  if (!Node) return nullptr;
  switch (Node->getKind()) {
    case VRANode::VRAScalarNodeK:
      return std::static_ptr_cast<VRAScalarNode>(Node);
    case VRANode::VRAStructNodeK: {
      std::shared_ptr<VRAStructNode> StructNode =
        std::static_ptr_cast<VRAStructNode>(Node);
      if (Offset.empty()) {
        return StructNode;
      } else {
        NodePtrT Field = StructNode->getNodeAt(Offset.back());
        Offset.pop_back();
        return fetchRange(Field, Offset);
      }
    }
    case VRANode::VRAGEPNodeK: {
      std::shared_ptr<VRAGEPNode> GEPNode =
        std::dynamic_ptr_cast<VRAGEPNode>(Node);
      const llvm::ArrayRef<unsigned> GEPOffset = GEPNode->getOffset();
      Offset.append(GEPOffset.begin(), GEPOffset.end());
      return fetchRange(GEPNode->getParent(), Offset);
    }
    case VRANode::VRAPtrNodeK: {
      std::shared_ptr<VRAPtrNode> PtrNode =
        std::dynamic_ptr_cast<VRAPtrNode>(Node);
      return fetchRange(PtrNode->getParent(), Offset);
    }
    default:
      llvm_unreachable("Unhandled node type.");
  }
}

bool
VRAStore::extractGEPOffset(const llvm::Type* source_element_type,
                           const llvm::iterator_range<llvm::User::const_op_iterator> indices,
                           llvm::SmallVectorImpl<unsigned>& offset) {
  assert(source_element_type != nullptr);
  LLVM_DEBUG(dbgs() << "indices: ");
  for (auto idx_it = indices.begin() + 1; // skip first index
       idx_it != indices.end(); ++idx_it) {
    if (isa<ArrayType>(source_element_type) || isa<VectorType>(source_element_type) )
      continue;
    const llvm::ConstantInt* int_i = dyn_cast<llvm::ConstantInt>(*idx_it);
    if (int_i) {
      int n = static_cast<int>(int_i->getSExtValue());
      offset.push_back(n);
      source_element_type =
        cast<StructType>(source_element_type)->getTypeAtIndex(n);
      LLVM_DEBUG(dbgs() << n << " ");
    } else {
      LLVM_DEBUG(Logger->logErrorln("Index of GEP not constant"));
      return false;
    }
  }
  LLVM_DEBUG(dbgs() << "\n");
  return true;
}
