#ifndef TAFFO_VRASTORE_HPP
#define TAFFO_VRASTORE_HPP

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include <list>
#include <vector>

#include "RangeNode.hpp"
#include "VRALogger.hpp"

namespace taffo
{

class VRAStore
{
public:
  virtual void convexMerge(const VRAStore &Other);

  virtual const range_ptr_t fetchRange(const llvm::Value *V);
  virtual RangeNodePtrT fetchRange(const NodePtrT Node) const;
  virtual const RangeNodePtrT fetchRangeNode(const llvm::Value *v);
  virtual void saveValueRange(const llvm::Value *v, const range_ptr_t Range);
  virtual void saveValueRange(const llvm::Value *v, const RangeNodePtrT Range);
  virtual NodePtrT getNode(const llvm::Value *v);
  virtual void setNode(const llvm::Value *V, NodePtrT Node);
  virtual NodePtrT loadNode(const NodePtrT Node) const;
  virtual void storeNode(NodePtrT Dst, const NodePtrT Src);
  virtual ~VRAStore() = default;

  enum VRAStoreKind { VRASK_VRAGlobalStore,
                      VRASK_VRAnalyzer,
                      VRASK_VRAFunctionStore };
  VRAStoreKind getKind() const { return Kind; }

protected:
  llvm::DenseMap<const llvm::Value *, NodePtrT> DerivedRanges;
  std::shared_ptr<VRALogger> Logger;

  std::shared_ptr<VRAScalarNode> assignScalarRange(NodePtrT Dst, const NodePtrT Src) const;
  void assignStructNode(NodePtrT Dst, const NodePtrT Src) const;
  bool extractGEPOffset(const llvm::Type *source_element_type,
                        const llvm::iterator_range<llvm::User::const_op_iterator> indices,
                        llvm::SmallVectorImpl<unsigned> &offset);
  NodePtrT loadNode(const NodePtrT Node, llvm::SmallVectorImpl<unsigned> &Offset) const;
  void storeNode(NodePtrT Dst, const NodePtrT Src, llvm::SmallVectorImpl<unsigned> &Offset);
  RangeNodePtrT fetchRange(const NodePtrT Node, llvm::SmallVectorImpl<unsigned> &Offset) const;

  VRAStore(VRAStoreKind K, std::shared_ptr<VRALogger> L)
      : DerivedRanges(), Logger(L), Kind(K) {}

private:
  const VRAStoreKind Kind;
};

} // end namespace taffo

#endif
