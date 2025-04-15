#pragma once

#include "TaffoInfo/ValueInfo.hpp"
#include "VRALogger.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/User.h>
#include <llvm/IR/Value.h>

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

class VRAStore
{
public:
  void convexMerge(const VRAStore &other);

  virtual std::shared_ptr<Range> fetchRange(const llvm::Value *v);
  virtual std::shared_ptr<ValueInfoWithRange> fetchRange(const std::shared_ptr<ValueInfo> valueInfo) const;
  virtual std::shared_ptr<ValueInfoWithRange> fetchRangeNode(const llvm::Value *v);
  virtual void saveValueRange(const llvm::Value *v, const std::shared_ptr<Range> range);
  virtual void saveValueRange(const llvm::Value *v, const std::shared_ptr<ValueInfoWithRange> valueInfoWithRange);
  virtual std::shared_ptr<ValueInfo> getNode(const llvm::Value *v);
  virtual void setNode(const llvm::Value *V, std::shared_ptr<ValueInfo> Node);
  virtual std::shared_ptr<ValueInfo> loadNode(const std::shared_ptr<ValueInfo> Node) const;
  virtual void storeNode(std::shared_ptr<ValueInfo> dst, const std::shared_ptr<ValueInfo> &src);
  virtual ~VRAStore() = default;

  enum VRAStoreKind { VRASK_VRAGlobalStore,
                      VRASK_VRAnalyzer,
                      VRASK_VRAFunctionStore };
  VRAStoreKind getKind() const { return Kind; }

protected:
  llvm::DenseMap<const llvm::Value*, std::shared_ptr<ValueInfo>> DerivedRanges;
  std::shared_ptr<VRALogger> Logger;

  std::shared_ptr<ScalarInfo> assignScalarRange(const std::shared_ptr<ValueInfo> &dst, const std::shared_ptr<ValueInfo> &src) const;
  void assignStructNode(const std::shared_ptr<ValueInfo> &dst, const std::shared_ptr<ValueInfo> &src) const;
  bool extractGEPOffset(const llvm::Type *sourceElementType,
                        const llvm::iterator_range<llvm::User::const_op_iterator> indices,
                        llvm::SmallVectorImpl<unsigned> &offset) const;
  std::shared_ptr<ValueInfo> loadNode(const std::shared_ptr<ValueInfo> &valueInfo, llvm::SmallVectorImpl<unsigned> &Offset) const;
  void storeNode(const std::shared_ptr<ValueInfo> &dst, const std::shared_ptr<ValueInfo> &src, llvm::SmallVectorImpl<unsigned> &offset);
  std::shared_ptr<ValueInfoWithRange> fetchRange(const   std::shared_ptr<ValueInfo> &valueInfo, llvm::SmallVectorImpl<unsigned> &offset) const;

  VRAStore(VRAStoreKind K, std::shared_ptr<VRALogger> L)
  : Logger(L), Kind(K) {}

private:
  const VRAStoreKind Kind;
};

} // end namespace taffo

#undef DEBUG_TYPE
