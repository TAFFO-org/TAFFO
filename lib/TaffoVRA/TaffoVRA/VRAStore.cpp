#include "RangeOperations.hpp"
#include "VRAStore.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace taffo;

void VRAStore::convexMerge(const VRAStore& other) {
  for (const auto& [value, otherValueInfo] : other.DerivedRanges) {
    if (std::shared_ptr<ValueInfo> valueInfo = this->getNode(value)) {
      if (std::isa_ptr<StructInfo>(valueInfo))
        assignStructNode(valueInfo, otherValueInfo);
      else if (std::shared_ptr<ScalarInfo> unionInfo = assignScalarRange(valueInfo, otherValueInfo))
        DerivedRanges[value] = unionInfo;
    }
    else {
      DerivedRanges[value] = otherValueInfo;
    }
  }
}

std::shared_ptr<Range> VRAStore::fetchRange(const Value* v) {
  if (const std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast_or_null<ScalarInfo>(fetchRangeNode(v)))
    return scalarInfo->range;
  return nullptr;
}

std::shared_ptr<ValueInfoWithRange> VRAStore::fetchRangeNode(const Value* v) {
  if (const std::shared_ptr<ValueInfo> valueInfo = getNode(v)) {
    if (std::shared_ptr<ScalarInfo> Scalar = std::dynamic_ptr_cast<ScalarInfo>(valueInfo))
      return Scalar;
    if (v->getType()->isPointerTy())
      return fetchRange(valueInfo);
  }
  // no info available
  return nullptr;
}

void VRAStore::saveValueRange(const Value* v, const std::shared_ptr<Range> range) {
  if (!range)
    return;
  // TODO: make specialized version of this to avoid creating useless node
  saveValueRange(v, std::make_shared<ScalarInfo>(nullptr, range));
}

void VRAStore::saveValueRange(const Value* v, const std::shared_ptr<ValueInfoWithRange> valueInfoWithRange) {
  assert(v && "Trying to save range for null value.");
  if (!valueInfoWithRange)
    return;
  if (std::shared_ptr<ValueInfo> unionInfo = assignScalarRange(getNode(v), valueInfoWithRange)) {
    DerivedRanges[v] = unionInfo;
    return;
  }
  DerivedRanges[v] = valueInfoWithRange;
}

std::shared_ptr<ValueInfo> VRAStore::getNode(const Value* v) {
  assert(v && "Trying to get node for null value.");
  const auto it = DerivedRanges.find(v);
  if (it != DerivedRanges.end())
    return it->second;
  return nullptr;
}

void VRAStore::setNode(const Value* V, std::shared_ptr<ValueInfo> Node) { DerivedRanges[V] = Node; }

std::shared_ptr<ValueInfo> VRAStore::loadNode(const std::shared_ptr<ValueInfo> Node) const {
  SmallVector<unsigned, 1U> Offset;
  return loadNode(Node, Offset);
}

std::shared_ptr<ValueInfo> VRAStore::loadNode(const std::shared_ptr<ValueInfo>& valueInfo,
                                              SmallVectorImpl<unsigned>& Offset) const {
  if (!valueInfo)
    return nullptr;
  switch (valueInfo->getKind()) {
  case ValueInfo::K_Scalar:
    return valueInfo;
  case ValueInfo::K_Struct:
    if (Offset.empty()) {
      return valueInfo;
    }
    else {
      std::shared_ptr<StructInfo> StructNode = std::static_ptr_cast<StructInfo>(valueInfo);
      std::shared_ptr<ValueInfo> Field = StructNode->getField(Offset.back());
      Offset.pop_back();
      if (Offset.empty())
        return Field;
      else
        return loadNode(Field, Offset);
    }
  case ValueInfo::K_GetElementPointer: {
    std::shared_ptr<GEPInfo> gepInfo = std::static_ptr_cast<GEPInfo>(valueInfo);
    const ArrayRef<unsigned> gepOffset = gepInfo->getOffset();
    Offset.append(gepOffset.begin(), gepOffset.end());
    return loadNode(gepInfo->getPointed(), Offset);
  }
  case ValueInfo::K_Pointer: {
    std::shared_ptr<PointerInfo> pointerInfo = std::static_ptr_cast<PointerInfo>(valueInfo);
    return pointerInfo->getPointed();
  }
  default:
    llvm_unreachable("Unhandled node type.");
  }
}

std::shared_ptr<ScalarInfo> VRAStore::assignScalarRange(const std::shared_ptr<ValueInfo>& dst,
                                                        const std::shared_ptr<ValueInfo>& src) const {
  std::shared_ptr<ScalarInfo> scalarDst = std::dynamic_ptr_cast_or_null<ScalarInfo>(dst);
  const std::shared_ptr<ScalarInfo> scalarSrc = std::dynamic_ptr_cast_or_null<ScalarInfo>(src);
  if (!scalarDst || !scalarSrc)
    return nullptr;
  if (scalarDst->isFinal())
    return scalarDst;
  std::shared_ptr<Range> unionRange = getUnionRange(scalarDst->range, scalarSrc->range);
  return std::make_shared<ScalarInfo>(nullptr, unionRange);
}

void VRAStore::assignStructNode(const std::shared_ptr<ValueInfo>& dst, const std::shared_ptr<ValueInfo>& src) const {
  const std::shared_ptr<StructInfo> structSrc = std::dynamic_ptr_cast_or_null<StructInfo>(src);
  std::shared_ptr<StructInfo> structDst = std::dynamic_ptr_cast_or_null<StructInfo>(dst);
  if (!(structDst && structSrc))
    return;
  for (unsigned i = 0; i < structSrc->getNumFields(); i++) {
    std::shared_ptr<ValueInfo> srcField = structSrc->getField(i);
    if (!srcField)
      continue;
    std::shared_ptr<ValueInfo> dstField = structDst->getField(i);
    if (!dstField)
      structDst->setField(i, srcField);
    else if (std::isa_ptr<StructInfo>(dstField))
      assignStructNode(dstField, srcField);
    else if (std::shared_ptr<ValueInfo> unionField = assignScalarRange(dstField, srcField))
      structDst->setField(i, unionField);
  }
}

void VRAStore::storeNode(const std::shared_ptr<ValueInfo> dst, const std::shared_ptr<ValueInfo>& src) {
  SmallVector<unsigned, 1U> Offset;
  storeNode(dst, src, Offset);
}

void VRAStore::storeNode(const std::shared_ptr<ValueInfo>& dst,
                         const std::shared_ptr<ValueInfo>& src,
                         SmallVectorImpl<unsigned>& offset) {
  if (!(dst && src))
    return;
  std::shared_ptr<ValueInfo> pointed = nullptr;
  switch (dst->getKind()) {
  case ValueInfo::K_GetElementPointer: {
    std::shared_ptr<GEPInfo> gepInfo = std::static_ptr_cast<GEPInfo>(dst);
    const ArrayRef<unsigned> gepOffset = gepInfo->getOffset();
    offset.append(gepOffset.begin(), gepOffset.end());
    storeNode(gepInfo->getPointed(), src, offset);
    break;
  }
  case ValueInfo::K_Struct: {
    std::shared_ptr<StructInfo> structDst = std::static_ptr_cast<StructInfo>(dst);
    if (offset.empty()) {
      assignStructNode(structDst, src);
    }
    else if (offset.size() == 1) {
      unsigned index = offset.front();
      if (std::shared_ptr<ValueInfo> unionInfo = assignScalarRange(structDst->getField(index), src))
        structDst->setField(index, unionInfo);
      else
        structDst->setField(index, src);
    }
    else {
      std::shared_ptr<ValueInfo> field = structDst->getField(offset.back());
      if (!field) {
        field = std::make_shared<StructInfo>(0);
        structDst->setField(offset.back(), field);
      }
      offset.pop_back();
      storeNode(field, src, offset);
    }
    break;
  }
  case ValueInfo::K_Pointer: {
    std::shared_ptr<PointerInfo> pointerDst = std::static_ptr_cast<PointerInfo>(dst);
    if (std::shared_ptr<ValueInfo> unionInfo = assignScalarRange(pointerDst->getPointed(), src))
      pointerDst->setPointed(unionInfo);
    else
      pointerDst->setPointed(src);
    break;
  }
  default:
    LLVM_DEBUG(log() << "WARNING: trying to store into a non-pointer node, aborted.\n");
  }
}

std::shared_ptr<ValueInfoWithRange> VRAStore::fetchRange(const std::shared_ptr<ValueInfo> valueInfo) const {
  SmallVector<unsigned, 1> offset;
  return fetchRange(valueInfo, offset);
}

std::shared_ptr<ValueInfoWithRange> VRAStore::fetchRange(const std::shared_ptr<ValueInfo>& valueInfo,
                                                         SmallVectorImpl<unsigned>& offset) const {
  if (!valueInfo)
    return nullptr;
  switch (valueInfo->getKind()) {
  case ValueInfo::K_Scalar:
    return std::static_ptr_cast<ScalarInfo>(valueInfo);
  case ValueInfo::K_Struct: {
    std::shared_ptr<StructInfo> StructNode = std::static_ptr_cast<StructInfo>(valueInfo);
    if (offset.empty()) {
      return StructNode;
    }
    else {
      std::shared_ptr<ValueInfo> field = StructNode->getField(offset.back());
      offset.pop_back();
      return fetchRange(field, offset);
    }
  }
  case ValueInfo::K_GetElementPointer: {
    std::shared_ptr<GEPInfo> GEPNode = std::dynamic_ptr_cast<GEPInfo>(valueInfo);
    const ArrayRef<unsigned> GEPOffset = GEPNode->getOffset();
    offset.append(GEPOffset.begin(), GEPOffset.end());
    return fetchRange(GEPNode->getPointed(), offset);
  }
  case ValueInfo::K_Pointer: {
    std::shared_ptr<PointerInfo> PtrNode = std::dynamic_ptr_cast<PointerInfo>(valueInfo);
    return fetchRange(PtrNode->getPointed(), offset);
  }
  default:
    llvm_unreachable("Unhandled node type.");
  }
}

bool VRAStore::extractGEPOffset(const Type* sourceElementType,
                                const iterator_range<User::const_op_iterator> indices,
                                SmallVectorImpl<unsigned>& offset) const {
  assert(sourceElementType != nullptr);
  LLVM_DEBUG(log() << "indices: ");
  for (auto indicesIter = indices.begin() + 1; // skip first index
       indicesIter != indices.end();
       indicesIter++) {
    if (isa<ArrayType>(sourceElementType) || isa<VectorType>(sourceElementType))
      continue;
    if (const ConstantInt* intConstant = dyn_cast<ConstantInt>(*indicesIter)) {
      int val = static_cast<int>(intConstant->getSExtValue());
      offset.push_back(val);
      sourceElementType = cast<StructType>(sourceElementType)->getTypeAtIndex(val);
      LLVM_DEBUG(log() << val << " ");
    }
    else {
      LLVM_DEBUG(Logger->logErrorln("Index of GEP not constant"));
      return false;
    }
  }
  LLVM_DEBUG(log() << "\n");
  return true;
}
