#include "RangeOperations.hpp"
#include "VRAStore.hpp"
#include "ValueRangeAnalysisPass.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "taffo-vra"

using namespace llvm;
using namespace tda;
using namespace taffo;

void VRAStore::convexMerge(const VRAStore& other) {
  for (const auto& [value, otherValueInfo] : other.DerivedRanges) {
    std::shared_ptr<ValueInfo> currentInfo = this->getNode(value);
    //std::shared_ptr<ValueInfo> finalInfo = currentInfo;

    if (currentInfo) {
      if (std::isa_ptr<StructInfo>(currentInfo)) {
        assignStructNode(currentInfo, otherValueInfo);
        //finalInfo = currentInfo;
      }
      else if (std::isa_ptr<ArrayInfo>(currentInfo)) {
        assignArrayNode(currentInfo, otherValueInfo);
      }
      else if (std::shared_ptr<ScalarInfo> unionInfo = assignScalarRange(currentInfo, otherValueInfo)) {
        DerivedRanges[value] = unionInfo;
        //finalInfo = unionInfo;
      }
    }
    else {
      DerivedRanges[value] = otherValueInfo;
      //finalInfo = otherValueInfo;
    }

    // remove comment in case of desperate debugging
    // LLVM_DEBUG({
    //   const BasicBlock* BB = nullptr;
    //   const Function* F = nullptr;
    //   if (const auto* I = dyn_cast<Instruction>(value)) {
    //     BB = I->getParent();
    //     F = I->getFunction();
    //   }
    //   else if (const auto* Arg = dyn_cast<Argument>(value)) {
    //     F = Arg->getParent();
    //   }
    //   const std::string BBName = BB && BB->hasName() ? BB->getName().str() : "";
    //   const std::string FnName = F ? F->getName().str() : "";

    //   Logger->lineHead();
    //   tda::log() << "\nmerge " << *value << " GlobalStore=";
    //   Logger->logRange(currentInfo);
    //   tda::log() << " Incoming";
    //   if (!BBName.empty() || !FnName.empty()) {
    //     tda::log() << " from ";
    //     if (!BBName.empty())
    //       tda::log() << BBName;
    //     if (!FnName.empty()) {
    //       if (!BBName.empty())
    //         tda::log() << "/";
    //       tda::log() << FnName;
    //     }
    //   }
    //   tda::log() << ":";
    //   Logger->logRange(otherValueInfo);
    //   tda::log() << " Final=";
    //   Logger->logRange(finalInfo);
    //   tda::log() << "\n\n";
    // });
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
  case ValueInfo::K_Scalar: return valueInfo;
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
  case ValueInfo::K_Array:
    if (Offset.empty()) {
      return valueInfo;
    }
    else {
      std::shared_ptr<ArrayInfo> ArrayNode = std::static_ptr_cast<ArrayInfo>(valueInfo);
      unsigned index = Offset.back();

      if (index >= ArrayNode->getNumElements())
        return nullptr; // out of bounds

      std::shared_ptr<ValueInfo> Element = ArrayNode->getElement(index);
      Offset.pop_back();
      if (Offset.empty())
        return Element;
      else
        return loadNode(Element, Offset);
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
  default: llvm_unreachable("Unhandled node type.");
  }
}

std::shared_ptr<ScalarInfo> VRAStore::assignScalarRange(const std::shared_ptr<ValueInfo>& dst, const std::shared_ptr<ValueInfo>& src) const {
  std::shared_ptr<ScalarInfo> scalarDst = std::dynamic_ptr_cast_or_null<ScalarInfo>(dst);
  const std::shared_ptr<ScalarInfo> scalarSrc = std::dynamic_ptr_cast_or_null<ScalarInfo>(src);
  if (!scalarDst || !scalarSrc)
    return nullptr;
  if (scalarDst->isFinal())
    return scalarDst;

  std::shared_ptr<Range> unionRange;
  if (scalarDst->range && scalarSrc->range)
    unionRange = scalarDst->range->join(scalarSrc->range);
  else if (scalarDst->range)
    unionRange = scalarDst->range->clone();
  else if (scalarSrc->range)
    unionRange = scalarSrc->range->clone();
  else
    return scalarDst;
  return std::make_shared<ScalarInfo>(nullptr, unionRange);
}

void VRAStore::assignArrayNode(const std::shared_ptr<ValueInfo>& dst, const std::shared_ptr<ValueInfo>& src) const {
  const std::shared_ptr<ArrayInfo> arraySrc = std::dynamic_ptr_cast_or_null<ArrayInfo>(src);
  std::shared_ptr<ArrayInfo> arrayDst = std::dynamic_ptr_cast_or_null<ArrayInfo>(dst);
  if (!(arrayDst && arraySrc))
    return;

  unsigned limit = std::min(arrayDst->getNumElements(), arraySrc->getNumElements());
  for (unsigned i = 0; i < limit; i++) {
    std::shared_ptr<ValueInfo> srcElement = arraySrc->getElement(i);
    if (!srcElement)
      continue;
    std::shared_ptr<ValueInfo> dstElement = arrayDst->getElement(i);
    if (!dstElement)
      arrayDst->setElement(i, srcElement);
    else if (std::isa_ptr<StructInfo>(dstElement))
      assignStructNode(dstElement, srcElement);
    else if (std::isa_ptr<ArrayInfo>(dstElement))
      assignArrayNode(dstElement, srcElement);
    else if (std::shared_ptr<ValueInfo> unionElement = assignScalarRange(dstElement, srcElement))
      arrayDst->setElement(i, unionElement);
  }
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
  case ValueInfo::K_Array: {
    std::shared_ptr<ArrayInfo> arrayDst = std::static_ptr_cast<ArrayInfo>(dst);
    if (offset.empty())
      assignArrayNode(arrayDst, src);
    else if (offset.size() == 1) {
      unsigned index = offset.front();
      if (index < arrayDst->getNumElements()) {
        if (std::shared_ptr<ValueInfo> unionInfo = assignScalarRange(arrayDst->getElement(index), src))
          arrayDst->setElement(index, unionInfo);
        else
          arrayDst->setElement(index, src);
      }
    }
    else {
      unsigned index = offset.back();
      if (index < arrayDst->getNumElements()) {
        std::shared_ptr<ValueInfo> element = arrayDst->getElement(index);
        if (!element) {
          element = std::make_shared<ArrayInfo>(0);
          arrayDst->setElement(index, element);
        }
        offset.pop_back();
        storeNode(element, src, offset);
      }
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
  default: LLVM_DEBUG(log() << "WARNING: trying to store into a non-pointer node, aborted.\n");
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
  case ValueInfo::K_Scalar: return std::static_ptr_cast<ScalarInfo>(valueInfo);
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
  case ValueInfo::K_Array: {
    std::shared_ptr<ArrayInfo> ArrayNode = std::static_ptr_cast<ArrayInfo>(valueInfo);
    if (offset.empty()) {
      return ArrayNode;
    }
    else {
      unsigned index = offset.back();
      if (index >= ArrayNode->getNumElements())
        return nullptr; // out of bounds
      std::shared_ptr<ValueInfo> element = ArrayNode->getElement(index);
      offset.pop_back();
      return fetchRange(element, offset);
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
  default: llvm_unreachable("Unhandled node type.");
  }
}

bool VRAStore::extractGEPOffset(const Type* sourceElementType,
                                const iterator_range<User::const_op_iterator> indices,
                                SmallVectorImpl<unsigned>& offset) const {
  assert(sourceElementType != nullptr);
  LLVM_DEBUG(log() << "indices: ");

  auto indicesIter = indices.begin();
  
  if (const ConstantInt* intConstant = dyn_cast<ConstantInt>(*indicesIter)) {
    int val = static_cast<int>(intConstant->getSExtValue());
    if (val != 0) {
      offset.push_back(val);
    }
  } else {
    LLVM_DEBUG(Logger->logErrorln("Index of GEP not constant"));
    return false;
  }
  
  indicesIter++; // Passiamo agli indici strutturali
  
  for (; indicesIter != indices.end(); indicesIter++) {
    if (isa<VectorType>(sourceElementType))
      continue;

    if (const ConstantInt* intConstant = dyn_cast<ConstantInt>(*indicesIter)) {
      int val = static_cast<int>(intConstant->getSExtValue());
      offset.push_back(val);

      if (auto* structType = dyn_cast<StructType>(sourceElementType)) {
        sourceElementType = structType->getTypeAtIndex(val);
      } else if (auto* arrayType = dyn_cast<ArrayType>(sourceElementType)) {
        sourceElementType = arrayType->getElementType();
      } 
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