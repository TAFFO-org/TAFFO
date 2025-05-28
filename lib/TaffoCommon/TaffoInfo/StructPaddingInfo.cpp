#include "StructPaddingInfo.hpp"

#include <llvm/BinaryFormat/Dwarf.h>

using namespace llvm;
using namespace taffo;

StructPaddingInfo::StructPaddingInfo(const DICompositeType* diCompositeType) {
  const unsigned structSize = diCompositeType->getSizeInBits() / 8;

  struct FieldInfo {
    unsigned offset;
    unsigned size;
  };
  SmallVector<FieldInfo, 8> fieldsInfo;

  for (const Metadata* md : diCompositeType->getElements()) {
    const auto* derived = dyn_cast<DIDerivedType>(md);
    if (!derived || derived->getTag() != dwarf::DW_TAG_member)
      continue; // skip anything that is not a field (enum, typedef, ...)
    unsigned offset = derived->getOffsetInBits() / 8;
    unsigned size = derived->getSizeInBits() / 8;
    fieldsInfo.emplace_back(offset, size);
  }

  if (fieldsInfo.empty())
    return;

  sort(fieldsInfo, [](const FieldInfo& a, const FieldInfo& b) { return a.offset < b.offset; });

  unsigned currOffset = 0;
  for (const FieldInfo& fieldInfo : fieldsInfo) {
    if (fieldInfo.offset > currOffset)
      paddingByteRanges.emplace_back(currOffset, fieldInfo.offset);
    currOffset = fieldInfo.offset + fieldInfo.size;
  }
  if (currOffset < structSize)
    paddingByteRanges.emplace_back(currOffset, structSize);
}

json StructPaddingInfo::serialize() const {
  json j;
  j["kind"] = "StructPaddingInfo";
  j["ranges"] = json::array();
  for (auto [b, e] : paddingByteRanges)
    j["ranges"].push_back({b, e});
  return j;
}

void StructPaddingInfo::deserialize(const json& j) {
  paddingByteRanges.clear();
  for (auto& elem : j.at("ranges"))
    paddingByteRanges.emplace_back(elem[0].get<unsigned>(), elem[1].get<unsigned>());
}

std::string StructPaddingInfo::toString() const {
  std::stringstream ss;
  ss << "{ ";
  bool first = true;
  for (auto [b, e] : paddingByteRanges) {
    if (!first)
      ss << ", ";
    ss << "[" << b << "," << e << ")";
    first = false;
  }
  ss << " }";
  return ss.str();
}
