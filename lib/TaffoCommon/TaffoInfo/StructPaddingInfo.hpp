#pragma once

#include "TaffoCommon/SerializationUtils.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DebugInfoMetadata.h>

namespace taffo {

/**
 *  Compact description of the padding of a structure.
 *  All ranges are expressed in bytes and written as half-open intervals [begin, end).
 */
class StructPaddingInfo final : public Serializable,
                                public Printable {
public:
  using ByteRange = std::pair<unsigned, unsigned>;

  StructPaddingInfo() = default;
  StructPaddingInfo(const llvm::DICompositeType* diCompositeType);

  llvm::ArrayRef<ByteRange> getPaddingRanges() const { return paddingByteRanges; }

  json serialize() const override;
  void deserialize(const json& j) override;
  std::string toString() const override;

private:
  llvm::SmallVector<ByteRange, 4> paddingByteRanges;
};

} // namespace taffo
