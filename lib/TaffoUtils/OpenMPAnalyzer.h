//===-- IPO/OpenMPOpt.cpp - Collection of OpenMP specific optimizations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

/// Used to map the values physically (in the IR) stored in an offload
/// array, to a vector in memory.
#include "llvm/IR/Instructions.h"

struct OffloadArray;
// Maps the values stored in the offload arrays passed as arguments to
/// \p RuntimeCall into the offload arrays in \p OAs.
bool getValuesInOffloadArrays(llvm::CallInst &RuntimeCall,
                              llvm::MutableArrayRef<OffloadArray> OAs);

struct OffloadArray {

  /// Physical array (in the IR).
  llvm::AllocaInst *Array = nullptr;
  /// Mapped values.
  llvm::SmallVector<llvm::Value *, 8> StoredValues;
  /// Last stores made in the offload array.
  llvm::SmallVector<llvm::StoreInst *, 8> LastAccesses;

  OffloadArray() = default;

  static const unsigned DeviceIDArgNum = 1;
  static const unsigned BasePtrsArgNum = 4;
  static const unsigned PtrsArgNum = 5;
  static const unsigned SizesArgNum = 6;


private:
  /// Initializes the OffloadArray with the values stored in \p Array before
  /// instruction \p Before is reached. Returns false if the initialization
  /// fails.
  /// This MUST be used immediately after the construction of the object.
  bool initialize(llvm::AllocaInst &Array, llvm::Instruction &Before);


  /// Traverses the BasicBlock where \p Array is, collecting the stores made to
  /// \p Array, leaving StoredValues with the values stored before the
  /// instruction \p Before is reached.
  bool getValues(llvm::AllocaInst &Array, llvm::Instruction &Before);


  /// Returns true if all values in StoredValues and
  /// LastAccesses are not nullptrs.
  bool isFilled();

  friend bool getValuesInOffloadArrays(llvm::CallInst &RuntimeCall,
                                       llvm::MutableArrayRef<OffloadArray> OAs);
};
