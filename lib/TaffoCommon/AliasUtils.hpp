//
// Created by jonathan on 16/04/2026.
//

#ifndef TAFFO_ALIASFUNCTION_HPP
#define TAFFO_ALIASFUNCTION_HPP

#include <llvm/IR/Intrinsics.h>

namespace aliasUtils::Intrinsic {
#if LLVM_VERSION_MAJOR < 22
constexpr auto getOrInsertDeclaration = llvm::Intrinsic::getDeclaration;
#else
constexpr auto getOrInsertDeclaration =
  static_cast<llvm::Function* (*) (llvm::Module * M, llvm::Intrinsic::ID id, llvm::ArrayRef<llvm::Type*> Tys)>(
    llvm::Intrinsic::getOrInsertDeclaration);
#endif
}
#else

#endif // TAFFO_ALIASFUNCTION_HPP
