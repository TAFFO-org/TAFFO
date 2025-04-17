//===- Mem2Reg.h - The -mem2reg pass, a wrapper around the Utils lib ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

namespace taffo {

class MemToRegPass : public llvm::PassInfoMixin<MemToRegPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager& AM);
};

} // end namespace taffo
