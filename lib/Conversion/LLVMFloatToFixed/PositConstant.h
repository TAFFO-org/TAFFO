#include "FixedPointType.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"


namespace PositConstant {
  llvm::Constant *get(llvm::LLVMContext &C, const flttofix::FixedPointType &fixpt, double floatVal);
  llvm::Constant *FoldBinOp(llvm::LLVMContext &C, const flttofix::FixedPointType &fixpt,
                            int opcode, llvm::Constant *c1, llvm::Constant *c2);
  llvm::Constant *FoldUnaryOp(llvm::LLVMContext &C, const flttofix::FixedPointType &fixpt,
                            int opcode, llvm::Constant *c);
  llvm::Constant *FoldConv(llvm::LLVMContext &C, const llvm::DataLayout *dl, const flttofix::FixedPointType &fixpt,
                           llvm::Constant *src, llvm::Type *dstType);
}
