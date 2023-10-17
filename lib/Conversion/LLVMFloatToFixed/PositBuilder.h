#ifndef POSIT_BUILDER
#define POSIT_BUILDER

#include "FixedPointType.h"
#include "LLVMFloatToFixedPass.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace flttofix
{

class PositBuilder {
public:
  PositBuilder(FloatToFixed *pass, llvm::IRBuilderBase &builder, const FixedPointType &metadata)
    : builder(builder)
    , C(this->builder.getContext())
    , M(this->builder.GetInsertBlock()->getParent()->getParent())
    , pass(pass)
    , metadata(metadata)
    , llvmType(llvm::cast<llvm::StructType>(this->metadata.scalarToLLVMType(builder.getContext())))
  {}

  virtual llvm::Value *CreateConstructor(llvm::Value *arg1, const FixedPointType *srcMetadata = nullptr) = 0;
  virtual llvm::Value *CreateConv(llvm::Value *from, llvm::Type *dstType, const FixedPointType *dstMetadata = nullptr) = 0;
  virtual llvm::Value *CreateBinOp(int opcode, llvm::Value *arg1, llvm::Value *arg2) = 0;
  virtual llvm::Value *CreateUnaryOp(int opcode, llvm::Value *arg1) = 0;
  virtual llvm::Value *CreateCmp(llvm::CmpInst::Predicate pred, llvm::Value *arg1, llvm::Value *arg2) = 0;
  virtual llvm::Value *CreateFMA(llvm::Value *arg1, llvm::Value *arg2, llvm::Value *arg3) = 0;

  static std::unique_ptr<PositBuilder> get(FloatToFixed *pass, llvm::IRBuilderBase &builder, const FixedPointType &metadata);

protected:
  llvm::IRBuilderBase &builder;
  llvm::LLVMContext &C;
  llvm::Module *M;
  FloatToFixed *pass;

  const FixedPointType &metadata;
  llvm::StructType *llvmType;
};

}

#endif
