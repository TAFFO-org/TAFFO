#include "LLVMFloatToFixedPass.h"
#include "TypeUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;
using namespace taffo;
using namespace flttofix;

// this function converts fixed-point arguments to the same format to be used in the intrinsic call
Value* FloatToFixed::TransformToIntrinsic(Value* val1,
                        Value* val2,
                        const FixedPointType& type1,
                        const FixedPointType& type2,
                        Instruction* instr) {
  assert(type1.isFixedPoint() && type2.isFixedPoint() &&
         "can only replace with intrinsic when arguments are fixed-point");
  // ensure that both arguments have the same bit width
  int result_bits = std::max(type1.scalarBitsAmt(), type2.scalarBitsAmt());
  int result_sign_bits = type1.scalarIsSigned() || type1.scalarIsSigned()? 1 : 0;
  // we need to ensure that both arguments are signed or unsigned
  // in case we have to convert from unsigned into signed we reduce the number of fractional bits by 1
  int new_type1_frac_bits = type1.scalarIsSigned()?
      type1.scalarFracBitsAmt() : type1.scalarFracBitsAmt() - result_sign_bits;
  int new_type2_frac_bits = type2.scalarIsSigned()?
      type2.scalarFracBitsAmt() : type2.scalarFracBitsAmt() - result_sign_bits;
  // we have to preserve the integer part as best as possible,
  // so result will have the least of its parameters fractional bits count
  int result_frac_bits = std::min(new_type1_frac_bits, new_type2_frac_bits);
  FixedPointType result_type(result_sign_bits, result_frac_bits, result_bits);
  Value* ext1 = genConvertFixedToFixed(val1, type1, result_type, instr);
  Value* ext2 = genConvertFixedToFixed(val2, type2, result_type, instr);
  cpMetaData(ext1, val1);
  cpMetaData(ext2, val2);
  return ToIntrinsic(ext1, ext2, result_type, result_type, instr);
}

// this function converts the instruction into an intrinsic implementation
// it expects parameters of the instruction to be of the same fixed-point type
Value* FloatToFixed::ToIntrinsic(Value* val1,
                        Value* val2,
                        const FixedPointType& type1,
                        const FixedPointType& type2,
                        Instruction* instr) {
  assert(type1.scalarBitsAmt() == type2.scalarBitsAmt() &&
         "types should have the same bit width");
  assert(type1.scalarFracBitsAmt() == type2.scalarFracBitsAmt() &&
         "types should have the same fractional bits count");
  assert(type1.scalarIsSigned() == type2.scalarIsSigned() &&
         "types should be both signed or both unsigned");
  IRBuilder<> builder(instr);
  // create list of formal argument types
  Type *Tys[] ={builder.getIntNTy(type1.scalarBitsAmt())};
  // create the funcCall as intrinsic
  Function *IntrFunc = Intrinsic::getDeclaration(
      instr->getParent()->getParent()->getParent(),
      //IInsertBefore->getParent() returns the BasicBlock
      type1.scalarIsSigned()? Intrinsic::smul_fix : Intrinsic::umul_fix,
      Tys);
  Value* fixop = builder.CreateCall(IntrFunc,
     {val1, val2, builder.getInt32(type1.scalarFracBitsAmt())});
  cpMetaData(fixop, instr);
  updateFPTypeMetadata(fixop, type1.scalarIsSigned(),
                       type1.scalarFracBitsAmt(),
                       type1.scalarBitsAmt());
  updateConstTypeMetadata(fixop, 0U, type1);
  updateConstTypeMetadata(fixop, 1U, type2);
  return fixop;
}


