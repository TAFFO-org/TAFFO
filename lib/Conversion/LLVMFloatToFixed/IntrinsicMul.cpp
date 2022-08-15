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

// todo: leave only signed mul intrinsic
// for that I need to convert the result type to signed and then back to unsigned
// that will affect the number of fractional bits
// then I need to support sdiv intrinsic
// then compile polybench provided by Galimberti with their magiclang script

// this function converts fixed-point arguments to the same format to be used in the intrinsic call
Value* FloatToFixed::TransformToMulIntrinsic(Value *val1,
                                             Value *val2,
                        const FixedPointType& type1,
                        const FixedPointType& type2,
                                             Instruction *instr,
                        const FixedPointType &result_type) {
  FixedPointType signedType1 = ToSigned(type1);
  FixedPointType signedType2 = ToSigned(type2);
  FixedPointType signedResultType = ToSigned(result_type);
  Value* signedVal1 = genConvertFixedToFixed(val1, type1, signedType1, instr);
  Value* signedVal2 = genConvertFixedToFixed(val2, type2, signedType2, instr);
  return TransformToMulIntrinsicOpSigned(signedVal1, signedVal2, signedType1, signedType2, instr, signedResultType);
}

FixedPointType FloatToFixed::ToSigned(const FixedPointType& type) {
  if (type.scalarIsSigned()) return type;
  int fracBits = type.scalarFracBitsAmt() - 1;
  fracBits = fracBits < 0 ? 0 : fracBits;
  FixedPointType signedType(true, fracBits, type.scalarBitsAmt());
  return signedType;
}

Value* FloatToFixed::TransformToMulIntrinsicOpSigned(Value *val1,
                                                     Value *val2,
                                          const FixedPointType& type1,
                                          const FixedPointType& type2,
                                                     Instruction *instr,
                                          const FixedPointType &result_type) {
  assert(type1.isFixedPoint() && type2.isFixedPoint() && result_type.isFixedPoint() &&
         "can only replace with intrinsic when arguments are fixed-point");
  // ensure that both arguments have the same bit width
  int result_bits = std::max(std::max(type1.scalarBitsAmt(), type2.scalarBitsAmt()), result_type.scalarBitsAmt());
  int result_sign_bits = result_type.scalarIsSigned()? 1 : 0;
  // we need to ensure that both arguments are signed or unsigned
  // in case we have to convert from unsigned into signed we reduce the number of fractional bits by 1
  int new_type1_frac_bits = type1.scalarIsSigned()?
                                                   type1.scalarFracBitsAmt() : type1.scalarFracBitsAmt() - result_sign_bits;
  int new_type2_frac_bits = type2.scalarIsSigned()?
                                                   type2.scalarFracBitsAmt() : type2.scalarFracBitsAmt() - result_sign_bits;
  // we have to preserve the integer part as best as possible,
  // so result will have the least of its parameters fractional bits count
  int result_frac_bits = std::min(
      std::min(std::min(new_type1_frac_bits, new_type2_frac_bits), result_type.scalarFracBitsAmt()),
      result_type.scalarBitsAmt());
  result_frac_bits = std::max(result_frac_bits, 0);
  FixedPointType output_type(result_sign_bits, result_frac_bits, result_bits);
  Value* ext1 = genConvertFixedToFixed(val1, type1, output_type, instr);
  Value* ext2 = genConvertFixedToFixed(val2, type2, output_type, instr);
  cpMetaData(ext1, val1);
  cpMetaData(ext2, val2);
  return ToMulIntrinsic(ext1, ext2, instr, output_type);
}

// this function converts the instruction into an intrinsic implementation
// it expects parameters of the instruction and the result to be of the same fixed-point type
Value* FloatToFixed::ToMulIntrinsic(Value *val1,
                                    Value *val2,
                                    Instruction *instr,
                        const FixedPointType &result_type) {
  IRBuilder<> builder(instr);
  // create list of formal argument types
  Type *Tys[] ={builder.getIntNTy(result_type.scalarBitsAmt())};
  // create the funcCall as intrinsic
  Function *IntrFunc = Intrinsic::getDeclaration(
      instr->getParent()->getParent()->getParent(),
      //IInsertBefore->getParent() returns the BasicBlock
      result_type.scalarIsSigned()? Intrinsic::smul_fix : Intrinsic::umul_fix,
      Tys);
  errs() << "---SMUL.TYPES---" << "\n";
  errs() << result_type.toString() << "\n";
  errs() << "--------" << "\n";
  Value* fixop = builder.CreateCall(IntrFunc,
     {val1, val2, builder.getInt32(result_type.scalarFracBitsAmt())});
  cpMetaData(fixop, instr);
  updateFPTypeMetadata(fixop, result_type.scalarIsSigned(),
                       result_type.scalarFracBitsAmt(),
                       result_type.scalarBitsAmt());
  updateConstTypeMetadata(fixop, 0U, result_type);
  updateConstTypeMetadata(fixop, 1U, result_type);
  return fixop;
}


