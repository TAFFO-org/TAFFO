#include "TAFFOMath.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>
#include <vector>

#define DEBUG_TYPE "taffo-conversion"




namespace flttofix
{


void fixrangeSinCos(FloatToFixed *ref, Function *newfs, FixedPointType &fxparg,
                    FixedPointType &fxpret, Value *arg_value,
                    llvm::IRBuilder<> &builder)
{
  assert(fxparg.scalarBitsAmt() == fxpret.scalarBitsAmt() &&
         "different type arg and ret");
  int int_lenght = fxparg.scalarBitsAmt();
  Module *M = newfs->getParent();
  llvm::LLVMContext &cont = newfs->getContext();
  DataLayout dataLayout(M);
  auto int_type = fxparg.scalarToLLVMType(cont);
  Value *generic = nullptr;
  int max = fxparg.scalarFracBitsAmt() > fxpret.scalarFracBitsAmt()
                ? fxparg.scalarFracBitsAmt()
                : fxpret.scalarFracBitsAmt();
  int min = fxparg.scalarFracBitsAmt() < fxpret.scalarFracBitsAmt()
                ? fxparg.scalarFracBitsAmt()
                : fxpret.scalarFracBitsAmt();

  max = max >= fxpret.scalarBitsAmt() - 4 ? fxpret.scalarBitsAmt() - 4 : max;
  bool can_continue = true;
  if (min > fxpret.scalarBitsAmt() - 4) {
    min = fxpret.scalarBitsAmt() - 4;
    can_continue = false;
  }


  LLVM_DEBUG(dbgs() << "Max: " << max << " Min: " << min << "\n");
  // create pi_2 table
  TaffoMath::pair_ftp_value<llvm::Constant *, 5> pi_2_vect;
  int created = 0;
  for (int i = 0; i <= max - min && can_continue; ++i) {
    pi_2_vect.fpt.push_back(
        flttofix::FixedPointType(fxparg.scalarIsSigned(), min + i, int_lenght));
    Constant *tmp = nullptr;
    flttofix::FixedPointType match = flttofix::FixedPointType(fxparg.scalarIsSigned(), min + i, int_lenght);
    auto &current_fpt = pi_2_vect.fpt.front();
    bool p2_creted = TaffoMath::createFixedPointFromConst(
        cont, ref, TaffoMath::pi_2, match, tmp, current_fpt);
    if (p2_creted) {
      pi_2_vect.value.push_back(tmp);
      created += 1;
    }
  }
  if (created != 0) {
    auto pi_2_ArrayType =
        llvm::ArrayType::get(pi_2_vect.value.front()->getType(), max - min + 1);
    LLVM_DEBUG(dbgs() << "ArrayType  " << pi_2_ArrayType << "\n");
    auto pi_2_ConstArray = llvm::ConstantArray::get(
        pi_2_ArrayType, llvm::ArrayRef<llvm::Constant *>(pi_2_vect.value));
    LLVM_DEBUG(dbgs() << "ConstantDataArray pi_2 " << pi_2_ConstArray << "\n");
    auto alignement_pi_2 =
        dataLayout.getPrefTypeAlignment(pi_2_vect.value.front()->getType());
    LLVM_DEBUG(dbgs() << (pi_2_ArrayType->dump(), pi_2_ConstArray->dump(), "alignment: ") << alignement_pi_2 << "\n");
    auto pi_2_arry_g =
        TaffoMath::createGlobalConst(M, "pi_2_global." + std::to_string(min) + "_" + std::to_string(max), pi_2_ArrayType,
                                     pi_2_ConstArray, alignement_pi_2);
    auto pointer_to_array = TaffoMath::addAllocaToStart(ref, newfs, builder, pi_2_ArrayType, nullptr, "pi_2_array");
    dyn_cast<llvm::AllocaInst>(pointer_to_array)->setAlignment(llvm::Align(alignement_pi_2));
    builder.CreateMemCpy(
        pointer_to_array, llvm::Align(alignement_pi_2), pi_2_arry_g, llvm::Align(alignement_pi_2),
        (max - min + 1) * (int_type->getScalarSizeInBits() >> 3));
    LLVM_DEBUG(dbgs() << "\nAdd pi_2 table"
                      << "\n");

    int current_arg_point = fxparg.scalarFracBitsAmt();
    int current_ret_point = fxpret.scalarFracBitsAmt();

    Value *iterator_pi_2 =
        TaffoMath::addAllocaToStart(ref, newfs, builder, int_type, nullptr, "Iterator_pi_2");
    Value *point_arg =
        TaffoMath::addAllocaToStart(ref, newfs, builder, int_type, nullptr, "point_arg");
    Value *point_ret =
        TaffoMath::addAllocaToStart(ref, newfs, builder, int_type, nullptr, "point_ret");

    builder.CreateStore(ConstantInt::get(int_type, current_arg_point), point_arg);
    builder.CreateStore(ConstantInt::get(int_type, current_ret_point), point_ret);

    if (current_arg_point < current_ret_point) {
      BasicBlock *normalize_cond =
          BasicBlock::Create(cont, "normalize_cond", newfs);
      BasicBlock *normalize = BasicBlock::Create(cont, "normalize", newfs);
      BasicBlock *cmp_bigger_than_2pi =
          BasicBlock::Create(cont, "cmp_bigger_than_2pi", newfs);
      BasicBlock *bigger_than_2pi =
          BasicBlock::Create(cont, "bigger_than_2pi", newfs);
      BasicBlock *end_loop = BasicBlock::Create(cont, "body", newfs);
      builder.CreateStore(ConstantInt::get(int_type, 0), iterator_pi_2);
      builder.CreateBr(normalize_cond);
      LLVM_DEBUG(dbgs() << "add Normalize\n");
      // normalize cond
      builder.SetInsertPoint(normalize_cond);
      Value *last_bit_mask =
          fxparg.scalarIsSigned()
              ? ConstantInt::get(int_type, 1 << (int_lenght - 2))
              : ConstantInt::get(int_type, 1 << (int_lenght - 1));
      generic = builder.CreateAnd(builder.CreateLoad(int_type, arg_value), last_bit_mask);
      generic = builder.CreateAnd(
          builder.CreateICmpEQ(generic, ConstantInt::get(int_type, 0)),
          builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(point_arg), point_arg),
                                builder.CreateLoad(getElementTypeFromValuePointer(point_ret), point_ret)));
      builder.CreateCondBr(generic, normalize, cmp_bigger_than_2pi);
      // normalize
      builder.SetInsertPoint(normalize);
      builder.CreateStore(builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            ConstantInt::get(int_type, 1)),
                          arg_value);
      builder.CreateStore(builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(iterator_pi_2), iterator_pi_2),
                                            ConstantInt::get(int_type, 1)),
                          iterator_pi_2);
      builder.CreateStore(builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(point_arg), point_arg),
                                            ConstantInt::get(int_type, 1)),
                          point_arg);
      builder.CreateBr(normalize_cond);
      LLVM_DEBUG(dbgs() << "add bigger than 2pi\n");
      // cmp_bigger_than_2pi
      builder.SetInsertPoint(cmp_bigger_than_2pi);
      generic = builder.CreateGEP(getElementTypeFromValuePointer(pointer_to_array),
                                  pointer_to_array,
                                  {ConstantInt::get(int_type, 0), builder.CreateLoad(getElementTypeFromValuePointer(iterator_pi_2), iterator_pi_2)});
      generic = builder.CreateLoad(getElementTypeFromValuePointer(generic), generic);
      builder.CreateCondBr(builder.CreateICmpSLE(generic,
                                                 builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value)),
                           bigger_than_2pi, end_loop);
      // bigger_than_2pi
      builder.SetInsertPoint(bigger_than_2pi);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            generic),
                          arg_value);
      builder.CreateBr(normalize_cond);
      builder.SetInsertPoint(end_loop);
    } else {
      LLVM_DEBUG(dbgs() << "add bigger than 2pi\n");
      BasicBlock *cmp_bigger_than_2pi =
          BasicBlock::Create(cont, "cmp_bigger_than_2pi", newfs);
      BasicBlock *bigger_than_2pi =
          BasicBlock::Create(cont, "bigger_than_2pi", newfs);
      BasicBlock *body = BasicBlock::Create(cont, "shift", newfs);
      builder.CreateStore(ConstantInt::get(int_type, 0), iterator_pi_2);
      builder.CreateBr(cmp_bigger_than_2pi);
      builder.SetInsertPoint(cmp_bigger_than_2pi);
      generic = builder.CreateGEP(getElementTypeFromValuePointer(pointer_to_array),
                                  pointer_to_array,
                                  {ConstantInt::get(int_type, 0), builder.CreateLoad(getElementTypeFromValuePointer(iterator_pi_2), iterator_pi_2)});
      generic = builder.CreateLoad(getElementTypeFromValuePointer(generic), generic);
      builder.CreateCondBr(builder.CreateICmpSLE(generic,
                                                 builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value)),
                           bigger_than_2pi, body);
      builder.SetInsertPoint(bigger_than_2pi);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            generic),
                          arg_value);
      builder.CreateBr(cmp_bigger_than_2pi);
      builder.SetInsertPoint(body);
    }
    // set point at same position
    {
      LLVM_DEBUG(dbgs() << "set point at same position\n");
      builder.CreateStore(ConstantInt::get(int_type, 0), iterator_pi_2);
      BasicBlock *body_2 = BasicBlock::Create(cont, "body", newfs);
      BasicBlock *true_block = BasicBlock::Create(cont, "left_shift", newfs);
      BasicBlock *false_block = BasicBlock::Create(cont, "right_shift", newfs);

      BasicBlock *end = BasicBlock::Create(cont, "body", newfs);
      builder.CreateCondBr(builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(point_arg), point_arg),
                                                builder.CreateLoad(getElementTypeFromValuePointer(point_ret), point_ret)),
                           end, body_2);
      builder.SetInsertPoint(body_2);
      builder.CreateCondBr(builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(point_arg), point_arg),
                                                 builder.CreateLoad(getElementTypeFromValuePointer(point_ret), point_ret)),
                           true_block, false_block);
      builder.SetInsertPoint(true_block);
      generic = builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(point_ret), point_ret),
                                  builder.CreateLoad(getElementTypeFromValuePointer(point_arg), point_arg));
      builder.CreateStore(
          builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), generic), arg_value);
      builder.CreateBr(end);
      builder.SetInsertPoint(false_block);
      generic = builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(point_arg), point_arg),
                                  builder.CreateLoad(getElementTypeFromValuePointer(point_ret), point_ret));
      builder.CreateStore(
          builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), generic), arg_value);
      builder.CreateBr(end);
      builder.SetInsertPoint(end);
    }
  } else {
    if (fxparg.scalarFracBitsAmt() > fxpret.scalarFracBitsAmt()) {
      builder.CreateStore(
          builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), fxparg.scalarFracBitsAmt() - fxpret.scalarFracBitsAmt()), arg_value);
    }
    if (fxparg.scalarFracBitsAmt() < fxpret.scalarFracBitsAmt()) {
      builder.CreateStore(
          builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), fxpret.scalarFracBitsAmt() - fxparg.scalarFracBitsAmt()), arg_value);
    }
  }
  LLVM_DEBUG(dbgs() << "End fixrange\n");
}


Value *generateSinLUT(FloatToFixed *ref, Function *newfs, FixedPointType &fxparg,
                      llvm::IRBuilder<> &builder)
{

  TaffoMath::pair_ftp_value<llvm::Constant *, 5> sin_vect;
  for (int i = 0; i < MathZ; ++i) {
    sin_vect.fpt.push_back(
        flttofix::FixedPointType(false, fxparg.scalarFracBitsAmt(), fxparg.scalarBitsAmt()));
    Constant *tmp = nullptr;
    flttofix::FixedPointType match = flttofix::FixedPointType(false, fxparg.scalarFracBitsAmt(), fxparg.scalarBitsAmt());
    auto &current_fpt = sin_vect.fpt.front();
    bool sin_creted = TaffoMath::createFixedPointFromConst(
        newfs->getContext(), ref, sin(i * TaffoMath::pi_half / MathZ), match, tmp, current_fpt);


    if (tmp == nullptr || !sin_creted) {
      llvm_unreachable("Ma nooooo\n");
    }
    sin_vect.value.push_back(tmp);
  }
  auto sin_ArrayType =
      llvm::ArrayType::get(fxparg.scalarToLLVMType(newfs->getContext()), MathZ);
  auto sin_ConstArray = llvm::ConstantArray::get(
      sin_ArrayType, llvm::ArrayRef<llvm::Constant *>(sin_vect.value));
  auto alignement_sin =
      newfs->getParent()->getDataLayout().getPrefTypeAlignment(sin_vect.value.front()->getType());
  auto sin_arry_g =
      TaffoMath::createGlobalConst(newfs->getParent(), "sin_global." + std::to_string(fxparg.scalarFracBitsAmt()) + "_" + std::to_string(fxparg.scalarBitsAmt()), sin_ArrayType,
                                   sin_ConstArray, alignement_sin);
  return sin_arry_g;
}


Value *generateCosLUT(FloatToFixed *ref, Function *newfs, FixedPointType &fxparg,
                      llvm::IRBuilder<> &builder)
{


  TaffoMath::pair_ftp_value<llvm::Constant *, 5> cos_vect;
  for (int i = 0; i < MathZ; ++i) {
    cos_vect.fpt.push_back(
        flttofix::FixedPointType(false, fxparg.scalarFracBitsAmt(), fxparg.scalarBitsAmt()));
    Constant *tmp = nullptr;
    flttofix::FixedPointType match = flttofix::FixedPointType(false, fxparg.scalarFracBitsAmt(), fxparg.scalarBitsAmt());
    auto &current_fpt = cos_vect.fpt.front();

    bool cos_creted = TaffoMath::createFixedPointFromConst(
        newfs->getContext(), ref, cos(i * TaffoMath::pi_half / MathZ), match, tmp, current_fpt);


    if (tmp == nullptr || !cos_creted) {
      llvm_unreachable("Ma nooooo\n");
    }
    cos_vect.value.push_back(tmp);
  }
  auto cos_ArrayType =
      llvm::ArrayType::get(fxparg.scalarToLLVMType(newfs->getContext()), MathZ);
  auto cos_ConstArray = llvm::ConstantArray::get(
      cos_ArrayType, llvm::ArrayRef<llvm::Constant *>(cos_vect.value));
  auto alignement_cos =
      newfs->getParent()->getDataLayout().getPrefTypeAlignment(cos_vect.value.front()->getType());
  auto cos_arry_g =
      TaffoMath::createGlobalConst(newfs->getParent(), "cos_global." + std::to_string(fxparg.scalarFracBitsAmt()) + "_" + std::to_string(fxparg.scalarBitsAmt()), cos_ArrayType,
                                   cos_ConstArray, alignement_cos);
  return cos_arry_g;
}


bool createSinCos(FloatToFixed * ref,
    llvm::Function *newfs, llvm::Function *oldf)
{
  //
  newfs->deleteBody();
  Value *generic;
  Module *M = oldf->getParent();
  bool isSin =
      oldf->getName().find("sin") == 0 || oldf->getName().find("_ZSt3sin") == 0;
  LLVM_DEBUG(dbgs() << "is sin: " << isSin << "\n");
  // retrive context used in later instruction
  llvm::LLVMContext &cont(oldf->getContext());
  DataLayout dataLayout(M);
  LLVM_DEBUG(dbgs() << "\nGet Context " << &cont << "\n");
  // get first basick block of function
  BasicBlock::Create(cont, "Entry", newfs);
  BasicBlock *where = &(newfs->getEntryBlock());
  LLVM_DEBUG(dbgs() << "\nGet entry point " << where);
  IRBuilder<> builder(where, where->getFirstInsertionPt());
  // get return type fixed point
  flttofix::FixedPointType fxpret;
  flttofix::FixedPointType fxparg;
  bool foundRet = false;
  bool foundArg = false;
  TaffoMath::getFixedFromRet(ref, oldf, fxpret, foundRet);
  // get argument fixed point
  TaffoMath::getFixedFromArg(ref, oldf, fxparg, 0, foundArg);
  if (!foundRet || !foundArg) {
    return partialSpecialCall(newfs, foundRet, fxpret);
  }
  TaffoMath::pair_ftp_value<llvm::Value *> arg(fxparg);
  arg.value = newfs->arg_begin();
  auto truefxpret = fxpret;


  LLVM_DEBUG(dbgs() << "fxpret: " << fxpret.scalarBitsAmt() << " frac part: " << fxpret.scalarFracBitsAmt() << " difference: " << fxpret.scalarBitsAmt() - fxpret.scalarFracBitsAmt() << "\n");

  auto int_type = fxpret.scalarToLLVMType(cont);
  auto internal_fxpt = flttofix::FixedPointType(true, fxpret.scalarBitsAmt() - 2, fxpret.scalarBitsAmt());
  // create local variable
  TaffoMath::pair_ftp_value<llvm::Value *> x_value(internal_fxpt);
  TaffoMath::pair_ftp_value<llvm::Value *> y_value(internal_fxpt);
  x_value.value = builder.CreateAlloca(int_type, nullptr, "x");
  y_value.value = builder.CreateAlloca(int_type, nullptr, "y");
  auto changeSign =
      builder.CreateAlloca(Type::getInt8Ty(cont), nullptr, "changeSign");
  auto changedFunction =
      builder.CreateAlloca(Type::getInt8Ty(cont), nullptr, "changefunc");
  auto specialAngle =
      builder.CreateAlloca(Type::getInt8Ty(cont), nullptr, "specialAngle");
  Value *arg_value = builder.CreateAlloca(int_type, nullptr, "arg");
  builder.CreateStore(newfs->getArg(0), arg_value);
  Value *i_iterator = builder.CreateAlloca(int_type, nullptr, "iterator");

  // create pi variable
  TaffoMath::pair_ftp_value<llvm::Constant *> pi(fxpret);
  TaffoMath::pair_ftp_value<llvm::Constant *> pi_2(fxpret);
  TaffoMath::pair_ftp_value<llvm::Constant *> pi_32(fxpret);
  TaffoMath::pair_ftp_value<llvm::Constant *> pi_half(fxpret);
  TaffoMath::pair_ftp_value<llvm::Constant *> pi_half_internal(internal_fxpt);
  TaffoMath::pair_ftp_value<llvm::Constant *> kopp(internal_fxpt);
  TaffoMath::pair_ftp_value<llvm::Constant *> zero(fxpret);
  TaffoMath::pair_ftp_value<llvm::Constant *> zeroarg(fxparg);
  TaffoMath::pair_ftp_value<llvm::Constant *> zero_internal(internal_fxpt);
  TaffoMath::pair_ftp_value<llvm::Constant *> one(fxpret);
  TaffoMath::pair_ftp_value<llvm::Constant *> one_internal(internal_fxpt);
  TaffoMath::pair_ftp_value<llvm::Constant *> minus_one(fxpret);
  TaffoMath::pair_ftp_value<llvm::Constant *> minus_one_internal(internal_fxpt);
  bool pi_created = TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::pi, fxpret, pi.value, pi.fpt);
  bool pi_2_created = TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::pi_2, fxpret, pi_2.value, pi_2.fpt);
  bool pi_32_created = TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::pi_32, fxpret, pi_32.value, pi_32.fpt);
  bool pi_half_created = TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::pi_half, fxpret, pi_half.value,
      pi_half.fpt);
  bool done = TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::pi_half, internal_fxpt, pi_half_internal.value,
      pi_half_internal.fpt);

  bool kopp_created = TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::Kopp, internal_fxpt, kopp.value, kopp.fpt);
  TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::zero, fxpret, zero.value, zero.fpt);
  TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::zero, fxparg, zeroarg.value, zeroarg.fpt);
  TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::one, fxpret, one.value, one.fpt);
  TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::one, internal_fxpt, one_internal.value, one_internal.fpt);
  TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::zero, internal_fxpt, zero_internal.value, zero_internal.fpt);
  TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::minus_one, fxpret, minus_one.value,
      minus_one.fpt);
  TaffoMath::createFixedPointFromConst(
      cont, ref, TaffoMath::minus_one, internal_fxpt, minus_one_internal.value,
      minus_one_internal.fpt);
  std::string S_ret_point = "." + std::to_string(fxpret.scalarFracBitsAmt());


  if (pi_created)
    pi.value = TaffoMath::createGlobalConst(
        M, "pi" + S_ret_point, pi.fpt.scalarToLLVMType(cont), pi.value,
        dataLayout.getPrefTypeAlignment(pi.fpt.scalarToLLVMType(cont)));
  if (pi_2_created)
    pi_2.value = TaffoMath::createGlobalConst(
        M, "pi_2" + S_ret_point, pi_2.fpt.scalarToLLVMType(cont), pi_2.value,
        dataLayout.getPrefTypeAlignment(pi_2.fpt.scalarToLLVMType(cont)));
  if (pi_32_created)
    pi_32.value = TaffoMath::createGlobalConst(
        M, "pi_32" + S_ret_point,
        pi_32.fpt.scalarToLLVMType(cont), pi_32.value,
        dataLayout.getPrefTypeAlignment(pi_32.fpt.scalarToLLVMType(cont)));
  if (pi_half_created)
    pi_half.value = TaffoMath::createGlobalConst(
        M, "pi_half" + S_ret_point,
        pi_half.fpt.scalarToLLVMType(cont), pi_half.value,
        dataLayout.getPrefTypeAlignment(pi_half.fpt.scalarToLLVMType(cont)));
  pi_half_internal.value = TaffoMath::createGlobalConst(
      M,
      "pi_half_internal_" + std::to_string(internal_fxpt.scalarFracBitsAmt()),
      pi_half_internal.fpt.scalarToLLVMType(cont), pi_half_internal.value,
      dataLayout.getPrefTypeAlignment(
          pi_half_internal.fpt.scalarToLLVMType(cont)));
  if (kopp_created)
    kopp.value = TaffoMath::createGlobalConst(
        M, "kopp" + S_ret_point, kopp.fpt.scalarToLLVMType(cont), kopp.value,
        dataLayout.getPrefTypeAlignment(kopp.fpt.scalarToLLVMType(cont)));
  zero.value = TaffoMath::createGlobalConst(
      M, "zero" + S_ret_point, zero.fpt.scalarToLLVMType(cont), zero.value,
      dataLayout.getPrefTypeAlignment(zero.fpt.scalarToLLVMType(cont)));
  zeroarg.value = TaffoMath::createGlobalConst(
      M, "zero_arg" + S_ret_point, zeroarg.fpt.scalarToLLVMType(cont),
      zeroarg.value,
      dataLayout.getPrefTypeAlignment(zeroarg.fpt.scalarToLLVMType(cont)));
  one.value = TaffoMath::createGlobalConst(
      M, "one" + S_ret_point, one.fpt.scalarToLLVMType(cont), one.value,
      dataLayout.getPrefTypeAlignment(one.fpt.scalarToLLVMType(cont)));
  minus_one.value = TaffoMath::createGlobalConst(
      M, "minus_one" + S_ret_point, minus_one.fpt.scalarToLLVMType(cont),
      minus_one.value,
      dataLayout.getPrefTypeAlignment(minus_one.fpt.scalarToLLVMType(cont)));
  minus_one_internal.value = TaffoMath::createGlobalConst(
      M, "minus_one_internal." + std::to_string(internal_fxpt.scalarFracBitsAmt()), minus_one_internal.fpt.scalarToLLVMType(cont),
      minus_one_internal.value,
      dataLayout.getPrefTypeAlignment(minus_one.fpt.scalarToLLVMType(cont)));
  one_internal.value = TaffoMath::createGlobalConst(
      M, "one_internal." + std::to_string(internal_fxpt.scalarFracBitsAmt()), one_internal.fpt.scalarToLLVMType(cont),
      one_internal.value,
      dataLayout.getPrefTypeAlignment(minus_one.fpt.scalarToLLVMType(cont)));

  /** create arctan table
   **/
  LLVM_DEBUG(dbgs() << "Create arctan table"
                    << "\n");
  TaffoMath::pair_ftp_value<llvm::Constant *,
                            TaffoMath::TABLELENGHT>
      arctan_2power;
  llvm::AllocaInst *pointer_to_array = nullptr;
  if (!MathZFlag) {
    for (int i = 0; i < TaffoMath::TABLELENGHT; i++) {
      arctan_2power.fpt.push_back(flttofix::FixedPointType(fxpret));
      Constant *tmp = nullptr;
      auto &current_fpt = arctan_2power.fpt.front();
      TaffoMath::createFixedPointFromConst(
          cont, ref, TaffoMath::arctan_2power[i], internal_fxpt, tmp, current_fpt);
      arctan_2power.value.push_back(tmp);
      LLVM_DEBUG(dbgs() << i << ")");
    }


    auto arctanArrayType = llvm::ArrayType::get(arctan_2power.value[0]->getType(),
                                                TaffoMath::TABLELENGHT);

    LLVM_DEBUG(dbgs() << "ArrayType  " << arctanArrayType << "\n");
    auto arctanConstArray = llvm::ConstantArray::get(
        arctanArrayType, llvm::ArrayRef<llvm::Constant *>(arctan_2power.value));
    LLVM_DEBUG(dbgs() << "ConstantDataArray tmp2 " << arctanConstArray << "\n");
    auto alignement_arctan =
        dataLayout.getPrefTypeAlignment(arctan_2power.value[0]->getType());
    auto arctan_g =
        TaffoMath::createGlobalConst(M, "arctan_g." + std::to_string(internal_fxpt.scalarFracBitsAmt()), arctanArrayType,
                                     arctanConstArray, alignement_arctan);

    pointer_to_array = builder.CreateAlloca(arctanArrayType);
    pointer_to_array->setAlignment(llvm::Align(alignement_arctan));

    builder.CreateMemCpy(
        pointer_to_array, llvm::Align(alignement_arctan), arctan_g, llvm::Align(alignement_arctan),
        TaffoMath::TABLELENGHT * (int_type->getScalarSizeInBits() >> 3));
    LLVM_DEBUG(dbgs() << "\nAdd to newf arctan table"
                      << "\n");
  }

  // code gen
  auto int_8_zero = ConstantInt::get(Type::getInt8Ty(cont), 0);
  auto int_8_one = ConstantInt::get(Type::getInt8Ty(cont), 1);
  auto int_8_minus_one = ConstantInt::get(Type::getInt8Ty(cont), -1);
  builder.CreateStore(int_8_zero, changeSign);
  builder.CreateStore(int_8_zero, changedFunction);
  builder.CreateStore(int_8_zero, specialAngle);
  BasicBlock *body = BasicBlock::Create(cont, "body", newfs);
  builder.CreateBr(body);
  builder.SetInsertPoint(body);
  arg.value = arg_value;
  BasicBlock *return_point = BasicBlock::Create(cont, "return_point", newfs);
  // handle unsigned arg
  if (!fxparg.scalarIsSigned()) {
    builder.CreateStore(builder.CreateLShr(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), ConstantInt::get(int_type, 1)), arg_value);
    fxparg.scalarFracBitsAmt() = fxparg.scalarFracBitsAmt() - 1;
    fxparg.scalarIsSigned() = true;
  }


  // handle negative
  if (isSin) {
    // sin(-x) == -sin(x)
    {
      BasicBlock *true_greater_zero =
          BasicBlock::Create(cont, "true_greater_zero", newfs);
      BasicBlock *false_greater_zero = BasicBlock::Create(cont, "body", newfs);

      generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value, "arg"),
                                      builder.CreateLoad(getElementTypeFromValuePointer(zeroarg.value), zeroarg.value));
      generic =
          builder.CreateCondBr(generic, true_greater_zero, false_greater_zero);
      builder.SetInsertPoint(true_greater_zero);
      generic = builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(zeroarg.value), zeroarg.value),
                                  builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value));
      builder.CreateStore(
          builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changeSign), changeSign), int_8_minus_one),
          changeSign);
      builder.CreateStore(generic, arg_value);
      builder.CreateBr(false_greater_zero);
      builder.SetInsertPoint(false_greater_zero);
    }

  } else {
    // cos(-x) == cos(x)
    {
      BasicBlock *true_greater_zero =
          BasicBlock::Create(cont, "true_greater_zero", newfs);
      BasicBlock *false_greater_zero = BasicBlock::Create(cont, "body", newfs);

      generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value, "arg"),
                                      builder.CreateLoad(getElementTypeFromValuePointer(zeroarg.value), zeroarg.value));
      generic =
          builder.CreateCondBr(generic, true_greater_zero, false_greater_zero);
      builder.SetInsertPoint(true_greater_zero);
      generic = builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(zeroarg.value), zeroarg.value),
                                  builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value));
      builder.CreateStore(generic, arg_value);
      builder.CreateBr(false_greater_zero);
      builder.SetInsertPoint(false_greater_zero);
    }
  }

  fixrangeSinCos(ref, newfs, fxparg, fxpret, arg_value, builder);

  // special case
  {
    // x = pi/2

    if (pi_half_created) {
      BasicBlock *BTrue = BasicBlock::Create(cont, "equal_pi_2", newfs);
      BasicBlock *BFalse = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value, "arg"),
                                     builder.CreateLoad(getElementTypeFromValuePointer(pi_half.value), pi_half.value));
      builder.CreateCondBr(generic, BTrue, BFalse);
      builder.SetInsertPoint(BTrue);
      builder.CreateStore(int_8_one, specialAngle);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(one_internal.value), one_internal.value), y_value.value);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value), x_value.value);
      builder.CreateBr(BFalse);
      builder.SetInsertPoint(BFalse);
    }
    // x= pi
    if (pi_created) {
      BasicBlock *BTrue = BasicBlock::Create(cont, "equal_pi", newfs);
      BasicBlock *BFalse = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value, "arg"),
                                     builder.CreateLoad(getElementTypeFromValuePointer(pi.value), pi.value));
      builder.CreateCondBr(generic, BTrue, BFalse);
      builder.SetInsertPoint(BTrue);
      builder.CreateStore(int_8_one, specialAngle);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value), y_value.value);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(minus_one_internal.value), minus_one_internal.value), x_value.value);
      builder.CreateBr(BFalse);
      builder.SetInsertPoint(BFalse);
    }
    // x = pi_32
    if (pi_32_created) {
      BasicBlock *BTrue = BasicBlock::Create(cont, "equal_pi_32", newfs);
      BasicBlock *BFalse = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value, "arg"),
                                     builder.CreateLoad(getElementTypeFromValuePointer(pi_32.value), pi_32.value));
      builder.CreateCondBr(generic, BTrue, BFalse);
      builder.SetInsertPoint(BTrue);
      builder.CreateStore(int_8_one, specialAngle);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(minus_one_internal.value), minus_one_internal.value), y_value.value);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value), x_value.value);
      builder.CreateBr(BFalse);
      builder.SetInsertPoint(BFalse);
    }
    // x = 0

    {
      BasicBlock *BTrue = BasicBlock::Create(cont, "equal_0", newfs);
      BasicBlock *BFalse = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value, "arg"),
                                     builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value));
      builder.CreateCondBr(generic, BTrue, BFalse);
      builder.SetInsertPoint(BTrue);
      builder.CreateStore(int_8_one, specialAngle);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value), y_value.value);
      builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(one_internal.value), one_internal.value), x_value.value);
      builder.CreateBr(BFalse);
      builder.SetInsertPoint(BFalse);
    }
    // handle premature return
    {
      BasicBlock *BTrue = BasicBlock::Create(cont, "Special", newfs);
      BasicBlock *BFalse = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(specialAngle), specialAngle, "arg"),
                                     int_8_one);
      builder.CreateCondBr(generic, BTrue, BFalse);
      builder.SetInsertPoint(BTrue);
      builder.CreateBr(return_point);
      builder.SetInsertPoint(BFalse);
    }
  }

  if (isSin) {
    // angle > pi_half && angle < pi sin(x) = cos(x - pi_half)
    if (pi_half_created && pi_created) {
      BasicBlock *in_II_quad = BasicBlock::Create(cont, "in_II_quad", newfs);
      BasicBlock *not_in_II_quad = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(pi_half.value), pi_half.value, "pi_half"),
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), "arg_greater_pi_half");
      Value *generic2 = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), builder.CreateLoad(getElementTypeFromValuePointer(pi.value), pi.value, "pi"),
          "arg_less_pi");
      generic = builder.CreateAnd(generic, generic2);
      builder.CreateCondBr(generic, in_II_quad, not_in_II_quad);
      builder.SetInsertPoint(in_II_quad);
      builder.CreateStore(builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changedFunction), changedFunction),
                                            int_8_minus_one),
                          changedFunction);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            builder.CreateLoad(getElementTypeFromValuePointer(pi_half.value), pi_half.value)),
                          arg_value);
      builder.CreateBr(not_in_II_quad);
      builder.SetInsertPoint(not_in_II_quad);
    }
    // angle > pi&& angle < pi_32(x) sin(x) = -sin(x - pi)
    if (pi_32_created && pi_created) {
      BasicBlock *in_III_quad = BasicBlock::Create(cont, "in_III_quad", newfs);
      BasicBlock *not_in_III_quad = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(pi.value), pi.value, "pi"),
                                      builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                      "arg_greater_pi");
      Value *generic2 = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
          builder.CreateLoad(getElementTypeFromValuePointer(pi_32.value), pi_32.value, "pi_32"), "arg_less_pi_32");
      generic = builder.CreateAnd(generic, generic2);
      builder.CreateCondBr(generic, in_III_quad, not_in_III_quad);
      builder.SetInsertPoint(in_III_quad);
      builder.CreateStore(
          builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changeSign), changeSign), int_8_minus_one),
          changeSign);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            builder.CreateLoad(getElementTypeFromValuePointer(pi.value), pi.value)),
                          arg_value);
      builder.CreateBr(not_in_III_quad);
      builder.SetInsertPoint(not_in_III_quad);
    }
    // angle > pi_32&& angle < pi_2(x) sin(x) = -cos(x - pi_32);
    if (pi_32_created && pi_2_created) {
      BasicBlock *in_IV_quad = BasicBlock::Create(cont, "in_IV_quad", newfs);
      BasicBlock *not_in_IV_quad = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(pi_32.value), pi_32.value, "pi_32"),
                                      builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                      "arg_greater_pi_32");
      Value *generic2 = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), builder.CreateLoad(getElementTypeFromValuePointer(pi_2.value), pi_2.value, "pi_2"),
          "arg_less_2pi");
      generic = builder.CreateAnd(generic, generic2);
      builder.CreateCondBr(generic, in_IV_quad, not_in_IV_quad);
      builder.SetInsertPoint(in_IV_quad);
      builder.CreateStore(
          builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changeSign), changeSign), int_8_minus_one),
          changeSign);
      builder.CreateStore(builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changedFunction), changedFunction),
                                            int_8_minus_one),
                          changedFunction);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            builder.CreateLoad(getElementTypeFromValuePointer(pi_32.value), pi_32.value)),
                          arg_value);
      builder.CreateBr(not_in_IV_quad);
      builder.SetInsertPoint(not_in_IV_quad);
    }
  } else {

    // angle > pi_half && angle < pi cos(x) = -sin(x - pi_half);
    if (pi_half_created && pi_created) {
      BasicBlock *in_II_quad = BasicBlock::Create(cont, "in_II_quad", newfs);
      BasicBlock *not_in_II_quad = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(pi_half.value), pi_half.value, "pi_half"),
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), "arg_greater_pi_half");
      Value *generic2 = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), builder.CreateLoad(getElementTypeFromValuePointer(pi.value), pi.value, "pi"),
          "arg_less_pi");
      generic = builder.CreateAnd(generic, generic2);
      builder.CreateCondBr(generic, in_II_quad, not_in_II_quad);
      builder.SetInsertPoint(in_II_quad);
      builder.CreateStore(
          builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changeSign), changeSign), int_8_minus_one),
          changeSign);
      builder.CreateStore(builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changedFunction), changedFunction),
                                            int_8_minus_one),
                          changedFunction);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            builder.CreateLoad(getElementTypeFromValuePointer(pi_half.value), pi_half.value)),
                          arg_value);
      builder.CreateBr(not_in_II_quad);
      builder.SetInsertPoint(not_in_II_quad);
    }
    // angle > pi&& angle < pi_32(x) cos(x) = -cos(x-pi)
    if (pi_32_created && pi_created) {
      BasicBlock *in_III_quad = BasicBlock::Create(cont, "in_III_quad", newfs);
      BasicBlock *not_in_III_quad = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(pi.value), pi.value, "pi"),
                                      builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                      "arg_greater_pi");
      Value *generic2 = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
          builder.CreateLoad(getElementTypeFromValuePointer(pi_32.value), pi_32.value, "pi_32"), "arg_less_pi_32");
      generic = builder.CreateAnd(generic, generic2);
      builder.CreateCondBr(generic, in_III_quad, not_in_III_quad);
      builder.SetInsertPoint(in_III_quad);
      builder.CreateStore(
          builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changeSign), changeSign), int_8_minus_one),
          changeSign);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            builder.CreateLoad(getElementTypeFromValuePointer(pi.value), pi.value)),
                          arg_value);
      builder.CreateBr(not_in_III_quad);
      builder.SetInsertPoint(not_in_III_quad);
    }
    // angle > pi_32&& angle < pi_2(x) cos(x) = sin(angle - pi_32);
    if (pi_32_created && pi_2_created) {
      BasicBlock *in_IV_quad = BasicBlock::Create(cont, "in_IV_quad", newfs);
      BasicBlock *not_in_IV_quad = BasicBlock::Create(cont, "body", newfs);
      generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(pi_32.value), pi_32.value, "pi_32"),
                                      builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                      "arg_greater_pi_32");
      Value *generic2 = builder.CreateICmpSLT(
          builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), builder.CreateLoad(getElementTypeFromValuePointer(pi_2.value), pi_2.value, "pi_2"),
          "arg_less_2pi");
      generic = builder.CreateAnd(generic, generic2);
      builder.CreateCondBr(generic, in_IV_quad, not_in_IV_quad);
      builder.SetInsertPoint(in_IV_quad);
      builder.CreateStore(builder.CreateXor(builder.CreateLoad(getElementTypeFromValuePointer(changedFunction), changedFunction),
                                            int_8_minus_one),
                          changedFunction);
      builder.CreateStore(builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                            builder.CreateLoad(getElementTypeFromValuePointer(pi_32.value), pi_32.value)),
                          arg_value);
      builder.CreateBr(not_in_IV_quad);
      builder.SetInsertPoint(not_in_IV_quad);
    }
  }


  // calculate sin and cos
  if (!MathZFlag) {
    builder.CreateStore(builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), internal_fxpt.scalarFracBitsAmt() - fxpret.scalarFracBitsAmt()), arg_value);
    auto zero_arg = builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value);
    // x=kopp
    builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(kopp.value), kopp.value), x_value.value);
    // y=0
    builder.CreateStore(zero_arg, y_value.value);

    BasicBlock *epilog_loop = BasicBlock::Create(cont, "epilog_loop", newfs);
    BasicBlock *start_loop = BasicBlock::Create(cont, "start_loop", newfs);

    // i=0
    builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value), i_iterator);
    builder.CreateBr(epilog_loop);
    builder.SetInsertPoint(epilog_loop);
    // i < TABLELENGHT
    generic = builder.CreateICmpSLT(
        builder.CreateLoad(getElementTypeFromValuePointer(i_iterator), i_iterator),
        ConstantInt::get(int_type, TaffoMath::TABLELENGHT));
    // i < size of int
    Value *generic2 = builder.CreateICmpSLT(
        builder.CreateLoad(getElementTypeFromValuePointer(i_iterator), i_iterator),
        ConstantInt::get(int_type, int_type->getScalarSizeInBits()));
    builder.CreateCondBr(builder.CreateAnd(generic, generic2), start_loop,
                         return_point);
    builder.SetInsertPoint(start_loop);
    // dn = arg >= 0 ? 1 : -1;
    Value *dn = builder.CreateSelect(
        builder.CreateICmpSGE(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), zero_arg),
        ConstantInt::get(int_type, 1), ConstantInt::get(int_type, -1));

    // xt = x >> i
    Value *xt = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(x_value.value), x_value.value),
                                   builder.CreateLoad(getElementTypeFromValuePointer(i_iterator), i_iterator));

    // yt = x >> i
    Value *yt = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(y_value.value), y_value.value),
                                   builder.CreateLoad(getElementTypeFromValuePointer(i_iterator), i_iterator));
    // arctan_2power[i]
    generic = builder.CreateGEP(getElementTypeFromValuePointer(pointer_to_array), pointer_to_array,
                                {zero_arg, builder.CreateLoad(getElementTypeFromValuePointer(i_iterator), i_iterator)});

    generic = builder.CreateLoad(getElementTypeFromValuePointer(generic), generic);

    // dn > 0
    auto dn_greate_zero = builder.CreateICmpSGT(dn, zero_arg);

    // arg = arg + (dn > 0 ? -arctan_2power[i] : arctan_2power[i]);
    generic = builder.CreateSelect(
        dn_greate_zero, builder.CreateSub(zero_arg, generic), generic);

    builder.CreateStore(
        builder.CreateAdd(generic, builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value)), arg_value);

    // x = x + (dn > 0 ? -yt : yt);
    generic = builder.CreateSelect(dn_greate_zero,
                                   builder.CreateSub(zero_arg, yt), yt);

    builder.CreateStore(
        builder.CreateAdd(generic, builder.CreateLoad(getElementTypeFromValuePointer(x_value.value), x_value.value)),
        x_value.value);
    ;
    // y = y + (dn > 0 ? xt : -xt);
    generic = builder.CreateSelect(dn_greate_zero, xt,
                                   builder.CreateSub(zero_arg, xt));
    builder.CreateStore(
        builder.CreateAdd(generic, builder.CreateLoad(getElementTypeFromValuePointer(y_value.value), y_value.value)),
        y_value.value);
    ;
    // i++
    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(i_iterator), i_iterator),
                                          ConstantInt::get(int_type, 1)),
                        i_iterator);
    builder.CreateBr(epilog_loop);
    builder.SetInsertPoint(return_point);
  } else {
    builder.CreateStore(builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
                                          internal_fxpt.scalarFracBitsAmt() -
                                              fxpret.scalarFracBitsAmt()),
                        arg_value);
    Value *sin_g = generateSinLUT(ref, newfs, internal_fxpt, builder);
    // Value *cos_g = generateCosLUT(this, oldf, internal_fxpt, builder);
    auto zero_arg = builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value);

    Value *tmp_angle = builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value);


    std::string function_name("llvm.udiv.fix.i");
    function_name.append(std::to_string(internal_fxpt.scalarToLLVMType(cont)->getScalarSizeInBits()));


    Function *udiv = M->getFunction(function_name);
    if (udiv == nullptr) {
      std::vector<llvm::Type *> fun_arguments;
      fun_arguments.push_back(
          internal_fxpt.scalarToLLVMType(cont)); // depends on your type
      fun_arguments.push_back(
          internal_fxpt.scalarToLLVMType(cont)); // depends on your type
      fun_arguments.push_back(Type::getInt32Ty(cont));
      FunctionType *fun_type = FunctionType::get(
          internal_fxpt.scalarToLLVMType(cont), fun_arguments, false);
      udiv = llvm::Function::Create(fun_type, GlobalValue::ExternalLinkage,
                                    function_name, M);
    }


    generic = builder.CreateCall(
        udiv, {tmp_angle, builder.CreateLShr(builder.CreateLoad(getElementTypeFromValuePointer(pi_half_internal.value), pi_half_internal.value), int(log2(MathZ))),
               llvm::ConstantInt::get(internal_fxpt.scalarToLLVMType(cont),
                                      internal_fxpt.scalarFracBitsAmt() -
                                          int(log2(MathZ)))});
    generic = builder.CreateLShr(
        generic, llvm::ConstantInt::get(internal_fxpt.scalarToLLVMType(cont),
                                        internal_fxpt.scalarFracBitsAmt() -
                                            int(log2(MathZ))));

    auto tmp = builder.CreateGEP(getElementTypeFromValuePointer(sin_g), sin_g, {zero_arg, generic});

    builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(tmp), tmp),
                        y_value.value);
    generic = builder.CreateSub(llvm::ConstantInt::get(internal_fxpt.scalarToLLVMType(cont), MathZ), generic);
    tmp = builder.CreateGEP(getElementTypeFromValuePointer(sin_g), sin_g, {zero_arg, generic});
    builder.CreateStore(builder.CreateLoad(getElementTypeFromValuePointer(tmp), tmp),
                        x_value.value);
    builder.CreateBr(return_point);
    builder.SetInsertPoint(return_point);
  }

  {
    auto zero_arg = builder.CreateLoad(getElementTypeFromValuePointer(zero.value), zero.value);
    auto zero_bool = int_8_zero;
    if (isSin) {
      generic = builder.CreateSelect(
          builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(changedFunction), changedFunction), zero_bool),
          builder.CreateLoad(getElementTypeFromValuePointer(y_value.value), y_value.value), builder.CreateLoad(getElementTypeFromValuePointer(x_value.value), x_value.value));
      builder.CreateStore(generic, arg_value);
    } else {
      generic = builder.CreateSelect(
          builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(changedFunction), changedFunction), zero_bool),
          builder.CreateLoad(getElementTypeFromValuePointer(x_value.value), x_value.value), builder.CreateLoad(getElementTypeFromValuePointer(y_value.value), y_value.value));
      builder.CreateStore(generic, arg_value);
    }

    generic = builder.CreateSelect(
        builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(changeSign), changeSign), zero_bool),
        builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value),
        builder.CreateSub(zero_arg, builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value)));
    builder.CreateStore(generic, arg_value);
  }
  if (internal_fxpt.scalarFracBitsAmt() > truefxpret.scalarFracBitsAmt()) {
    builder.CreateStore(builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), internal_fxpt.scalarFracBitsAmt() - truefxpret.scalarFracBitsAmt()), arg_value);
  } else if (internal_fxpt.scalarFracBitsAmt() < truefxpret.scalarFracBitsAmt()) {
    builder.CreateStore(builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value), truefxpret.scalarFracBitsAmt() - internal_fxpt.scalarFracBitsAmt()), arg_value);
  }

  auto ret = builder.CreateLoad(getElementTypeFromValuePointer(arg_value), arg_value);


  BasicBlock *end = BasicBlock::Create(cont, "end", newfs);
  builder.CreateBr(end);
  builder.SetInsertPoint(end);
  builder.CreateRet(ret);
  return true;
}


} // end namespace flttofix