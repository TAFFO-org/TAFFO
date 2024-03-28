#include "CreateSpecialFunction.h"
#include "DebugPrint.h"
#include "TAFFOMath.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <llvm/ADT/APInt.h>
#include <llvm/IR/Type.h>
#include <string>

namespace taffo
{

namespace intrinsic
{
enum intrinsicNames {
#define GET_INTRINSIC_NAME_TABLE
#include "llvm/IR/IntrinsicEnums.inc"
#undef GET_INTRINSIC_NAME_TABLE
};
} // namespace intrinsic


} // namespace taffo


void fixrangeSinCos(flttofix::FloatToFixed *ref, llvm::Function *new_f, flttofix::FixedPointType &fxparg,
                    flttofix::FixedPointType &fxpret, Value *arg_value,
                    llvm::IRBuilder<> &builder)
{
  auto &m = *new_f->getParent();
  if (!fxparg.isFloatingPoint()) {
    assert(fxparg.scalarBitsAmt() == fxpret.scalarBitsAmt() &&
           "different type arg and ret");
    int int_lenght = fxparg.scalarBitsAmt();
    llvm::LLVMContext &cont = new_f->getContext();
    DataLayout dataLayout(new_f->getParent());
    auto int_type = fxparg.scalarToLLVMType(cont);
    Value *generic = nullptr;
    int max = fxparg.scalarFracBitsAmt() > fxpret.scalarFracBitsAmt()
                  ? fxparg.scalarFracBitsAmt()
                  : fxpret.scalarFracBitsAmt();
    int min = fxparg.scalarFracBitsAmt() < fxpret.scalarFracBitsAmt()
                  ? fxparg.scalarFracBitsAmt()
                  : fxpret.scalarFracBitsAmt();

    max = max >= fxpret.scalarBitsAmt() - 3 ? fxpret.scalarBitsAmt() - 3 : max;
    bool can_continue = true;
    if (min > fxpret.scalarBitsAmt() - 3) {
      min = fxpret.scalarBitsAmt() - 3;
      can_continue = false;
    }

    if (fxparg.scalarFracBitsAmt() > fxparg.scalarBitsAmt() - 3) {
      wrapper_printf(builder, m, "Not to normalize\n");
      can_continue = false;
    }


    LLVM_DEBUG(dbgs() << "Max: " << max << " Min: " << min << "\n");
    // create pi_2 table
    std::vector<flttofix::FixedPointType> pi_2_fxp;
    std::vector<llvm::Constant *> pi_2_const;

    int created = 0;
    for (int i = 0; i <= max - min && can_continue; ++i) {
      flttofix::FixedPointType match = flttofix::FixedPointType(fxparg.scalarIsSigned(), min + i, int_lenght);
      Constant *tmp = TaffoMath::createFixedPointFromConst(
          cont, ref, TaffoMath::pi_2, match);
      if (tmp != nullptr) {
        pi_2_fxp.emplace_back(match);
        pi_2_const.emplace_back(tmp);
        created += 1;
      }
    }
    if (created != 0) {
      auto pi_2_ArrayType =
          llvm::ArrayType::get(int_type, max - min + 1);
      LLVM_DEBUG(dbgs() << "ArrayType  " << pi_2_ArrayType << "\n");
      auto pi_2_ConstArray = llvm::ConstantArray::get(
          pi_2_ArrayType, llvm::ArrayRef<llvm::Constant *>(pi_2_const));
      LLVM_DEBUG(dbgs() << "ConstantDataArray pi_2 " << pi_2_ConstArray << "\n");
      auto alignement_pi_2 =
          dataLayout.getPrefTypeAlign(int_type);
      auto pi_2_arry_g =
          TaffoMath::createGlobalConst(new_f->getParent(), "pi_2_global." + std::to_string(min) + "_" + std::to_string(max), pi_2_ArrayType,
                                       pi_2_ConstArray, alignement_pi_2);
      auto pointer_to_array = TaffoMath::addAllocaToStart(ref, new_f, builder, pi_2_ArrayType, nullptr, "pi_2_array");
      dyn_cast<llvm::AllocaInst>(pointer_to_array)->setAlignment(alignement_pi_2);
      builder.CreateMemCpy(
          pointer_to_array,
          alignement_pi_2,
          pi_2_arry_g,
          alignement_pi_2,
          uint64_t{(max - min + 1) * (int_type->getScalarSizeInBits() >> 3)});
      LLVM_DEBUG(dbgs() << "\nAdd pi_2 table"
                        << "\n");

      int current_arg_point = fxparg.scalarFracBitsAmt();
      int current_ret_point = fxpret.scalarFracBitsAmt();
      wrapper_printf(builder, m, "Start Normalization\n");

      Value *iterator_pi_2 =
          TaffoMath::addAllocaToStart(ref, new_f, builder, int_type, nullptr, "Iterator_pi_2");
      Value *point_arg =
          TaffoMath::addAllocaToStart(ref, new_f, builder, int_type, nullptr, "point_arg");
      Value *point_ret =
          TaffoMath::addAllocaToStart(ref, new_f, builder, int_type, nullptr, "point_ret");

      builder.CreateStore(ConstantInt::get(int_type, current_arg_point), point_arg);
      builder.CreateStore(ConstantInt::get(int_type, current_ret_point), point_ret);

      if (current_arg_point < current_ret_point) {
        BasicBlock *normalize_cond =
            BasicBlock::Create(cont, "normalize_cond", new_f);
        BasicBlock *normalize = BasicBlock::Create(cont, "normalize", new_f);
        BasicBlock *cmp_bigger_than_2pi =
            BasicBlock::Create(cont, "cmp_bigger_than_2pi", new_f);
        BasicBlock *bigger_than_2pi =
            BasicBlock::Create(cont, "bigger_than_2pi", new_f);
        BasicBlock *end_loop = BasicBlock::Create(cont, "body", new_f);
        builder.CreateStore(ConstantInt::get(int_type, 0), iterator_pi_2);
        wrapper_printf(builder, m, "ArgValue: %i \n", builder.CreateLoad(arg_value));
        builder.CreateBr(normalize_cond);

        LLVM_DEBUG(dbgs() << "add Normalize\n");
        // normalize cond
        builder.SetInsertPoint(normalize_cond);

        wrapper_printf(builder, m, "iterator_pi_2: %i \n", builder.CreateLoad(iterator_pi_2));
        wrapper_printf(builder, m, "point_ret: %i \n", builder.CreateLoad(point_ret));
        wrapper_printf(builder, m, "point_arg: %i \n", builder.CreateLoad(point_arg));


        Value *last_bit_mask =
            fxparg.scalarIsSigned()
                ? ConstantInt::get(int_type, 1 << (int_lenght - 2))
                : ConstantInt::get(int_type, 1 << (int_lenght - 1));
        generic = builder.CreateAnd(builder.CreateLoad(arg_value), last_bit_mask);
        generic = builder.CreateAnd(
            builder.CreateICmpEQ(generic, ConstantInt::get(int_type, 0)),
            builder.CreateICmpSLT(builder.CreateLoad(point_arg),
                                  builder.CreateLoad(point_ret)));
        builder.CreateCondBr(generic, normalize, cmp_bigger_than_2pi);
        // normalize
        builder.SetInsertPoint(normalize);
        wrapper_printf(builder, m, "Normalize \n");
        builder.CreateStore(builder.CreateShl(builder.CreateLoad(arg_value),
                                              ConstantInt::get(int_type, 1)),
                            arg_value);
        builder.CreateStore(builder.CreateAdd(builder.CreateLoad(iterator_pi_2),
                                              ConstantInt::get(int_type, 1)),
                            iterator_pi_2);
        builder.CreateStore(builder.CreateAdd(builder.CreateLoad(point_arg),
                                              ConstantInt::get(int_type, 1)),
                            point_arg);
        builder.CreateBr(normalize_cond);
        LLVM_DEBUG(dbgs() << "add bigger than 2pi\n");
        // cmp_bigger_than_2pi
        builder.SetInsertPoint(cmp_bigger_than_2pi);
        generic = builder.CreateGEP(
            pointer_to_array,
            {ConstantInt::get(int_type, 0), builder.CreateLoad(iterator_pi_2)});
        generic = builder.CreateLoad(generic);
        wrapper_printf(builder, m, "%i < %i \n", generic, builder.CreateLoad(arg_value));
        builder.CreateCondBr(builder.CreateICmpULE(generic,
                                                   builder.CreateLoad(arg_value)),
                             bigger_than_2pi, end_loop);
        // bigger_than_2pi
        builder.SetInsertPoint(bigger_than_2pi);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              generic),
                            arg_value);
        wrapper_printf(builder, m, "ArgValue subtracted: %i \n", builder.CreateLoad(arg_value));

        builder.CreateBr(normalize_cond);
        builder.SetInsertPoint(end_loop);
      } else {
        LLVM_DEBUG(dbgs() << "add bigger than 2pi\n");
        BasicBlock *cmp_bigger_than_2pi =
            BasicBlock::Create(cont, "cmp_bigger_than_2pi", new_f);
        BasicBlock *bigger_than_2pi =
            BasicBlock::Create(cont, "bigger_than_2pi", new_f);
        BasicBlock *body = BasicBlock::Create(cont, "shift", new_f);
        builder.CreateStore(ConstantInt::get(int_type, 0), iterator_pi_2);
        builder.CreateBr(cmp_bigger_than_2pi);
        builder.SetInsertPoint(cmp_bigger_than_2pi);
        generic = builder.CreateGEP(
            pointer_to_array,
            {ConstantInt::get(int_type, 0), builder.CreateSub(ConstantInt::get(int_type, max - min + 1), builder.CreateLoad(iterator_pi_2))});
        generic = builder.CreateLoad(generic);
        wrapper_printf(builder, m, "PI selected: %i \n", generic);
        builder.CreateCondBr(builder.CreateICmpULE(generic,
                                                   builder.CreateLoad(arg_value)),
                             bigger_than_2pi, body);
        builder.SetInsertPoint(bigger_than_2pi);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              generic),
                            arg_value);
        builder.CreateBr(cmp_bigger_than_2pi);
        builder.SetInsertPoint(body);
      }
      // set point at same position
      {
        wrapper_printf(builder, m, "ArgValue end: %i \n", builder.CreateLoad(arg_value));
        LLVM_DEBUG(dbgs() << "set point at same position\n");
        builder.CreateStore(ConstantInt::get(int_type, 0), iterator_pi_2);
        BasicBlock *body_2 = BasicBlock::Create(cont, "body", new_f);
        BasicBlock *true_block = BasicBlock::Create(cont, "left_shift", new_f);
        BasicBlock *false_block = BasicBlock::Create(cont, "right_shift", new_f);

        BasicBlock *end = BasicBlock::Create(cont, "body", new_f);
        builder.CreateCondBr(builder.CreateICmpEQ(builder.CreateLoad(point_arg),
                                                  builder.CreateLoad(point_ret)),
                             end, body_2);
        builder.SetInsertPoint(body_2);
        builder.CreateCondBr(builder.CreateICmpSLT(builder.CreateLoad(point_arg),
                                                   builder.CreateLoad(point_ret)),
                             true_block, false_block);
        builder.SetInsertPoint(true_block);
        generic = builder.CreateSub(builder.CreateLoad(point_ret),
                                    builder.CreateLoad(point_arg));
        builder.CreateStore(
            builder.CreateShl(builder.CreateLoad(arg_value), generic), arg_value);
        builder.CreateBr(end);
        builder.SetInsertPoint(false_block);
        generic = builder.CreateSub(builder.CreateLoad(point_arg),
                                    builder.CreateLoad(point_ret));
        builder.CreateStore(
            builder.CreateAShr(builder.CreateLoad(arg_value), generic), arg_value);

        wrapper_printf(builder, m, "ArgValue shifted: %i \n", builder.CreateLoad(arg_value));
        builder.CreateBr(end);
        builder.SetInsertPoint(end);
        wrapper_printf(builder, m, "End Normalization\n");
      }
    } else {
      wrapper_printf(builder, m, "Pre out value %i\n", builder.CreateLoad(arg_value));
      if (fxparg.scalarFracBitsAmt() > fxpret.scalarFracBitsAmt()) {
        builder.CreateStore(
            builder.CreateLShr(builder.CreateLoad(arg_value), fxparg.scalarFracBitsAmt() - fxpret.scalarFracBitsAmt()), arg_value);
      }
      if (fxparg.scalarFracBitsAmt() < fxpret.scalarFracBitsAmt()) {
        builder.CreateStore(
            builder.CreateShl(builder.CreateLoad(arg_value), fxpret.scalarFracBitsAmt() - fxparg.scalarFracBitsAmt()), arg_value);
      }
      wrapper_printf(builder, m, "Output value %i\n", builder.CreateLoad(arg_value));
    }
  } else {
    builder.CreateStore(builder.CreateFRem(builder.CreateLoad(arg_value), llvm::ConstantFP::get(arg_value->getType(), TaffoMath::pi_2)), arg_value);
  }

  LLVM_DEBUG(dbgs() << "End fixrange\n");
}


Value *generateSinLUT(flttofix::FloatToFixed *ref, Function *new_f, flttofix::FixedPointType &fxparg,
                      llvm::IRBuilder<> &builder)
{

  if (!fxparg.isFloatingPoint()) {
    std::vector<llvm::Constant *> sin_arr_const;

    for (int i = 0; i < MathZ; ++i) {
      flttofix::FixedPointType match = flttofix::FixedPointType(false, fxparg.scalarFracBitsAmt(), fxparg.scalarBitsAmt());
      auto tmp = TaffoMath::createFixedPointFromConst(
          new_f->getContext(), ref, sin(i * TaffoMath::pi_half / MathZ), match);


      if (tmp == nullptr) {
        llvm_unreachable("Ma nooooo\n");
      }
      sin_arr_const.push_back(tmp);
    }
    auto sin_ArrayType =
        llvm::ArrayType::get(fxparg.scalarToLLVMType(new_f->getContext()), MathZ);
    auto sin_ConstArray = llvm::ConstantArray::get(
        sin_ArrayType, llvm::ArrayRef<llvm::Constant *>(sin_arr_const));
    auto alignement_sin =
        new_f->getParent()->getDataLayout().getPrefTypeAlign(sin_arr_const.front()->getType());
    auto sin_arry_g =
        TaffoMath::createGlobalConst(new_f->getParent(), "sin_global." + std::to_string(fxparg.scalarFracBitsAmt()) + "_" + std::to_string(fxparg.scalarBitsAmt()), sin_ArrayType,
                                     sin_ConstArray, alignement_sin);
    return sin_arry_g;
  } else {
    std::vector<llvm::Constant *> sin_arr_const;

    for (int i = 0; i < MathZ; ++i) {
      flttofix::FixedPointType match = flttofix::FixedPointType(false, fxparg.scalarFracBitsAmt(), fxparg.scalarBitsAmt());
      auto tmp = TaffoMath::createFixedPointFromConst(
          new_f->getContext(), ref, sin(i * TaffoMath::pi_half / MathZ), match);


      if (tmp == nullptr) {
        llvm_unreachable("Ma nooooo\n");
      }
      sin_arr_const.push_back(tmp);
    }
    auto sin_ArrayType =
        llvm::ArrayType::get(fxparg.scalarToLLVMType(new_f->getContext()), MathZ);
    auto sin_ConstArray = llvm::ConstantArray::get(
        sin_ArrayType, llvm::ArrayRef<llvm::Constant *>(sin_arr_const));
    auto alignement_sin =
        new_f->getParent()->getDataLayout().getPrefTypeAlign(sin_arr_const.front()->getType());
    auto sin_arry_g =
        TaffoMath::createGlobalConst(new_f->getParent(), std::string("sin_global.") + (sin_arr_const[0]->getType() == llvm::Type::getFloatTy(new_f->getContext()) ? "float" : "duble"), sin_ArrayType,
                                     sin_ConstArray, alignement_sin);
    return sin_arry_g;
  }
}


// Common function that create both sin and cos
bool createSinCos(flttofix::FloatToFixed *float_to_fixed,
                  llvm::Function *new_f, llvm::Function *old_f, const flttofix::FixedPointType *old_ret_fxpt,
                  const flttofix::FixedPointType *old_arg_fxpt)
{
  auto &m = *new_f->getParent();
  LLVM_DEBUG(llvm::dbgs() << "####" << __func__ << " ####");
  Value *generic;
  bool is_sin = taffo::start_with(taffo::HandledSpecialFunction::demangle((std::string)old_f->getName()), "sin") || taffo::start_with(taffo::HandledSpecialFunction::demangle((std::string)old_f->getName()), "__dev-sin");
  ;
  // retrive context used in later instruction
  llvm::LLVMContext &cntx(old_f->getContext());
  // retruve the data llayout
  DataLayout dataLayout(old_f->getParent());
  // Create new block
  BasicBlock::Create(cntx, "Entry", new_f);
  BasicBlock *where = &(new_f->getEntryBlock());
  // builder to new_f
  llvm::IRBuilder<> builder(where, where->getFirstInsertionPt());
  // get return type fixed point
  flttofix::FixedPointType fxpret = *old_ret_fxpt;
  flttofix::FixedPointType fxparg = *old_arg_fxpt;
  flttofix::FixedPointType truefxpret = *old_ret_fxpt;


  llvm::dbgs() << "\n"
               << *new_f;

  auto new_arg_type = new_f->getArg(0)->getType();


  // common part
  auto internal_fxpt = !fxparg.isFloatingPoint() ? flttofix::FixedPointType(false, fxparg.scalarBitsAmt() - 3, fxparg.scalarBitsAmt()) : fxparg;


  auto changeSign =
      builder.CreateAlloca(Type::getInt8Ty(cntx), nullptr, "changeSign");
  auto changedFunction =
      builder.CreateAlloca(Type::getInt8Ty(cntx), nullptr, "changefunc");
  auto specialAngle =
      builder.CreateAlloca(Type::getInt8Ty(cntx), nullptr, "specialAngle");

  Value *arg_value = builder.CreateAlloca(new_arg_type, nullptr, "arg");
  builder.CreateStore(new_f->getArg(0), arg_value);

  auto pi = TaffoMath::make_fxp(internal_fxpt, TaffoMath::pi, cntx, float_to_fixed);
  auto pi_2 = TaffoMath::make_fxp(internal_fxpt, TaffoMath::pi_2, cntx, float_to_fixed);
  auto pi_32 = TaffoMath::make_fxp(internal_fxpt, TaffoMath::pi_32, cntx, float_to_fixed);
  auto pi_half = TaffoMath::make_fxp(internal_fxpt, TaffoMath::pi_half, cntx, float_to_fixed);
  auto pi_half_internal = TaffoMath::make_fxp(internal_fxpt, TaffoMath::pi_half, cntx, float_to_fixed);
  auto zero = TaffoMath::make_fxp(internal_fxpt, TaffoMath::zero, cntx, float_to_fixed);
  auto zeroarg = TaffoMath::make_fxp(fxparg, TaffoMath::zero, cntx, float_to_fixed);
  auto zero_internal = TaffoMath::make_fxp(internal_fxpt, TaffoMath::zero, cntx, float_to_fixed);
  auto one = TaffoMath::make_fxp(internal_fxpt, TaffoMath::one, cntx, float_to_fixed);
  auto one_internal = TaffoMath::make_fxp(internal_fxpt, TaffoMath::one, cntx, float_to_fixed);


  std::string S_ret_point = "." + (!fxparg.isFloatingPoint() ? std::to_string(fxparg.scalarFracBitsAmt()) : "flt");

  // code gen
  auto int_8_zero = ConstantInt::get(Type::getInt8Ty(cntx), 0);
  auto int_8_one = ConstantInt::get(Type::getInt8Ty(cntx), 1);
  auto int_8_minus_one = ConstantInt::get(Type::getInt8Ty(cntx), -1);
  builder.CreateStore(int_8_zero, changeSign);
  builder.CreateStore(int_8_zero, changedFunction);
  builder.CreateStore(int_8_zero, specialAngle);

  BasicBlock *body = BasicBlock::Create(cntx, "body", new_f);
  builder.CreateBr(body);
  builder.SetInsertPoint(body);


  // handle fxp arg
  if (!new_arg_type->isFloatingPointTy()) {

    wrapper_printf(builder, m, "start arg: %i\n", builder.CreateLoad(arg_value));

    // handle unsigned arg
    if (!fxparg.scalarIsSigned()) {
      builder.CreateStore(builder.CreateLShr(builder.CreateLoad(arg_value), ConstantInt::get(new_arg_type, 1)), arg_value);
      fxparg.scalarFracBitsAmt() = fxparg.scalarFracBitsAmt() - 1;
      fxparg.scalarIsSigned() = true;
    }

    // handle negative
    if (is_sin) {
      // sin(-x) == -sin(x)

      BasicBlock *true_greater_zero =
          BasicBlock::Create(cntx, "true_greater_zero", new_f);
      BasicBlock *false_greater_zero = BasicBlock::Create(cntx, "body_2", new_f);

      generic = builder.CreateICmpSLT(builder.CreateLoad(arg_value, "arg"),
                                      zeroarg.first);
      generic =
          builder.CreateCondBr(generic, true_greater_zero, false_greater_zero);
      builder.SetInsertPoint(true_greater_zero);
      generic = builder.CreateSub(zeroarg.first,
                                  builder.CreateLoad(arg_value));
      builder.CreateStore(
          builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
          changeSign);
      builder.CreateStore(generic, arg_value);
      builder.CreateBr(false_greater_zero);
      builder.SetInsertPoint(false_greater_zero);


    } else {
      // cos(-x) == cos(x)
      {
        BasicBlock *true_greater_zero =
            BasicBlock::Create(cntx, "true_greater_zero", new_f);
        BasicBlock *false_greater_zero = BasicBlock::Create(cntx, "body", new_f);

        generic = builder.CreateICmpSLT(builder.CreateLoad(arg_value, "arg"),
                                        zeroarg.first);
        generic =
            builder.CreateCondBr(generic, true_greater_zero, false_greater_zero);
        builder.SetInsertPoint(true_greater_zero);
        generic = builder.CreateSub(zeroarg.first,
                                    builder.CreateLoad(arg_value));
        builder.CreateStore(generic, arg_value);
        builder.CreateBr(false_greater_zero);
        builder.SetInsertPoint(false_greater_zero);
      }
    }

    wrapper_printf(builder, m, "pre norm: %i\n", builder.CreateLoad(arg_value));
    if (fxparg.scalarIsSigned()) {
      builder.CreateStore(builder.CreateShl(builder.CreateLoad(arg_value), ConstantInt::get(new_arg_type, 1)), arg_value);
      fxparg.scalarFracBitsAmt() = fxparg.scalarFracBitsAmt() + 1;
      fxparg.scalarIsSigned() = false;
    }

    wrapper_printf(builder, m, "Fxparg scalarfrac %i\n", ConstantInt::get(new_arg_type, fxparg.scalarFracBitsAmt()));


    fixrangeSinCos(float_to_fixed, new_f, fxparg, internal_fxpt, arg_value, builder);

    bool pi_half_created = pi_half.first != nullptr;
    bool pi_created = pi.first != nullptr;
    bool pi_32_created = pi_32.first != nullptr;
    bool pi_2_created = pi_2.first != nullptr;


    if (is_sin) {
      // angle > pi_half && angle < pi sin(x) = cos(x - pi_half)
      if (pi_half_created && pi_created) {
        BasicBlock *in_II_quad = BasicBlock::Create(cntx, "in_II_quad", new_f);
        BasicBlock *not_in_II_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateICmpULT(
            pi_half.first,
            builder.CreateLoad(arg_value), "arg_greater_pi_half");
        Value *generic2 = builder.CreateICmpULT(
            builder.CreateLoad(arg_value), pi.first,
            "arg_less_pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_II_quad, not_in_II_quad);
        builder.SetInsertPoint(in_II_quad);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              pi_half.first),
                            arg_value);
        builder.CreateBr(not_in_II_quad);
        builder.SetInsertPoint(not_in_II_quad);
      }
      // angle > pi&& angle < pi_32(x) sin(x) = -sin(x - pi)
      if (pi_32_created && pi_created) {
        BasicBlock *in_III_quad = BasicBlock::Create(cntx, "in_III_quad", new_f);
        BasicBlock *not_in_III_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateICmpULT(pi.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi");
        Value *generic2 = builder.CreateICmpULT(
            builder.CreateLoad(arg_value),
            pi_32.first, "arg_less_pi_32");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_III_quad, not_in_III_quad);
        builder.SetInsertPoint(in_III_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              pi.first),
                            arg_value);
        builder.CreateBr(not_in_III_quad);
        builder.SetInsertPoint(not_in_III_quad);
      }
      // angle > pi_32&& angle < pi_2(x) sin(x) = -cos(x - pi_32);
      if (pi_32_created && pi_2_created) {
        BasicBlock *in_IV_quad = BasicBlock::Create(cntx, "in_IV_quad", new_f);
        BasicBlock *not_in_IV_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateICmpULT(pi_32.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi_32");
        Value *generic2 = builder.CreateICmpULT(
            builder.CreateLoad(arg_value), pi_2.first,
            "arg_less_2pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_IV_quad, not_in_IV_quad);
        builder.SetInsertPoint(in_IV_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              pi_32.first),
                            arg_value);
        builder.CreateBr(not_in_IV_quad);
        builder.SetInsertPoint(not_in_IV_quad);
      }
    } else {

      // angle > pi_half && angle < pi cos(x) = -sin(x - pi_half);
      if (pi_half_created && pi_created) {
        BasicBlock *in_II_quad = BasicBlock::Create(cntx, "in_II_quad", new_f);
        BasicBlock *not_in_II_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateICmpULT(
            pi_half.first,
            builder.CreateLoad(arg_value), "arg_greater_pi_half");
        Value *generic2 = builder.CreateICmpULT(
            builder.CreateLoad(arg_value), pi.first,
            "arg_less_pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_II_quad, not_in_II_quad);
        builder.SetInsertPoint(in_II_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              pi_half.first),
                            arg_value);
        builder.CreateBr(not_in_II_quad);
        builder.SetInsertPoint(not_in_II_quad);
      }
      // angle > pi&& angle < pi_32(x) cos(x) = -cos(x-pi)
      if (pi_32_created && pi_created) {
        BasicBlock *in_III_quad = BasicBlock::Create(cntx, "in_III_quad", new_f);
        BasicBlock *not_in_III_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateICmpULT(pi.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi");
        Value *generic2 = builder.CreateICmpULT(
            builder.CreateLoad(arg_value),
            pi_32.first, "arg_less_pi_32");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_III_quad, not_in_III_quad);
        builder.SetInsertPoint(in_III_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              pi.first),
                            arg_value);
        builder.CreateBr(not_in_III_quad);
        builder.SetInsertPoint(not_in_III_quad);
      }
      // angle > pi_32&& angle < pi_2(x) cos(x) = sin(angle - pi_32);
      if (pi_32_created && pi_2_created) {
        BasicBlock *in_IV_quad = BasicBlock::Create(cntx, "in_IV_quad", new_f);
        BasicBlock *not_in_IV_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateICmpULT(pi_32.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi_32");
        Value *generic2 = builder.CreateICmpULT(
            builder.CreateLoad(arg_value), pi_2.first,
            "arg_less_2pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_IV_quad, not_in_IV_quad);
        builder.SetInsertPoint(in_IV_quad);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateSub(builder.CreateLoad(arg_value),
                                              pi_32.first),
                            arg_value);
        builder.CreateBr(not_in_IV_quad);
        builder.SetInsertPoint(not_in_IV_quad);
      }
    }

  } else {
    // handle float arg

    // handle negative
    if (is_sin) {
      // sin(-x) == -sin(x)

      BasicBlock *true_greater_zero =
          BasicBlock::Create(cntx, "true_greater_zero", new_f);
      BasicBlock *false_greater_zero = BasicBlock::Create(cntx, "body_2", new_f);

      generic = builder.CreateFCmpOLT(builder.CreateLoad(arg_value, "arg"),
                                      zeroarg.first);
      generic =
          builder.CreateCondBr(generic, true_greater_zero, false_greater_zero);
      builder.SetInsertPoint(true_greater_zero);
      generic = builder.CreateFSub(zeroarg.first,
                                   builder.CreateLoad(arg_value));
      builder.CreateStore(
          builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
          changeSign);
      builder.CreateStore(generic, arg_value);
      builder.CreateBr(false_greater_zero);
      builder.SetInsertPoint(false_greater_zero);


    } else {
      // cos(-x) == cos(x)
      {
        BasicBlock *true_greater_zero =
            BasicBlock::Create(cntx, "true_greater_zero", new_f);
        BasicBlock *false_greater_zero = BasicBlock::Create(cntx, "body", new_f);

        generic = builder.CreateFCmpOLT(builder.CreateLoad(arg_value, "arg"),
                                        zeroarg.first);
        generic =
            builder.CreateCondBr(generic, true_greater_zero, false_greater_zero);
        builder.SetInsertPoint(true_greater_zero);
        generic = builder.CreateFSub(zeroarg.first,
                                     builder.CreateLoad(arg_value));
        builder.CreateStore(generic, arg_value);
        builder.CreateBr(false_greater_zero);
        builder.SetInsertPoint(false_greater_zero);
      }
    }

    fixrangeSinCos(float_to_fixed, new_f, fxparg, internal_fxpt, arg_value, builder);

    bool pi_half_created = pi_half.first != nullptr;
    bool pi_created = pi.first != nullptr;
    bool pi_32_created = pi_32.first != nullptr;
    bool pi_2_created = pi_2.first != nullptr;


    if (is_sin) {
      // angle > pi_half && angle < pi sin(x) = cos(x - pi_half)
      if (pi_half_created && pi_created) {
        BasicBlock *in_II_quad = BasicBlock::Create(cntx, "in_II_quad", new_f);
        BasicBlock *not_in_II_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateFCmpOLT(
            pi_half.first,
            builder.CreateLoad(arg_value), "arg_greater_pi_half");
        Value *generic2 = builder.CreateFCmpOLT(
            builder.CreateLoad(arg_value), pi.first,
            "arg_less_pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_II_quad, not_in_II_quad);
        builder.SetInsertPoint(in_II_quad);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateFSub(builder.CreateLoad(arg_value),
                                               pi_half.first),
                            arg_value);
        builder.CreateBr(not_in_II_quad);
        builder.SetInsertPoint(not_in_II_quad);
      }
      // angle > pi&& angle < pi_32(x) sin(x) = -sin(x - pi)
      if (pi_32_created && pi_created) {
        BasicBlock *in_III_quad = BasicBlock::Create(cntx, "in_III_quad", new_f);
        BasicBlock *not_in_III_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateFCmpOLT(pi.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi");
        Value *generic2 = builder.CreateFCmpOLT(
            builder.CreateLoad(arg_value),
            pi_32.first, "arg_less_pi_32");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_III_quad, not_in_III_quad);
        builder.SetInsertPoint(in_III_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateFSub(builder.CreateLoad(arg_value),
                                               pi.first),
                            arg_value);
        builder.CreateBr(not_in_III_quad);
        builder.SetInsertPoint(not_in_III_quad);
      }
      // angle > pi_32&& angle < pi_2(x) sin(x) = -cos(x - pi_32);
      if (pi_32_created && pi_2_created) {
        BasicBlock *in_IV_quad = BasicBlock::Create(cntx, "in_IV_quad", new_f);
        BasicBlock *not_in_IV_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateFCmpOLT(pi_32.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi_32");
        Value *generic2 = builder.CreateFCmpOLT(
            builder.CreateLoad(arg_value), pi_2.first,
            "arg_less_2pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_IV_quad, not_in_IV_quad);
        builder.SetInsertPoint(in_IV_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateFSub(builder.CreateLoad(arg_value),
                                               pi_32.first),
                            arg_value);
        builder.CreateBr(not_in_IV_quad);
        builder.SetInsertPoint(not_in_IV_quad);
      }
    } else {

      // angle > pi_half && angle < pi cos(x) = -sin(x - pi_half);
      if (pi_half_created && pi_created) {
        BasicBlock *in_II_quad = BasicBlock::Create(cntx, "in_II_quad", new_f);
        BasicBlock *not_in_II_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateFCmpOLT(
            pi_half.first,
            builder.CreateLoad(arg_value), "arg_greater_pi_half");
        Value *generic2 = builder.CreateFCmpOLT(
            builder.CreateLoad(arg_value), pi.first,
            "arg_less_pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_II_quad, not_in_II_quad);
        builder.SetInsertPoint(in_II_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateFSub(builder.CreateLoad(arg_value),
                                               pi_half.first),
                            arg_value);
        builder.CreateBr(not_in_II_quad);
        builder.SetInsertPoint(not_in_II_quad);
      }
      // angle > pi&& angle < pi_32(x) cos(x) = -cos(x-pi)
      if (pi_32_created && pi_created) {
        BasicBlock *in_III_quad = BasicBlock::Create(cntx, "in_III_quad", new_f);
        BasicBlock *not_in_III_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateFCmpOLT(pi.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi");
        Value *generic2 = builder.CreateFCmpOLT(
            builder.CreateLoad(arg_value),
            pi_32.first, "arg_less_pi_32");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_III_quad, not_in_III_quad);
        builder.SetInsertPoint(in_III_quad);
        builder.CreateStore(
            builder.CreateXor(builder.CreateLoad(changeSign), int_8_minus_one),
            changeSign);
        builder.CreateStore(builder.CreateFSub(builder.CreateLoad(arg_value),
                                               pi.first),
                            arg_value);
        builder.CreateBr(not_in_III_quad);
        builder.SetInsertPoint(not_in_III_quad);
      }
      // angle > pi_32&& angle < pi_2(x) cos(x) = sin(angle - pi_32);
      if (pi_32_created && pi_2_created) {
        BasicBlock *in_IV_quad = BasicBlock::Create(cntx, "in_IV_quad", new_f);
        BasicBlock *not_in_IV_quad = BasicBlock::Create(cntx, "body", new_f);
        generic = builder.CreateFCmpOLT(pi_32.first,
                                        builder.CreateLoad(arg_value),
                                        "arg_greater_pi_32");
        Value *generic2 = builder.CreateFCmpOLT(
            builder.CreateLoad(arg_value), pi_2.first,
            "arg_less_2pi");
        generic = builder.CreateAnd(generic, generic2);
        builder.CreateCondBr(generic, in_IV_quad, not_in_IV_quad);
        builder.SetInsertPoint(in_IV_quad);
        builder.CreateStore(builder.CreateXor(builder.CreateLoad(changedFunction),
                                              int_8_minus_one),
                            changedFunction);
        builder.CreateStore(builder.CreateFSub(builder.CreateLoad(arg_value),
                                               pi_32.first),
                            arg_value);
        builder.CreateBr(not_in_IV_quad);
        builder.SetInsertPoint(not_in_IV_quad);
      }
    }
  }


  Value *sin_g = generateSinLUT(float_to_fixed, new_f, internal_fxpt, builder);
  // Value *cos_g = generateCosLUT(this, oldf, internal_fxpt, builder);
  auto zero_arg = zero.first;
  Value *tmp_angle = builder.CreateLoad(arg_value);


  if (!fxpret.isFloatingPoint()) {
    std::string function_name("llvm.udiv.fix.i");
    function_name.append(std::to_string(internal_fxpt.scalarToLLVMType(cntx)->getScalarSizeInBits()));

    wrapper_printf(builder, m, "post subtraction %i \n", builder.CreateLoad(arg_value));

    Function *udiv = nullptr;
    if ((udiv = new_f->getParent()->getFunction(function_name)) == 0) {
      std::vector<llvm::Type *> fun_arguments;
      fun_arguments.push_back(
          internal_fxpt.scalarToLLVMType(cntx)); // depends on your type
      fun_arguments.push_back(
          internal_fxpt.scalarToLLVMType(cntx)); // depends on your type
      fun_arguments.push_back(Type::getInt32Ty(cntx));
      FunctionType *fun_type = FunctionType::get(
          internal_fxpt.scalarToLLVMType(cntx), fun_arguments, false);
      udiv = llvm::Function::Create(fun_type, GlobalValue::ExternalLinkage,
                                    function_name, new_f->getParent());
    }


    generic = builder.CreateCall(
        udiv, {tmp_angle, builder.CreateLShr(pi_half_internal.first, int(log2(MathZ))),
               llvm::ConstantInt::get(internal_fxpt.scalarToLLVMType(cntx),
                                      internal_fxpt.scalarFracBitsAmt() -
                                          int(log2(MathZ)))});
    generic = builder.CreateLShr(
        generic, llvm::ConstantInt::get(internal_fxpt.scalarToLLVMType(cntx),
                                        internal_fxpt.scalarFracBitsAmt() -
                                            int(log2(MathZ))));


  } else {

    generic = builder.CreateFDiv(builder.CreateFMul(tmp_angle, llvm::ConstantFP::get(pi_half_internal.first->getType(), MathZ)), pi_half_internal.first);


    generic = builder.CreateFPToUI(generic, llvm::Type::getInt32Ty(cntx));
  }

  wrapper_printf(builder, m, "post div %i \n", generic);

  auto y_value = builder.CreateAlloca(new_arg_type);
  auto x_value = builder.CreateAlloca(new_arg_type);

  builder.CreateStore(builder.CreateLoad(builder.CreateGEP(sin_g, {zero_arg, generic})),
                      y_value);
  generic = builder.CreateSub(llvm::ConstantInt::get(internal_fxpt.scalarToLLVMType(cntx), MathZ), generic);
  builder.CreateStore(builder.CreateLoad(builder.CreateGEP(sin_g, {zero_arg, generic})),
                      x_value);


  wrapper_printf(builder, m, "x %i \n", builder.CreateLoad(x_value));
  wrapper_printf(builder, m, "y %i \n", builder.CreateLoad(y_value));


  BasicBlock *return_point =
      BasicBlock::Create(cntx, "return_point", new_f);
  builder.CreateBr(return_point);
  builder.SetInsertPoint(return_point);
  {
    auto zero_arg = zero.first;
    auto zero_bool = int_8_zero;
    if (is_sin) {
      generic = builder.CreateSelect(
          builder.CreateICmpEQ(builder.CreateLoad(changedFunction), zero_bool),
          builder.CreateLoad(y_value), builder.CreateLoad(x_value));
      builder.CreateStore(generic, arg_value);
    } else {
      generic = builder.CreateSelect(
          builder.CreateICmpEQ(builder.CreateLoad(changedFunction), zero_bool),
          builder.CreateLoad(x_value), builder.CreateLoad(y_value));
      builder.CreateStore(generic, arg_value);
    }

    generic = builder.CreateSelect(
        builder.CreateICmpEQ(builder.CreateLoad(changeSign), zero_bool),
        builder.CreateLoad(arg_value),
        builder.CreateSub(zero_arg, builder.CreateLoad(arg_value)));
    builder.CreateStore(generic, arg_value);
  }
  if (!internal_fxpt.isFloatingPoint()) {
    if (internal_fxpt.scalarFracBitsAmt() > truefxpret.scalarFracBitsAmt()) {
      builder.CreateStore(builder.CreateAShr(builder.CreateLoad(arg_value), internal_fxpt.scalarFracBitsAmt() - truefxpret.scalarFracBitsAmt()), arg_value);
    } else if (internal_fxpt.scalarFracBitsAmt() < truefxpret.scalarFracBitsAmt()) {
      builder.CreateStore(builder.CreateShl(builder.CreateLoad(arg_value), truefxpret.scalarFracBitsAmt() - internal_fxpt.scalarFracBitsAmt()), arg_value);
    }
  }


  auto ret = builder.CreateLoad(arg_value);
  wrapper_printf(builder, m, "ret :%i\n", ret);
  builder.CreateRet(ret);

  return true;
}


llvm::Function *TaffoMath::CreateSpecialFunction::sinHandler(OldInfo &old_info, NewInfo &new_info)
{
  if (old_info.old_args_fxpt.size() == 1) {
    createSinCos(this->float_to_fixed, new_info.new_f, old_info.old_f, old_info.old_ret_fxpt, old_info.old_args_fxpt[0]);
  } else {
    llvm_unreachable("Incorrect args");
  }
  return new_info.new_f;
}


llvm::Function *TaffoMath::CreateSpecialFunction::cosHandler(OldInfo &old_info, NewInfo &new_info)
{
  if (old_info.old_args_fxpt.size() == 1) {
    createSinCos(this->float_to_fixed, new_info.new_f, old_info.old_f, old_info.old_ret_fxpt, old_info.old_args_fxpt[0]);
  } else {
    llvm_unreachable("Incorrect args");
  }
  return new_info.new_f;
}
