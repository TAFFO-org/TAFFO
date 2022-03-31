#include "CreateSpecialFunction.h"
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


Value *generateASinLUT(flttofix::FloatToFixed *ref, Function *new_f, flttofix::FixedPointType &fxpret,
                       llvm::IRBuilder<> &builder)
{

  LLVM_DEBUG(llvm::dbgs() << "GENERATE ASIN LUT\n");
  if (!fxpret.isFloatingPoint()) {
    std::vector<llvm::Constant *> asin_arr_const;

    for (int i = 0; i <= MathZ; ++i) {
      flttofix::FixedPointType match = flttofix::FixedPointType(true, fxpret.scalarFracBitsAmt(), fxpret.scalarBitsAmt());
      auto tmp = TaffoMath::createFixedPointFromConst(
          new_f->getContext(), ref, asin(static_cast<double>(i) * 2.0 / static_cast<double>(MathZ) - 1.0), match);


      if (tmp == nullptr) {
        llvm_unreachable("Ma nooooo\n");
      }
      asin_arr_const.push_back(tmp);
    }
    auto asin_ArrayType =
        llvm::ArrayType::get(fxpret.scalarToLLVMType(new_f->getContext()), MathZ + 1);
    auto asin_ConstArray = llvm::ConstantArray::get(
        asin_ArrayType, llvm::ArrayRef<llvm::Constant *>(asin_arr_const));
    auto alignement_sin =
        new_f->getParent()->getDataLayout().getPrefTypeAlign(asin_arr_const.front()->getType());
    auto asin_arry_g =
        TaffoMath::createGlobalConst(new_f->getParent(), "asin_global." + std::to_string(fxpret.scalarFracBitsAmt()) + "_" + std::to_string(fxpret.scalarBitsAmt()), asin_ArrayType,
                                     asin_ConstArray, alignement_sin);
    return asin_arry_g;
  } else {
    std::vector<llvm::Constant *> asin_arr_const;

    for (int i = 0; i <= MathZ; ++i) {
      flttofix::FixedPointType match = flttofix::FixedPointType(true, fxpret.scalarFracBitsAmt(), fxpret.scalarBitsAmt());
      auto tmp = TaffoMath::createFixedPointFromConst(
          new_f->getContext(), ref, asin(static_cast<double>(i) * 2.0 / static_cast<double>(MathZ) - 1.0), match);


      if (tmp == nullptr) {
        llvm_unreachable("Ma nooooo\n");
      }
      asin_arr_const.push_back(tmp);
    }
    auto asin_ArrayType =
        llvm::ArrayType::get(fxpret.scalarToLLVMType(new_f->getContext()), MathZ + 1);
    auto asin_ConstArray = llvm::ConstantArray::get(
        asin_ArrayType, llvm::ArrayRef<llvm::Constant *>(asin_arr_const));
    auto alignement_asin =
        new_f->getParent()->getDataLayout().getPrefTypeAlign(asin_arr_const.front()->getType());
    auto asin_arry_g =
        TaffoMath::createGlobalConst(new_f->getParent(), std::string("asin_global.") + (asin_arr_const[0]->getType() == llvm::Type::getFloatTy(new_f->getContext()) ? "float" : "duble"), asin_ArrayType,
                                     asin_ConstArray, alignement_asin);
    return asin_arry_g;
  }
}


Value *generateACosLUT(flttofix::FloatToFixed *ref, Function *new_f, flttofix::FixedPointType &fxpret,
                       llvm::IRBuilder<> &builder)
{

  LLVM_DEBUG(llvm::dbgs() << "GENERATE ACOS LUT\n");
  if (!fxpret.isFloatingPoint()) {
    fxpret.scalarIsSigned() = false;
    std::vector<llvm::Constant *> acos_arr_const;

    for (int i = 0; i <= MathZ; ++i) {
      flttofix::FixedPointType match = flttofix::FixedPointType(false, fxpret.scalarFracBitsAmt(), fxpret.scalarBitsAmt());
      auto tmp = TaffoMath::createFixedPointFromConst(
          new_f->getContext(), ref, acos(static_cast<double>(i) * 2.0 / static_cast<double>(MathZ) - 1.0), match);


      if (tmp == nullptr) {
        llvm_unreachable("Ma nooooo\n");
      }
      acos_arr_const.push_back(tmp);
    }
    auto acos_ArrayType =
        llvm::ArrayType::get(fxpret.scalarToLLVMType(new_f->getContext()), MathZ + 1);
    auto acos_ConstArray = llvm::ConstantArray::get(
        acos_ArrayType, llvm::ArrayRef<llvm::Constant *>(acos_arr_const));
    auto alignement_acos =
        new_f->getParent()->getDataLayout().getPrefTypeAlign(acos_arr_const.front()->getType());
    auto acos_arry_g =
        TaffoMath::createGlobalConst(new_f->getParent(), "acos_arr_const." + std::to_string(fxpret.scalarFracBitsAmt()) + "_" + std::to_string(fxpret.scalarBitsAmt()), acos_ArrayType,
                                     acos_ConstArray, alignement_acos);
    return acos_arry_g;
  } else {
    std::vector<llvm::Constant *> acos_arr_const;

    for (int i = 0; i <= MathZ; ++i) {
      flttofix::FixedPointType match = flttofix::FixedPointType(true, fxpret.scalarFracBitsAmt(), fxpret.scalarBitsAmt());
      auto tmp = TaffoMath::createFixedPointFromConst(
          new_f->getContext(), ref, acos(static_cast<double>(i) * 2.0 / static_cast<double>(MathZ) - 1.0), match);


      if (tmp == nullptr) {
        llvm_unreachable("Ma nooooo\n");
      }
      acos_arr_const.push_back(tmp);
    }
    auto acos_ArrayType =
        llvm::ArrayType::get(fxpret.scalarToLLVMType(new_f->getContext()), MathZ + 1);
    auto acos_ConstArray = llvm::ConstantArray::get(
        acos_ArrayType, llvm::ArrayRef<llvm::Constant *>(acos_arr_const));
    auto alignement_acos =
        new_f->getParent()->getDataLayout().getPrefTypeAlign(acos_arr_const.front()->getType());
    auto acos_arry_g =
        TaffoMath::createGlobalConst(new_f->getParent(), std::string("acos_global.") + (acos_arr_const[0]->getType() == llvm::Type::getFloatTy(new_f->getContext()) ? "float" : "duble"), acos_ArrayType,
                                     acos_ConstArray, alignement_acos);
    return acos_arry_g;
  }
}


bool create_asin_acos(flttofix::FloatToFixed *float_to_fixed,
                      llvm::Function *new_f, llvm::Function *old_f, const flttofix::FixedPointType *old_ret_fxpt,
                      const flttofix::FixedPointType *old_arg_fxpt, std::function<Value *(flttofix::FloatToFixed *, Function *, flttofix::FixedPointType &, llvm::IRBuilder<> &)> lut_creator)
{
  LLVM_DEBUG(llvm::dbgs() << "####" << __func__ << " ####");
  Value *generic;
  // retrive context used in later instruction
  llvm::LLVMContext &cntx(old_f->getContext());
  //retruve the data llayout
  DataLayout dataLayout(old_f->getParent());
  // Create new block
  BasicBlock::Create(cntx, "Entry", new_f);
  BasicBlock *where = &(new_f->getEntryBlock());
  //builder to new_f
  llvm::IRBuilder<> builder(where, where->getFirstInsertionPt());
  // get return type fixed point
  flttofix::FixedPointType fxpret = *old_ret_fxpt;
  flttofix::FixedPointType fxparg = *old_arg_fxpt;
  flttofix::FixedPointType truefxpret = *old_ret_fxpt;


  auto fxpt_norm_arg = flttofix::FixedPointType(true, fxparg.scalarBitsAmt() - 2, fxparg.scalarBitsAmt());
  auto one = TaffoMath::make_fxp(fxpt_norm_arg, TaffoMath::one, cntx, float_to_fixed);
  auto new_arg_type = new_f->getArg(0)->getType();
  auto arg_store = builder.CreateAlloca(new_arg_type);

  builder.CreateStore(new_f->getArg(0), arg_store);
  TaffoMath::wrapper_printf(builder, new_f, std::string("Angle stored %i\n"), builder.CreateLoad(arg_store));


  generic = builder.CreateLoad(arg_store);

  if (!fxparg.isFloatingPoint()) {

    if (fxparg.scalarFracBitsAmt() < fxparg.scalarBitsAmt() - 2) {
      generic = builder.CreateShl(builder.CreateLoad(arg_store), ConstantInt::get(new_arg_type, fxparg.scalarBitsAmt() - 2 - fxparg.scalarFracBitsAmt()));
      if (!fxparg.scalarIsSigned()) {
        fxparg.scalarIsSigned() = true;
      }
    } else {
      if (fxparg.scalarIsSigned()) {
        generic = builder.CreateAShr(builder.CreateLoad(arg_store), ConstantInt::get(new_arg_type, -fxparg.scalarBitsAmt() - 2 + fxparg.scalarFracBitsAmt()));
      } else {
        generic = builder.CreateLShr(builder.CreateLoad(arg_store), ConstantInt::get(new_arg_type, -fxparg.scalarBitsAmt() - 2 + fxparg.scalarFracBitsAmt()));
        fxparg.scalarIsSigned() = true;
      }
    }
    builder.CreateStore(generic, arg_store);
    TaffoMath::wrapper_printf(builder, new_f, std::string("After Shift %i\n"), builder.CreateLoad(arg_store));

    builder.CreateStore(builder.CreateSelect(
                            builder.CreateICmpSGE(ConstantInt::get(new_arg_type, 0), generic),
                            builder.CreateAdd(one.first, generic),
                            builder.CreateAdd(generic, one.first)),
                        arg_store);
    TaffoMath::wrapper_printf(builder, new_f, std::string("Angle +1 and unsigned %i\n"), builder.CreateLoad(arg_store));

    fxparg.scalarIsSigned() = false;

  } else {
    llvm_unreachable("not implemented");
  }


  if (!fxpret.isFloatingPoint()) {
    auto internal_fxp = flttofix::FixedPointType(true, fxparg.scalarBitsAmt() - 2, fxparg.scalarBitsAmt());
    auto lut = lut_creator(float_to_fixed, new_f, internal_fxp, builder);
    //  2 bit to store number +1 (to divide by 2) +  number of bit to store the multiplication of the table lenght
    generic = builder.CreateLShr(builder.CreateLoad(arg_store), ConstantInt::get(new_arg_type, fxparg.scalarBitsAmt() - 2 + 1 - static_cast<int>(log2(MathZ))));

    TaffoMath::wrapper_printf(builder, new_f, std::string("Arc sin gep: %i\n"), generic);
    builder.CreateStore(builder.CreateLoad(builder.CreateGEP(lut, {ConstantInt::get(new_arg_type, 0), generic})),
                        arg_store);
    if (fxparg.scalarBitsAmt() - 2 > fxpret.scalarFracBitsAmt()) {
      if (taffo::start_with(new_f->getName(), "asin")) {
        builder.CreateStore(builder.CreateAShr(builder.CreateLoad(arg_store), ConstantInt::get(new_arg_type, fxparg.scalarBitsAmt() - 2 - fxpret.scalarFracBitsAmt())), arg_store);
      } else {
        builder.CreateStore(builder.CreateLShr(builder.CreateLoad(arg_store), ConstantInt::get(new_arg_type, fxparg.scalarBitsAmt() - 2 - fxpret.scalarFracBitsAmt())), arg_store);
      }
    } else if (fxparg.scalarBitsAmt() - 2 < fxpret.scalarFracBitsAmt()) {
      builder.CreateStore(builder.CreateAShr(builder.CreateLoad(arg_store), ConstantInt::get(new_arg_type, -fxparg.scalarBitsAmt() - 2 + fxpret.scalarFracBitsAmt())), arg_store);
    }
    builder.CreateRet(builder.CreateLoad(arg_store));

  } else {
    llvm_unreachable("not implemented");
  }
}


bool create_acos(flttofix::FloatToFixed *float_to_fixed,
                 llvm::Function *new_f, llvm::Function *old_f, const flttofix::FixedPointType *old_ret_fxpt,
                 const flttofix::FixedPointType *old_arg_fxpt)
{
}


llvm::Function *TaffoMath::CreateSpecialFunction::asinHandler(OldInfo &old_info, NewInfo &new_info)
{
  if (old_info.old_args_fxpt.size() == 1) {
    create_asin_acos(this->float_to_fixed, new_info.new_f, old_info.old_f, old_info.old_ret_fxpt, old_info.old_args_fxpt[0], &generateASinLUT);
  } else {
    llvm_unreachable("Incorrect args");
  }
  return new_info.new_f;
}


llvm::Function *TaffoMath::CreateSpecialFunction::acosHandler(OldInfo &old_info, NewInfo &new_info)
{
  if (old_info.old_args_fxpt.size() == 1) {
    create_asin_acos(this->float_to_fixed, new_info.new_f, old_info.old_f, old_info.old_ret_fxpt, old_info.old_args_fxpt[0], &generateACosLUT);
  } else {
    llvm_unreachable("Incorrect args");
  }
  return new_info.new_f;
}