#include "TAFFOMath.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>
#include <vector>

#define DEBUG_TYPE "taffo-conversion"


void print_fixp( Module* M, IRBuilder<> &builder, const char * c_str, Value *to_print, int comma) {
  auto &cont = M->getContext();
  Value *generic =
      builder.CreateSIToFP(to_print, Type::getDoubleTy(cont));
  generic = builder.CreateFDiv(
      generic, ConstantFP::get(Type::getDoubleTy(cont), pow(2, comma)));
  Value *str = builder.CreateGlobalStringPtr(c_str);
  Function *fun;
  if ((fun = M->getFunction("printf")) == 0) {
    std::vector<Type *> fun_arguments;
    fun_arguments.push_back(Type::getInt8PtrTy(cont)); // depends on your type
    FunctionType *fun_type =
        FunctionType::get(Type::getInt32Ty(cont), fun_arguments, true);
    fun = llvm::Function::Create(fun_type, GlobalValue::ExternalLinkage,
                                 "printf", M);    
  }
  builder.CreateCall(fun, {str, generic});
}

bool createACos(FloatToFixed* ref, llvm::Function *newfs, llvm::Function *oldf) {
  newfs->deleteBody();
  Value *generic;
  Module *M = oldf->getParent();
  // retrive context used in later instruction
  llvm::LLVMContext &cont(oldf->getContext());
  DataLayout dataLayout(M);
  // get first basick block of function
  BasicBlock::Create(cont, "Entry", newfs);
  BasicBlock *where = &(newfs->getEntryBlock());
  IRBuilder<> builder(where, where->getFirstInsertionPt());
  // get return type fixed point
  flttofix::FixedPointType fxpret;
  flttofix::FixedPointType fxparg;
  bool foundRet = false;
  bool foundArg = false;
  int arg_size = newfs->getArg(0)->getType()->getPrimitiveSizeInBits();
  Type *arg_type = newfs->getArg(0)->getType();
  Type *ret_type = newfs->getReturnType();
  assert(arg_type->getTypeID() == ret_type->getTypeID() && "mismatch type");
  int ret_size = newfs->getReturnType()->getPrimitiveSizeInBits();
  TaffoMath::getFixedFromRet(ref, oldf, fxpret, foundRet);
  // get argument fixed point
  TaffoMath::getFixedFromArg(ref, oldf, fxparg, 0, foundArg);
  if (!foundRet || !foundArg) {
    return partialSpecialCall(newfs, foundRet, fxpret);
  }
  // create variable
  TaffoMath::pair_ftp_value<llvm::Value *> x(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> y(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> theta(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> t(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> x1(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> y1(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> d(
      FixedPointType(true, arg_size - 3, arg_size));

 //const
  TaffoMath::pair_ftp_value<llvm::Constant *> one(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Constant *> minus(
      FixedPointType(true, arg_size - 3, arg_size));

  //assign constant
  auto zero = ConstantInt::get(Type::getInt32Ty(cont), 0);
  TaffoMath::createFixedPointFromConst(
      cont, ref, 1,  x.fpt, one.value, one.fpt); 
TaffoMath::createFixedPointFromConst(
    cont, ref, -1,  x.fpt, minus.value, minus.fpt); 

  // alloca variable
  x.value = builder.CreateAlloca(arg_type, nullptr, "x");
  y.value = builder.CreateAlloca(arg_type, nullptr, "y");
  theta.value = builder.CreateAlloca(arg_type, nullptr, "theta");
  t.value = builder.CreateAlloca(arg_type, nullptr, "t");
  x1.value = builder.CreateAlloca(arg_type, nullptr, "x1");
  y1.value = builder.CreateAlloca(arg_type, nullptr, "y1");
  d.value = builder.CreateAlloca(arg_type, nullptr, "d");



  // create arctan
  LLVM_DEBUG(dbgs() << "Create arctan table"
                    << "\n");
  TaffoMath::pair_ftp_value<llvm::Constant *, TaffoMath::TABLELENGHT>    arctan_2power;

    for (int i = 0; i < TaffoMath::TABLELENGHT; i++) {
      arctan_2power.fpt.push_back(flttofix::FixedPointType(fxpret));
      Constant *tmp = nullptr;
      auto &current_fpt = arctan_2power.fpt.front();
      TaffoMath::createFixedPointFromConst(cont, ref,
                                           TaffoMath::arctan_2power[i],
                                           theta.fpt, tmp, current_fpt);
      arctan_2power.value.push_back(tmp);
      LLVM_DEBUG(dbgs() << i << ")");
    }

    auto arctanArrayType = llvm::ArrayType::get(
        arctan_2power.value[0]->getType(), TaffoMath::TABLELENGHT);

    LLVM_DEBUG(dbgs() << "ArrayType  " << arctanArrayType << "\n");
    auto arctanConstArray = llvm::ConstantArray::get(
        arctanArrayType, llvm::ArrayRef<llvm::Constant *>(arctan_2power.value));
    LLVM_DEBUG(dbgs() << "ConstantDataArray tmp2 " << arctanConstArray << "\n");
    auto alignement_arctan =
        dataLayout.getPrefTypeAlignment(arctan_2power.value[0]->getType());
    auto arctan_g = TaffoMath::createGlobalConst(
        M, "arctan_g." + std::to_string(theta.fpt.scalarFracBitsAmt()),
        arctanArrayType, arctanConstArray, alignement_arctan);

    auto loop_entry = BasicBlock::Create(cont, "Loop entry", newfs);
    auto cordic_body = BasicBlock::Create(cont, "cordic body", newfs);
    auto end_loop = BasicBlock::Create(cont, "end", newfs);
    auto i = builder.CreateAlloca(Type::getInt32Ty(cont),nullptr, "i");
    builder.CreateStore(zero, i);


    //set up value

    builder.CreateStore(zero, y.value);
    builder.CreateStore(zero, theta.value);
    builder.CreateStore(one.value, x.value);
    if(fxparg.scalarFracBitsAmt() > arg_size - 3){
        builder.CreateStore(builder.CreateAShr(newfs->getArg(0), fxparg.scalarFracBitsAmt() - arg_size + 3), t.value);
    }else if (fxparg.scalarFracBitsAmt() < arg_size - 3){
        builder.CreateStore(builder.CreateShl(newfs->getArg(0), -fxparg.scalarFracBitsAmt() + arg_size - 3), t.value);
    }else{
        builder.CreateStore(newfs->getArg(0), t.value);
    }
    builder.CreateBr(loop_entry);
    //end entry

    //begin loop entry
    builder.SetInsertPoint(loop_entry);
    generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(i), i), ConstantInt::get(Type::getInt32Ty(cont), arg_size/2));
    builder.CreateCondBr(generic, cordic_body, end_loop );

    builder.SetInsertPoint(cordic_body);

    // y>0
    generic= builder.CreateICmpSGE(builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value),  zero);
    
    // d = 1 if x >= t else -1
    builder.CreateStore(builder.CreateSelect(
        builder.CreateICmpSGE(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value), builder.CreateLoad(getElementTypeFromValuePointer(t.value), t.value)), 
        builder.CreateSelect(generic, one.value, zero),         
        builder.CreateSelect(generic, zero, one.value))
    ,d.value);

    
    //theta = theta + 2*d*math.atan(math.pow(2,-i))
    auto tmp = builder.CreateGEP(getElementTypeFromValuePointer(arctan_g), arctan_g, {zero,builder.CreateLoad(getElementTypeFromValuePointer(i), i)});
    generic = builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(tmp),tmp), ConstantInt::get(Type::getInt32Ty(cont), 1));
    generic = builder.CreateSelect(builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero),builder.CreateSub(zero, generic), generic);
    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value), generic ), theta.value);


    //    t = t + t*math.pow(2,-2*i) 
    generic = builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(i), i), ConstantInt::get(Type::getInt32Ty(cont), 1));
    builder.CreateStore(
        builder.CreateAdd(
            builder.CreateLoad(getElementTypeFromValuePointer(t.value), t.value), 
            builder.CreateAShr( builder.CreateLoad(getElementTypeFromValuePointer(t.value), t.value), generic
            )), t.value);


    //    x1= x-d*math.pow(2,-i)*y
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value), generic), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value), generic)), 
        x1.value);

    
    // y1= d*math.pow(2,-i)*x + y
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value), generic), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value), generic)), 
        y1.value);


    //    x= x1-d*math.pow(2,-i)*y1
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(y1.value), y1.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(x1.value), x1.value), generic), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(x1.value), x1.value), generic)), 
        x.value);
    
    // y= d*math.pow(2,-i)*x1 + y1
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(x1.value), x1.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(y1.value), y1.value), generic), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(y1.value), y1.value),generic)), 
        y.value);

    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(i), i), ConstantInt::get(Type::getInt32Ty(cont), 1)), i );

    builder.CreateBr(loop_entry);

    builder.SetInsertPoint(end_loop);

    if(fxpret.scalarFracBitsAmt() > arg_size - 3){
        builder.CreateStore(builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value), fxpret.scalarFracBitsAmt() - arg_size + 3), theta.value);
    }else if (fxpret.scalarFracBitsAmt() < arg_size - 3){
        builder.CreateStore(builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value), -fxpret.scalarFracBitsAmt() + arg_size - 3), theta.value);
    }
    builder.CreateRet(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value));


  }



  bool createASin( FloatToFixed * ref, llvm::Function * newfs, llvm::Function * oldf) {
  newfs->deleteBody();
  Value *generic;
  Module *M = oldf->getParent();
  // retrive context used in later instruction
  llvm::LLVMContext &cont(oldf->getContext());
  DataLayout dataLayout(M);
  // get first basick block of function
  BasicBlock::Create(cont, "Entry", newfs);
  BasicBlock *where = &(newfs->getEntryBlock());
  IRBuilder<> builder(where, where->getFirstInsertionPt());
  // get return type fixed point
  flttofix::FixedPointType fxpret;
  flttofix::FixedPointType fxparg;
  bool foundRet = false;
  bool foundArg = false;
  int arg_size = newfs->getArg(0)->getType()->getPrimitiveSizeInBits();
  Type *arg_type = newfs->getArg(0)->getType();
  Type *ret_type = newfs->getReturnType();
  assert(arg_type->getTypeID() == ret_type->getTypeID() && "mismatch type");
  int ret_size = newfs->getReturnType()->getPrimitiveSizeInBits();
  TaffoMath::getFixedFromRet(ref, oldf, fxpret, foundRet);
  // get argument fixed point
  TaffoMath::getFixedFromArg(ref, oldf, fxparg, 0, foundArg);
  if (!foundRet || !foundArg) {
    return partialSpecialCall(newfs, foundRet, fxpret);
  }
  // create variable
  TaffoMath::pair_ftp_value<llvm::Value *> x(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> y(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> theta(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> t(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> x1(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> y1(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Value *> d(
      FixedPointType(true, arg_size - 3, arg_size));

 //const
  TaffoMath::pair_ftp_value<llvm::Constant *> one(
      FixedPointType(true, arg_size - 3, arg_size));
  TaffoMath::pair_ftp_value<llvm::Constant *> minus(
      FixedPointType(true, arg_size - 3, arg_size));

  //assign constant
  auto zero = ConstantInt::get(Type::getInt32Ty(cont), 0);
  TaffoMath::createFixedPointFromConst(
      cont, ref, 1,  x.fpt, one.value, one.fpt); 
TaffoMath::createFixedPointFromConst(
    cont, ref, -1,  x.fpt, minus.value, minus.fpt); 

  // alloca variable
  x.value = builder.CreateAlloca(arg_type, nullptr, "x");
  y.value = builder.CreateAlloca(arg_type, nullptr, "y");
  theta.value = builder.CreateAlloca(arg_type, nullptr, "theta");
  t.value = builder.CreateAlloca(arg_type, nullptr, "t");
  x1.value = builder.CreateAlloca(arg_type, nullptr, "x1");
  y1.value = builder.CreateAlloca(arg_type, nullptr, "y1");
  d.value = builder.CreateAlloca(arg_type, nullptr, "d");



  // create arctan
  LLVM_DEBUG(dbgs() << "Create arctan table"
                    << "\n");
  TaffoMath::pair_ftp_value<llvm::Constant *, TaffoMath::TABLELENGHT>    arctan_2power;

    for (int i = 0; i < TaffoMath::TABLELENGHT; i++) {
      arctan_2power.fpt.push_back(flttofix::FixedPointType(fxpret));
      Constant *tmp = nullptr;
      auto &current_fpt = arctan_2power.fpt.front();
      TaffoMath::createFixedPointFromConst(cont, ref,
                                           TaffoMath::arctan_2power[i],
                                           theta.fpt, tmp, current_fpt);
      arctan_2power.value.push_back(tmp);
      LLVM_DEBUG(dbgs() << i << ")");
    }

    auto arctanArrayType = llvm::ArrayType::get(
        arctan_2power.value[0]->getType(), TaffoMath::TABLELENGHT);

    LLVM_DEBUG(dbgs() << "ArrayType  " << arctanArrayType << "\n");
    auto arctanConstArray = llvm::ConstantArray::get(
        arctanArrayType, llvm::ArrayRef<llvm::Constant *>(arctan_2power.value));
    LLVM_DEBUG(dbgs() << "ConstantDataArray tmp2 " << arctanConstArray << "\n");
    auto alignement_arctan =
        dataLayout.getPrefTypeAlignment(arctan_2power.value[0]->getType());
    auto arctan_g = TaffoMath::createGlobalConst(
        M, "arctan_g." + std::to_string(theta.fpt.scalarFracBitsAmt()),
        arctanArrayType, arctanConstArray, alignement_arctan);

    auto loop_entry = BasicBlock::Create(cont, "Loop entry", newfs);
    auto cordic_body = BasicBlock::Create(cont, "cordic body", newfs);
    auto end_loop = BasicBlock::Create(cont, "end", newfs);
    auto i = builder.CreateAlloca(Type::getInt32Ty(cont),nullptr, "i");
    builder.CreateStore(zero, i);


    //set up value

    builder.CreateStore(zero, y.value);
    builder.CreateStore(zero, theta.value);
    builder.CreateStore(one.value, x.value);
    if(fxparg.scalarFracBitsAmt() > arg_size - 3){
        builder.CreateStore(builder.CreateAShr(newfs->getArg(0), fxparg.scalarFracBitsAmt() - arg_size + 3), t.value);
    }else if (fxparg.scalarFracBitsAmt() < arg_size - 3){
        builder.CreateStore(builder.CreateShl(newfs->getArg(0), -fxparg.scalarFracBitsAmt() + arg_size - 3), t.value);
    }else{
        builder.CreateStore(newfs->getArg(0), t.value);
    }
    builder.CreateBr(loop_entry);
    //end entry

    //begin loop entry
    builder.SetInsertPoint(loop_entry);
    generic = builder.CreateICmpSLT(builder.CreateLoad(getElementTypeFromValuePointer(i), i), ConstantInt::get(Type::getInt32Ty(cont), arg_size/2));
    builder.CreateCondBr(generic, cordic_body, end_loop );

    builder.SetInsertPoint(cordic_body);

    // x>0
    generic= builder.CreateICmpSGE(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value),  zero);
    
    // d = 1 if x >= t else -1
    builder.CreateStore(builder.CreateSelect(
        builder.CreateICmpSGE(builder.CreateLoad(getElementTypeFromValuePointer(t.value), t.value), builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value)), 
        builder.CreateSelect(generic, one.value, zero),         
        builder.CreateSelect(generic, zero, one.value))
    ,d.value);

    
    //theta = theta + 2*d*math.atan(math.pow(2,-i))
    auto tmp = builder.CreateGEP(getElementTypeFromValuePointer(arctan_g), arctan_g, {zero,builder.CreateLoad(getElementTypeFromValuePointer(i), i)});
    generic = builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(tmp),tmp), ConstantInt::get(Type::getInt32Ty(cont), 1));
    generic = builder.CreateSelect(builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero),builder.CreateSub(zero, generic), generic);
    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value), generic ), theta.value);


    //    t = t + t*math.pow(2,-2*i) 
    generic = builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(i), i), ConstantInt::get(Type::getInt32Ty(cont), 1));
    builder.CreateStore(
        builder.CreateAdd(
            builder.CreateLoad(getElementTypeFromValuePointer(t.value), t.value), 
            builder.CreateAShr( builder.CreateLoad(getElementTypeFromValuePointer(t.value), t.value), generic
            )), t.value);


    //    x1= x-d*math.pow(2,-i)*y
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value), generic), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value), generic)), 
        x1.value);

    
    // y1= d*math.pow(2,-i)*x + y
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(x.value), x.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value), generic), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(y.value), y.value), generic)), 
        y1.value);


    //    x= x1-d*math.pow(2,-i)*y1
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(y1.value), y1.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(x1.value), x1.value), generic), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(x1.value), x1.value), generic)), 
        x.value);
    
    // y= d*math.pow(2,-i)*x1 + y1
    generic = builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(x1.value), x1.value), builder.CreateLoad(getElementTypeFromValuePointer(i), i));
    builder.CreateStore(
        builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateLoad(getElementTypeFromValuePointer(d.value), d.value), zero), 
                builder.CreateSub(builder.CreateLoad(getElementTypeFromValuePointer(y1.value), y1.value), generic), 
                builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(y1.value), y1.value),generic)), 
        y.value);

    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(getElementTypeFromValuePointer(i), i), ConstantInt::get(Type::getInt32Ty(cont), 1)), i );

    builder.CreateBr(loop_entry);

    builder.SetInsertPoint(end_loop);

    if(fxpret.scalarFracBitsAmt() > arg_size - 3){
        builder.CreateStore(builder.CreateAShr(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value), fxpret.scalarFracBitsAmt() - arg_size + 3), theta.value);
    }else if (fxpret.scalarFracBitsAmt() < arg_size - 3){
        builder.CreateStore(builder.CreateShl(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value), -fxpret.scalarFracBitsAmt() + arg_size - 3), theta.value);
    }
    builder.CreateRet(builder.CreateLoad(getElementTypeFromValuePointer(theta.value), theta.value));


  }