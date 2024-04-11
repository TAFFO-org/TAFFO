#include "CreateSpecialFunction.h"
#include "HandleSpecialFunction.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <string>

namespace taffo
{


// Create a new function to implements special known function
llvm::Function *CreateSpecialFunction::create(flttofix::FloatToFixed *f_t_f, llvm::CallSite *call, bool &alreadyHandledNewF)
{
  LLVM_DEBUG(
      assert(taffo::HandledSpecialFunction::is_handled(call->getCalledFunction()) == true && "Trying to create not supported function"););
  auto instance = get_instance(f_t_f);
  return instance->handle(call, alreadyHandledNewF);
}


CreateSpecialFunction *CreateSpecialFunction::get_instance(flttofix::FloatToFixed *f_t_f)
{
  static CreateSpecialFunction *instance = nullptr;
  if (instance == nullptr) {
    instance = new CreateSpecialFunction(f_t_f);
  }
  return instance;
}

// Create a new function to implements special known function
//  it uses a hashmap to store <string, pointer_to_creator>
llvm::Function *CreateSpecialFunction::handle(llvm::CallSite *call, bool &alreadyHandledNewF)
{

  LLVM_DEBUG(llvm::dbgs() << "####" << __func__ << "####");
  LLVM_DEBUG(llvm::dbgs() << "\t handle function " << call->getCalledFunction()->getName() << "\n");


  auto old_f = call->getCalledFunction();
  auto old_func_type = old_f->getFunctionType();
  auto old_ret_type = old_func_type->getReturnType();
  flttofix::FixedPointType *old_ret_fxpt = nullptr;
  bool old_ret_found = false;


  if (float_to_fixed->hasInfo(call)) {

    old_ret_fxpt = &float_to_fixed->fixPType(call);
    old_ret_found = true;
    LLVM_DEBUG(llvm::dbgs() << "\tFound info of ret  " << *old_ret_fxpt << "\n");

  } else {
    LLVM_DEBUG(llvm::dbgs() << "\tNot info of ret \n");
    old_ret_found = false;
  }


  std::string prefix_name = *HandledSpecialFunction::getMatch(old_f);

  // get new return type

  llvm::Type *new_ret_type = old_ret_type;

  if (old_ret_found) {

    if (old_ret_fxpt->isFixedPoint()) {
      prefix_name = prefix_name + "_" + (old_ret_fxpt->scalarIsSigned() ? "i" : "u") + std::to_string(old_ret_fxpt->scalarBitsAmt()) + "_" + std::to_string(old_ret_fxpt->scalarFracBitsAmt());
    } else {
      prefix_name += "_nfxpt";
    }

    new_ret_type = float_to_fixed->getLLVMFixedPointTypeForFloatValue(call);
  }


  // get new Args type

  std::vector<llvm::Type *> new_type_args;
  std::vector<flttofix::FixedPointType *> old_args_fxpt;
  LLVM_DEBUG(llvm::dbgs() << "\tArgs: \n");
  for (auto arg = old_f->arg_begin(); arg != old_f->arg_end(); arg++) {
    Value *v = dyn_cast<Value>(arg);
    Type *newTy;
    if (float_to_fixed->hasInfo(v)) {
      auto tmp = &float_to_fixed->fixPType(call);
      LLVM_DEBUG(llvm::dbgs() << "\t\tFound fxpt" << tmp << " \n");
      old_args_fxpt.push_back(tmp);

      newTy = float_to_fixed->getLLVMFixedPointTypeForFloatValue(v);

    } else {
      LLVM_DEBUG(llvm::dbgs() << "\t\tNot found fxpt"
                              << " \n");
      old_args_fxpt.push_back(nullptr);
      newTy = v->getType();
    }
    new_type_args.push_back(newTy);
  }

  FunctionType *new_func_type = FunctionType::get(
      new_ret_type,
      new_type_args, old_f->isVarArg());


  // See if we have a definition for the specified function already.
  GlobalValue *F = old_f->getParent()->getNamedValue(prefix_name);
  if (!F) {


    LLVM_DEBUG({
      dbgs() << "creating special function " << prefix_name << "_"
             << " with types ";
      for (auto arg_t : new_type_args) {
        dbgs() << "(" << arg_t << ") ";
      }
      dbgs() << "\n";
    });

    // Nope, add it
    Function *new_f = Function::Create(new_func_type, old_f->getLinkage(), prefix_name, old_f->getParent());
    new_f->setLinkage(llvm::GlobalValue::InternalLinkage);
    // create body

    auto old_info = OldInfo{
        old_f,
        old_ret_fxpt,
        old_args_fxpt};


    auto new_info = NewInfo{new_f};
    return dispatch[*HandledSpecialFunction::getMatch(old_f)](old_info, new_info);
  } else if (auto new_f = dyn_cast<llvm::Function>(F)) {
    LLVM_DEBUG({
      dbgs() << "Alredy exist special function " << prefix_name << "_"
             << "fixp"
             << " with types ";
      for (auto arg_t : new_type_args) {
        dbgs() << "(" << arg_t << ") ";
      }
      dbgs() << "\n";
    });
    alreadyHandledNewF = true;
    // alredy created
    return new_f;
  }

  llvm_unreachable("Found same name but is not a function");
}


CreateSpecialFunction::CreateSpecialFunction(flttofix::FloatToFixed *f_t_f)
{
  using namespace std::placeholders;

  this->float_to_fixed = f_t_f;

  dispatch.insert({"__dev-sin", [this](OldInfo &O, NewInfo &N) { return this->sinHandler(O, N); }});
  dispatch.insert({"__dev-cos", [this](OldInfo &O, NewInfo &N) { return this->cosHandler(O, N); }});

  dispatch.insert({"__dev-asin", [this](OldInfo &O, NewInfo &N) { return this->asinHandler(O, N); }});
  dispatch.insert({"__dev-acos", [this](OldInfo &O, NewInfo &N) { return this->acosHandler(O, N); }});

  // dispatch.insert({"sin", [this](OldInfo &O, NewInfo &N) { return this->sinHandler(O, N); }});
  // dispatch.insert({"cos", [this](OldInfo &O, NewInfo &N) { return this->cosHandler(O, N); }});

  dispatch.insert({"asin", [this](OldInfo &O, NewInfo &N) { return this->asinHandler(O, N); }});
  dispatch.insert({"acos", [this](OldInfo &O, NewInfo &N) { return this->acosHandler(O, N); }});
}

}; // namespace taffo