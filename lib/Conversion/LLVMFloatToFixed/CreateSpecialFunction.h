
#pragma once

#include "CallSiteVersions.h"
#include "LLVMFloatToFixedPass.h"
#include <functional>
#include <llvm/ADT/StringMap.h>


namespace taffo
{


class CreateSpecialFunction
{
public:
  static llvm::Function *create(flttofix::FloatToFixed *, llvm::CallSite *call, bool &alreadyHandledNewF);

private:
  struct OldInfo {
    llvm::Function *old_f;
    flttofix::FixedPointType *old_ret_fxpt;
    std::vector<flttofix::FixedPointType *> old_args_fxpt;
  };
  struct NewInfo {
    llvm::Function *new_f;
  };
  CreateSpecialFunction(flttofix::FloatToFixed *);
  static CreateSpecialFunction *get_instance(flttofix::FloatToFixed *);
  llvm::Function *handle(llvm::CallSite *, bool &);
  llvm::StringMap<std::function<llvm::Function *(OldInfo &, NewInfo &)>> dispatch;
  flttofix::FloatToFixed *float_to_fixed;
  //Handler functions
  llvm::Function *sinHandler(OldInfo &, NewInfo &);
  llvm::Function *cosHandler(OldInfo &, NewInfo &);
};

} // namespace taffo