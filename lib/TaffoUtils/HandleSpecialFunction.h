
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>
#include <string>
#include <string_view>


#ifndef TAFFO_HANDLED_FUNCTION
#define TAFFO_HANDLED_FUNCTION

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "taffo-utils"
#endif

using namespace llvm;


namespace taffo
{

template <typename U, typename T>
inline bool start_with(const U &name, T &&prefix) noexcept
{
  return name.find(std::forward<T>(prefix)) == 0;
}

class HandledSpecialFunction
{

public:
  /**Check if a function is handled
   * @param f the function to check*/
  static bool is_handled(const llvm::Function *f)
  {
    auto match = find_inside(f);
    auto &list_of_handled = get_handled_function()->handledFunction;
    auto founded = match != std::end(list_of_handled);
    LLVM_DEBUG(if (founded) llvm::dbgs() << f->getName() << " is handled in a special way\n";);
    return founded;
  }

  //Get the match if it is present otherwise it panics
  //Must be used after is_handled
  static const std::string *getMatch(const llvm::Function *f)
  {
    LLVM_DEBUG(
        assert(is_handled(f) && "Not handled"););
    return find_inside(f);
  }

  static const llvm::SmallVector<std::string, 3U> &
  handledFunctions()
  {
    auto handled = get_handled_function();
    return handled->handledFunction;
  }


  // Demangle function return a demangled name or the same string
  static std::string demangle(const std::string &MangledName)
  {
    char *Demangled;
    if (is_itanium_encoding(MangledName))
      Demangled = llvm::itaniumDemangle(MangledName.c_str(), nullptr, nullptr, nullptr);
    else
      return MangledName;

    if (!Demangled)
      return MangledName;

    std::string Ret = Demangled;
    std::free(Demangled);
    return Ret;
  }

private:
  /**Constructor the list of handled function
   */
  bool enabled(bool b)
  {
    return b;
  }

  static const std::string *find_inside(const llvm::Function *f)
  {
    llvm::StringRef old_fName = f->getName();
    std::string fName(demangle(old_fName.str()));
    auto handled = get_handled_function();
    auto &list_of_handled = handled->handledFunction;
    // check if function name is present
    auto founded = std::find_if(std::cbegin(list_of_handled), std::cend(list_of_handled),
                                [&fName](const std::string &toComp) {
                                  return fName.find(toComp) == 0;
                                });
    return founded;
  }

  static bool
  is_itanium_encoding(const std::string &MangledName)
  {
    size_t Pos = MangledName.find_first_not_of('_');
    // A valid Itanium encoding requires 1-4 leading underscores, followed by 'Z'.
    return Pos > 0 && Pos <= 4 && MangledName[Pos] == 'Z';
  }


  HandledSpecialFunction()
  {
    handledFunction.emplace_back("asin");
    handledFunction.emplace_back("acos");
    handledFunction.emplace_back("sin");
    handledFunction.emplace_back("cos");
  }

  /*get an instance of HandledFunction
   * singleton implementation
   * used only to check what special function are supported*/
  static HandledSpecialFunction *get_handled_function()
  {
    static HandledSpecialFunction *instance;
    if (instance != nullptr) {
      return instance;
    }

    instance = new HandledSpecialFunction();
    return instance;
  }

  llvm::SmallVector<std::string, 3> handledFunction;
};

} // namespace taffo

#endif
