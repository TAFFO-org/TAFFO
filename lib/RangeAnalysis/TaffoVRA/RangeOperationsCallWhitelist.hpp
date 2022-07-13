#ifndef TAFFO_RANGE_OPERATIONS_CALL_WHITELIST_HPP
#define TAFFO_RANGE_OPERATIONS_CALL_WHITELIST_HPP

#include "Range.hpp"

#include <list>
#include <map>

#define DEBUG_TYPE "taffo-vra"

namespace taffo
{

#define CMATH_WHITELIST_FUN(BASE_NAME, POINTER) \
  {BASE_NAME, POINTER},                         \
      {BASE_NAME "f", POINTER},                 \
      {BASE_NAME "l", POINTER},                 \
      {"llvm." BASE_NAME ".f32", POINTER},      \
  {                                             \
    "llvm." BASE_NAME ".f64", POINTER           \
  }

using map_value_t = range_ptr_t (*)(const std::list<range_ptr_t> &);
extern const std::map<const std::string, map_value_t> functionWhiteList;
}; // namespace taffo

#undef DEBUG_TYPE

#endif /* end of include guard: TAFFO_RANGE_OPERATIONS_CALL_WHITELIST_HPP */
