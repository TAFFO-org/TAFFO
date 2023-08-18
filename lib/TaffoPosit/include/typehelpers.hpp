/**
 * Copyright (C) 2017-2023 Emanuele Ruffaldi, Federico Rossi
 * 
 * Distributed under the terms of the BSD 3-Clause License.  
 * 
 * (See accompanying file LICENSE)
 * 
 * --
 */

#pragma once
#include <inttypes.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "bithippop.hpp"
#define HAS128T



/// returns the largest type between two
template <class A, class B>
using largest_type =
    typename std::conditional<sizeof(A) >= sizeof(B), A, B>::type;

/// Given size in bits returns the integer with given size
/// Equivalent to:
/// http://www.boost.org/doc/libs/1_48_0/libs/integer/doc/html/boost_integer/integer.html#boost_integer.integer.sized
namespace detail_least {

template <int Category>
struct int_least_helper {};
#ifdef HAS128T
template <>
struct int_least_helper<1> {
  __extension__ typedef __int128 signed_type;
  __extension__ typedef unsigned __int128 unsigned_type;
};
#endif
template <>
struct int_least_helper<2> {
  typedef int64_t signed_type;
  typedef uint64_t unsigned_type;
};
template <>
struct int_least_helper<3> {
  typedef int32_t signed_type;
  typedef uint32_t unsigned_type;
};
template <>
struct int_least_helper<4> {
  typedef int16_t signed_type;
  typedef uint16_t unsigned_type;
};
template <>
struct int_least_helper<5> {
  typedef int8_t signed_type;
  typedef uint8_t unsigned_type;
};

}  // namespace detail_least

/// Given size in bits returns the integer with given size
template <unsigned int N>
struct int_least_bits
    : public detail_least::int_least_helper<
          ((N) <= 8) + ((N) <= 16) + ((N) <= 32) + ((N) <= 64) + ((N) <= 128)> {
};

/// Helper for avoiding the fact that int8_t and uint8_t are printerd as chars
/// in iostream
template <class T>
struct printableinttype {
  using type = T;
};

template <class T>
struct printableinttype<const T> {
  using typex = typename printableinttype<T>::type;
  using type = const typex;
};

template <>
struct printableinttype<uint8_t> {
  using type = uint16_t;
};

template <>
struct printableinttype<int8_t> {
  using type = int16_t;
};

/// next integer type in size: signed and unsigned
template <class T>
struct nextinttype {
  struct error_type_is_too_large_or_missing_128bit_integer {};
  using type = error_type_is_too_large_or_missing_128bit_integer;
};

template <>
struct nextinttype<uint64_t> {
  __extension__ using type = unsigned __int128;
};

template <>
struct nextinttype<uint32_t> {
  using type = uint64_t;
};

template <>
struct nextinttype<uint16_t> {
  using type = uint32_t;
};

template <>
struct nextinttype<uint8_t> {
  using type = uint16_t;
};

template <>
struct nextinttype<int64_t>
{
	__extension__ using type = __int128;
};

template <>
struct nextinttype<int32_t> {
  using type = int64_t;
};

template <>
struct nextinttype<int16_t> {
  using type = int32_t;
};

template <>
struct nextinttype<int8_t> {
  using type = int16_t;
};

template <class T>
struct nextintop {
  static inline typename nextinttype<T>::type extramul(T a, T b) {
    return ((typename nextinttype<T>::type)a) * b;
  }

  static inline typename nextinttype<T>::type extradiv(
      typename nextinttype<T>::type a, T b) {
    return (a / b);
  }
};

#ifndef FPGAHLS
#if 0
/**
 * @brief reinterprets i32 to f32 using union
 * Not available on FPGA
 *
 * @param i
 * @return float
 */
inline float uint32_to_float(uint32_t i) {
  union {
    float f;
    uint32_t i;
  } x;
  x.i = i;
  return x.f;
}
#endif
#endif

namespace posit {
	namespace math {
		namespace detail {
			// generic abs algorithm
			template <typename T>
			constexpr auto abs(const T& value) -> T {
				return (T{} < value) ? value : -value;
			}
		}  // namespace detail

		template <typename T>
		constexpr auto abs(const T& value) -> T {
		  using std::abs;
		  using detail::abs;
		  return abs(value);
		}
	}  // namespace math
}  // namespace posit

