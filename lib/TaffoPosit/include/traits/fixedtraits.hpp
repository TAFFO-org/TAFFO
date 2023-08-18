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


/**
 * @brief provides information about the charactderistics of a fixed point type
 * @tparam T holding type (enough for size N)
 * @tparam N total number of bits 
 * @tparam F fraction size (less than N)
 *
 * @todo support other notation in which the integral part is the number of bits shift (pos/neg)
 */
template <class T, int N, int F = N/2>
struct fixedtrait
{
    static_assert(sizeof(T)*8 >= N,"fixedtrait holding type is too small");
    static_assert(N > 0,"fixedtrait total bits should be positive");
    static_assert(F <= N && F >= 0,"fraction bits should be less than N and not negative");
    //static_assert(std::is_integral<T>::value && std::is_signed<T>::value,"only for signed integrals");
	using value_t = T;
	static constexpr int totalbits = N;
	static constexpr int fraction_bits = F;
};

