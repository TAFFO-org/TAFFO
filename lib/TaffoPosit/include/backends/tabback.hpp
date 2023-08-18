/**
 * Copyright (C) 2017-2023 Emanuele Ruffaldi, Federico Rossi
 * 
 * Distributed under the terms of the BSD 3-Clause License.  
 * 
 * (See accompanying file LICENSE)
 * 
 * --
 */

#include <type_traits>

namespace posit {

    template <class T>
	struct is_posit_backend;

	/// @brief Posit table trait struct: describes characteristic of a tabulated set of operations
	/// @tparam T holder type
	/// @tparam n number of Posit bits
	/// @tparam e number of exponent bits
	/// @tparam log True if contains logarithmic alternativ to mul and div tables
	template <class T, int n, int e, bool log>
	struct PositTableTrait {
		using type = T;
		using nbits = std::integral_constant<int,n>;
		using esbits = std::integral_constant<int,e>;
		using isLogtab = std::integral_constant<bool,log>;
	};

	/// @brief Tabulated Backend for posit
	/// @tparam PositEmu Posit type used for full-software emulation as a fallback
	/// @tparam PTable Posit Table class containing the tables used for the operations as static constexpr members
	/// @tparam PTableTrait Posit Table trait 
	template <class PTableTrait, class PositEmu, class PTable>
	struct TabulatedBackend: public HwBaseBackend
	{
		struct single_tag{};
		using T = typename PTableTrait::type;
        constexpr static T indexMask = (1<<PTableTrait::nbits::value)-1;
		TabulatedBackend() {}
		TabulatedBackend(single_tag, T x): v(x) {}

		explicit TabulatedBackend(int x) {v=PositEmu(x).v;}
		explicit TabulatedBackend(long x) {v=PositEmu(x).v;}
		explicit TabulatedBackend(float x) {v=PositEmu(x).v;}
		explicit TabulatedBackend(double x) {v=PositEmu(x).v;}

		constexpr operator float () const {return (float)PositEmu::from_sraw(v);}
		constexpr operator double () const {return (double)PositEmu::from_sraw(v);}
		constexpr operator int () const {return (int)PositEmu::from_sraw(v);}
		constexpr operator long () const {return (long)PositEmu::from_sraw(v);}

	 	TabulatedBackend operator + (TabulatedBackend o) const { 
            return TabulatedBackend{{},PTable::add[v & indexMask][o.v & indexMask]};
        } 
		TabulatedBackend operator * (TabulatedBackend o) const { 
			if constexpr (PTableTrait::isLogtab::value) {
				T idxa = v & indexMask, idxb = o.v & indexMask;
				T sign = (idxa >> (PTableTrait::nbits::value-1)) ^ (idxb >> (PTableTrait::nbits::value-1));
				T logA = PTable::log[idxa], logB = PTable::log[idxb];
				T logAB = PTable::add[logA & indexMask][logB & indexMask];
				T expAB = PTable::exp[logAB];
				if(sign) expAB = - expAB;
				return TabulatedBackend{{},expAB};
			} else {
            	return TabulatedBackend{{},PTable::mul[v & indexMask][o.v & indexMask]};
			}
        } 
	 	TabulatedBackend operator / (TabulatedBackend o) const { 
			if constexpr (PTableTrait::isLogtab::value) {
				T idxa = v & indexMask, idxb = o.v & indexMask;
				T sign = (idxa >> (PTableTrait::nbits::value-1)) ^ (idxb >> (PTableTrait::nbits::value-1));
				T logA = PTable::log[idxa], logB = -PTable::log[idxb];
				T logAB = PTable::add[logA & indexMask][logB & indexMask];
				T expAB = PTable::exp[logAB];
				if(sign) expAB = - expAB;
				return TabulatedBackend{{},expAB};
			} else
            	return TabulatedBackend{{},PTable::div[v & indexMask][o.v & indexMask]};
        } 
		T v;

	};

	template <class T, class PositEmu, class PTable>
	struct is_posit_backend<TabulatedBackend<T,PositEmu,PTable> >: public std::true_type
	{
	};
    
}