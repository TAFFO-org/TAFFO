/**
 * Copyright (C) 2017-2023 Emanuele Ruffaldi, Federico Rossi
 * 
 * Distributed under the terms of the BSD 3-Clause License.  
 * 
 * (See accompanying file LICENSE)
 * 
 * --
 */
namespace posit {

    template <class T>
	struct is_posit_backend;

	/// @brief Demo Backend for ISA-extended Posit backend
	/// @tparam T holder type
	/// @tparam PositEmu Posit type used for full-software emulation as fallback
	template <class T,class PositEmu>
	struct BackendXPosit: public HwBaseBackend
	{
		struct single_tag{};

		BackendXPosit() {}
		BackendXPosit(single_tag, T x): v(x) {}

		explicit BackendXPosit(int x) {v=PositEmu(x).v;}
		explicit BackendXPosit(long x) {v=PositEmu(x).v;}
		explicit BackendXPosit(float x) {v=PositEmu(x).v;}
		explicit BackendXPosit(double x) {v=PositEmu(x).v;}

		constexpr operator float () const {return (float)PositEmu::from_sraw(v);}
		constexpr operator double () const {return (double)PositEmu::from_sraw(v);}
		constexpr operator int () const {return (int)PositEmu::from_sraw(v);}
		constexpr operator long () const {return (long)PositEmu::from_sraw(v);}

	 	BackendXPosit operator + (BackendXPosit o) const { return BackendXPosit{{},v+o.v}; }
		BackendXPosit operator * (BackendXPosit o) const { return BackendXPosit{{},v*o.v}; }
	 	BackendXPosit operator - (BackendXPosit o) const { return BackendXPosit{{},v-o.v}; }
	 	BackendXPosit operator / (BackendXPosit o) const { return BackendXPosit{{},v/o.v}; }
		T v;

	};

    template <class T, class PositEmu>
	struct is_posit_backend<BackendXPosit<T,PositEmu> >: public std::true_type
	{
	};
    
}