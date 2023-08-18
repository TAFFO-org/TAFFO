/**
 * Copyright (C) 2017-2023 Emanuele Ruffaldi, Federico Rossi
 * 
 * Distributed under the terms of the BSD 3-Clause License.  
 * 
 * (See accompanying file LICENSE)
 * 
 * --
 */

///@{ 
/// @name std member function overloads
namespace posit {
    template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	class Posit;
}

namespace std
{



	template <class PType,class PAccumType>
	inline CONSTEXPR14 PAccumType fma(PType x,
								  PType y,
								  PAccumType z) 
	{
		return z + x.template to_posit<PAccumType>() * y.template to_posit<PAccumType>();
	}	


	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	inline CONSTEXPR14 posit::Posit<T,totalbits,esbits,FT,positspec> min(posit::Posit<T,totalbits,esbits,FT,positspec> a, posit::Posit<T,totalbits,esbits,FT,positspec> b)
	{
		return a <=  b ? a : b;
	}

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	inline CONSTEXPR14 posit::Posit<T,totalbits,esbits,FT,positspec> max(posit::Posit<T,totalbits,esbits,FT,positspec> a, posit::Posit<T,totalbits,esbits,FT,positspec> b)
	{
		return a >= b ? a : b;
	}

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	inline CONSTEXPR14 posit::Posit<T,totalbits,esbits,FT,positspec> abs(posit::Posit<T,totalbits,esbits,FT,positspec> a)
	{
		return posit::Posit<T,totalbits,esbits,FT,positspec>::from_sraw(pcabs(a.v));;
	}

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> exp(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::exp((float)a);
	}

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> sqrt(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::sqrt((float)a);
	}	

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> log(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::log((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> log1p(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::log1p((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> log10(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::log1p((float)a);
	}	


	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> log2(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::log1p((float)a);
	}					

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> sin(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::sin((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> cos(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::cos((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> tan(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::tan((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> asin(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::asin((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> acos(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::acos((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> atan(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::atan((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> atan2(const posit::Posit<T,totalbits,esbits,FT,positspec>& a,
																  const posit::Posit<T,totalbits,esbits,FT,positspec>& b)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::atan2((float)a,(float)b);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> sinh(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::sinh((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> cosh(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::cosh((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> tanh(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::tanh((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> asinh(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::asinh((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> acosh(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::acosh((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> atanh(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::atanh((float)a);
	}		

	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> isfinite(const posit::Posit<T,totalbits,esbits,FT,positspec>& a)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::isfinite((float)a);
	}	



	template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec>
	constexpr posit::Posit<T,totalbits,esbits,FT,positspec> pow(const posit::Posit<T,totalbits,esbits,FT,positspec>& a,
																const posit::Posit<T,totalbits,esbits,FT,positspec>& b)
	{
		using PP = posit::Posit<T,totalbits,esbits,FT,positspec>;
		return (PP)std::pow((float)a,(float)b);
	}				




	/**
	 * @brief Specialization of std::numerical_limits for Posit
	 * 
	 * @tparam B as T of Posit
	 * @tparam totalbits as all bits of Posit
	 * @tparam esbits as exponent bits of Posit
	 * @tparam FT as backend type
	 * @tparam positspec as detailed specification
	 */
	template <class B,int totalbits, int esbits, class FT, posit::PositSpec positspec> class numeric_limits<posit::Posit<B,totalbits,esbits,FT,positspec> > {
	public:
	  using T=posit::Posit<B,totalbits,esbits,FT,positspec>;
	  using PT=typename T::PT;
	  static constexpr bool is_specialized = true;
	  static constexpr T min() noexcept { return T::min(); }
	  static constexpr T max() noexcept { return T::max(); }
	  static constexpr T lowest() noexcept { return T::lowest	(); }
	  static constexpr int  digits10 = ((totalbits-3)*30000)/100000;  // *log10(2)
	  static constexpr bool is_signed = true;
	  static constexpr bool is_integer = false;
	  static constexpr bool is_exact = false;
	  static constexpr int radix = 2;
	  static constexpr T epsilon() noexcept { return T::one().next()-T::one(); }
	
	  // this is also the maximum integer
	  static constexpr int  min_exponent = PT::minexponent();
	  static constexpr int  max_exponent = PT::maxexponent();

	  static constexpr bool has_infinity = true;
	  static constexpr bool has_quiet_NaN = true;
	  static constexpr bool has_signaling_NaN = false;
	  static constexpr bool has_denorm_loss = false;
	  static constexpr T infinity() noexcept { return T::infinity(); }
	  static constexpr T quiet_NaN() noexcept { return T::nan(); }
	  static constexpr T denorm_min() noexcept { return T::min(); }

	  static constexpr bool is_iec559 = false;
	  static constexpr bool is_bounded = false;
	  static constexpr bool is_modulo = false;

	  static constexpr bool traps = false;
	  static constexpr bool tinyness_before = false;
	};

	/* FOR SET-LIKE */
	template <class B,int totalbits, int esbits, class FT, posit::PositSpec positspec>	
	struct hash<posit::Posit<B,totalbits,esbits,FT,positspec>>
    {
        std::size_t operator()(const posit::Posit<B,totalbits,esbits,FT,positspec>& p) const noexcept
        {
            return p.v;
        }
    };

}
///@}
