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
#include <backends/softback.hpp>
#include <traits/posittraits.hpp>
#include <backends/fixed.hpp>
#include <backends/float.hpp>
#include <backends/hwbaseback.hpp>
#include <backends/xpositback.hpp>
#include <backends/unpacked.hpp>
#include <backends/ptaback.hpp>
#include <backends/tabback.hpp>
#include <interface/std/std.hpp>
#include <complex>
#include <iostream>
#include <typeinfo>

#ifdef POSIT_VALUE_LOGGER
#include "interface/logger/logger.hpp"
#endif





/**
 * @brief The posit namespace provies wrapped access to cppPosit
 */
namespace posit
{

	/**
	 * @brief The posit::math namespace provides general math functionalities 
	 */
	namespace math
	{

		/**
		 * @brief computes the dot product between posit values vectors	 
		 * The accumulating type is anything that can be converted to/from the posit:
		 *   - floating point: float, double
		 *	 - another posit
		 *	 - fixed point
		 * 	 - unpacked posit (AKA soft float)
		 *
		 * @todo decide if resulting type is Posit or another one
		 *
		 * @tparam It1 type of first iterator
		 * @tparam It2 type of second iterator
		 * @tparam A is the product type, default is self
		 * @tparam B is the accumulating type, default is self
		 * @param begin1 first iterator
		 * @param begin2 second iterator
		 * @param N length of vectors
		 */
	    template <class It1, class It2, class A, class B>
	    auto dot(It1 begin1, It2 begin2, int N, A tprod, B tsum) -> typename std::remove_reference<decltype(*begin1)>::type
	    {
	    	using PP = typename std::remove_reference<decltype(*begin1)>::type;
	    	if(N < 1)
	    		return PP(0);
	    	else
	    	{
		    	B v0 = static_cast<A>((*begin1++))*static_cast<A>((*begin2++)); // first
		    	for(int i = 1; i < N; i++)
			    	v0 += static_cast<A>((*begin1++))*static_cast<A>((*begin2++));
		    	return static_cast<PP>(v0);
		    }
	    }

	    template <class It1, class It2>
	    auto dot(It1 begin1, It2 begin2, int N) -> typename std::remove_reference<decltype(*begin1)>::type
	    {
	    	using PP = typename std::remove_reference<decltype(*begin1)>::type;
	    	return dot<It1,It2,PP,PP>(begin1,begin2,N,PP(),PP());
	    }


		/**
		 * @brief computes the dot product between posit values vectors using Kahan summation algorithm	 
		 *
		 * @tparam It1 type of first iterator
		 * @tparam It2 type of second iterator
		 * @tparam A is the product type
		 * @tparam B is the accumulating type
		 * @tparam C is the running compensation type
		 * @param begin1 first iterator
		 * @param begin2 second iterator
		 * @param N length of vectors
		 */
	    template <class It1, class It2, class A, class B, class C>
	    auto comp_dot(It1 begin1, It2 begin2, int N,A tprod, B tsum, C tcomp) -> typename std::remove_reference<decltype(*begin1)>::type
	    {
	    	using PP = typename std::remove_reference<decltype(*begin1)>::type;
	    	if(N < 1)
	    		return PP(0); 
	    	else
	    	{
		    	B vsum(static_cast<A>(*begin1++)*static_cast<A>(*begin2++)); // first
		    	C c(0);
		    	for(int i = 1; i < N; i++)
		    	{
		    		B iv = static_cast<A>(*begin1++)*static_cast<A>(*begin2++);
		    		B t = vsum + iv; // large
		    		if (math::abs(vsum) >= math::abs(iv))
		    			c += (vsum-t)+iv;
		    		else
		    			c += (iv-t)+vsum;
		    		vsum = t;
		    	}
		    	return static_cast<PP>(vsum);
		    }
	    }

		/**
		 * @brief Fused multiply and accumulation using a QuireT type accumulator
		 */
	    template <class It1, class It2, class AccumT>
	    auto fma_dot(It1 begin1, It2 begin2, int N, AccumT& accum) -> typename std::remove_reference<decltype(*begin1)>::type
	    {
	    	using PP = typename std::remove_reference<decltype(*begin1)>::type;
	    	if(N < 1)
	    		return PP(0);
	    	else
	    	{
		    	accum = std::fma((*begin1++), (*begin2++), accum); // first
		    	for(int i = 1; i < N; i++)
			    	accum = std::fma((*begin1++), (*begin2++), accum);
		    	return PP(double(accum));
		    }
	    }		
	}
	






	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	class Posit;

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 auto unpack_posit(const Posit<T,totalbits,esbits,FT,positspec> & p) -> typename Posit<T,totalbits,esbits,FT,positspec>::BackendT ;

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> pack_posit(const typename Posit<T,totalbits,esbits,FT,positspec>::BackendT & x);




	template <class T>
	struct is_posit_backend: public std::false_type
	{
	};



	template <class PT, class Target>
	struct PositPacker
	{

	};

	


	/**
	 * @brief Posit value type
	 * Stores the posit in the LSB part of the holding type T as signed integer
	 *
	 * Operations are implemented at 4 levels:
	 * - Level 1: direct manipulation of bit representation 
	 * - Level 2: decoding of the posit in its componentds (s,k,E,F)
	 * - Level 3: decoding and manipulation of the posit in full exponent form (s,X,f)
	 * - Level 4: using the posit backend (softfloat by Unpacked, FPU by BackendFloat, fixed point by BackendFixed)
	 *
	 * \tparam T the holding type that has to be signed due to complement 2 sign method
	 * \tparam totalbits the significant bits of posit stored in T right aligned. Due to the 2 complement scheme the MSB bits are extension of the sign
	 * \tparam esbits the size of the exponent 
	 * \tparam FT the backend type that can be: unsigned integer for softposit, float/double for emulation, generic Backend, e.g. fixed point
	 * \tparam positspec specifies the number capabilities
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	class Posit
	{
	public:

	#ifdef POSIT_VALUE_LOGGER
		static Logger<double> _logger;
	#endif

		template <class VV>
		void logValue(VV v) 
		{
			#ifdef POSIT_VALUE_LOGGER
				_logger.logValue(v);
			#endif
		}

		using PT=PositTrait<T,totalbits,esbits,positspec>;
		
		/// special constant used internally for safely constructing the posit from raw
		struct DeepInit{}; 

	    enum { vtotalbits = totalbits, vesbits = esbits};

	    using value_t = T;

	    /// FT can be a Backend otherwise select it from float or integral
	    using BackendT=typename std::conditional<is_posit_backend<FT>::value,FT,typename std::conditional<std::is_floating_point<FT>::value,BackendFloat<FT>,Unpacked<FT,typename PT::exponenttype> >::type >::type;
	    using UnpackedPosit = posit::UnpackedPosit<PT>;
		using exponenttype = typename PT::exponenttype;
		using single_tag = typename BackendT::single_tag;
		T v; // index in the N2 space

		/// returns half (Level 1)
		CONSTEXPR14 Posit half() const;

		/// returns half (Level 1)
		CONSTEXPR14 Posit twice() const;

		/// transform from bit representation (Level 1) to unpacked low (Level 2)
		CONSTEXPR14 UnpackedPosit unpack_low() const;

		/// transform from unpacled low (Level 2) to bit representation (Level 1)
		static CONSTEXPR14 Posit pack_low(UnpackedPosit);

		/// transform from Level 2 to Level 4 
		static CONSTEXPR14 BackendT unpacked_low2full(UnpackedPosit x) { return PositPacker<PT,BackendT>::unpacked_to_backend(x); } 

		/// transform from Level 2 to Level 4
		static CONSTEXPR14 UnpackedPosit unpacked_full2low(BackendT x) { return PositPacker<PT,BackendT>::backend_to_unpacked(x); }

		/**
		 * @brief Convert the posit to another posit described by PT2
		 * \tparam PT2 is a posit type declaration 
		 * (e.g. posit::Posit<int8_t, 8 , 0, uint_fast32_t, posit::PositSpec::WithNan>)
		**/ 
		template <class PT2>
		PT2 to_posit() {
			if constexpr(PT2::vesbits == 0 && esbits == 0) {
					using NT = typename PT2::value_t;
					NT pt;
					if constexpr(PT2::vtotalbits > totalbits)
						pt = ((NT)v) << (PT2::vtotalbits - totalbits);
					else
						pt = NT(v >> (totalbits - PT2::vtotalbits));
					return PT2::from_sraw(pt);
			}
			return PT2(double(*this));
		}

		/**
		 * @brief Convert the posit to a backend fixed point for quire-like accumulation
		 * \tparam QuireT is the quire type (e.g. uint64_t)
		 */
		template <class QuireT>
		QuireT 	to_quire() {
			BackendT be = this->to_backend();
			QuireT q = (QuireT)(be.v);
			bool sign = (be.v < 0);
			q = (sign)? -q:q; // start with abs value
			q << (sizeof(q)-sizeof(be.v))*8;
			return (sign)?-q:q; // restore sign
		}


		/// Returns true if the two Posits are comparable
	 	constexpr bool comparable(const Posit & u) const { return !(PT::withnan && (is_nan()||u.is_nan())); }

	    static constexpr Posit ldexp(const Posit & u, int exp); // exponent product

	    /// empty constructor
		constexpr Posit() : v(0) {}

		/// Construct the Posit with its backend direct initialization
		CONSTEXPR14 explicit Posit(single_tag t, uint32_t p) { v = pack_posit<T,totalbits,esbits,FT,positspec>(BackendT(t,p)).v; }

	    /// Construct the Posit passing the holding type x
		CONSTEXPR14 explicit Posit(DeepInit, T x) : v(x) {} 

		/// Construct the Posit using its Unpacked from (s, R,E,F)
		CONSTEXPR14 explicit Posit(UnpackedPosit u) : v(pack_low(u).v) {} 

		/// Construct the Posit from its backend BackendT
		CONSTEXPR14 explicit Posit(BackendT u) : v(pack_posit<T,totalbits,esbits,FT,positspec>(u).v) {} 

		
		/// @brief Convert the integer i to Posit 
		/// @param i 32-bit signed integer value
		/// @return 
		CONSTEXPR14 Posit(int i): Posit(BackendT(i)) {}

		/// @brief Convert the long integer i to Posit 
		/// @param i 64-bit signed integer value
		/// @return 
		CONSTEXPR14 Posit(long int i): Posit(BackendT(i)) {}

		/// @brief Convert the long integer i to Posit 
		/// @param i 64-bit signed integer value
		/// @return 
		CONSTEXPR14 Posit(long long int i): Posit(BackendT(i)) {}

		/// @brief Convert the long integer i to Posit 
		/// @param i 32-bit unsigned integer value
		/// @return 
		CONSTEXPR14 Posit(unsigned int i): Posit(BackendT(i)) {}

		/// @brief Convert the long integer i to Posit 
		/// @param i 64-bit unsigned integer value
		/// @return 
		CONSTEXPR14 Posit(long unsigned int i): Posit(BackendT(i)) {}

		/// @brief Convert the long integer i to Posit 
		/// @param i 64-bit unsigned integer value
		/// @return 
		CONSTEXPR14 Posit(long long unsigned int i): Posit(BackendT(i)) {}
		CONSTEXPR14 Posit(std::complex<float> c): Posit(BackendT(c.real())) {}
		CONSTEXPR14 Posit(std::complex<double> c): Posit(BackendT(c.real())) {}

		/// @brief Convert the float number f to Posit 
		/// @param f 32-bit float value
		/// @return 
	    CONSTEXPR14 Posit(float f): Posit(BackendT(f)) {}

		/// @brief Convert the double number d to Posit 
		/// @param f 64-bit double value
		/// @return 
		CONSTEXPR14 Posit(double d): Posit(BackendT(d)) {}

		/// @brief Assignment operator with a float value
		/// @param f 32-bit float value
		/// @return Posit value converted from the float f
		CONSTEXPR14 Posit & operator= (float f) {
			v=Posit(f).v;
			return *this;		
		}

		/// @brief Assignment operator with a double value
		/// @param d 64-bit double value
		/// @return Posit value converted from the double d
		CONSTEXPR14 Posit & operator= (double d) {
			v = Posit(d).v;
			return *this;		
		}

		/// @brief Assignment operator with a integer value
		/// @param i 32-bit integer value
		/// @return Posit value converted from the integer i
		CONSTEXPR14 Posit & operator= (int i) {
			v = Posit(i).v;
			return *this;		
		}

		/// @brief Build the Posit from a floating point value expressed as the representing integer 
		/// @tparam XFT floating point trait type 
		/// @param x floating point representing integer
		/// @return Posit built from the floating point conversion 
		template <class XFT>
		static CONSTEXPR14 Posit from_floatval(typename XFT::holder_t x) { return BackendT::template make_floati<XFT>(x); }

		/// @brief Build Posit using its representing signed integer x 
		/// @param x signed integer representation of the Posit
		/// @return Posit built using its signed integer representation
		static CONSTEXPR14 Posit from_sraw(T x) { return Posit(DeepInit(), x); }

		/// @brief Build Posit using its representing unsigned integer x 
		/// @param x unsigned integer representation of the Posit
		/// @return Posit built using its unsigned integer representation		
		static CONSTEXPR14 Posit from_uraw(typename PT::POSIT_UTYPE x) { return Posit(DeepInit(), x); }

		/// @brief Returns Posit integer representation 
		/// @return Posit integer field v
		constexpr T to_sraw() const { return v;}

		/// @brief Returns Posit unsigned integer representation 
		/// @return Posit integer field v as unsigned type
		constexpr typename PT::POSIT_UTYPE to_uraw() const { return (typename PT::POSIT_UTYPE)v;}

		/// @brief Unpack the Posit to its backend type BackendT
		/// @return An instance of the BackendT initialized with the current posit value
		constexpr BackendT to_backend() const { 
			return unpack_posit<T,totalbits,esbits,FT,positspec>(*this);
		}

		/// @brief Compute the absolute value of the Posit (L1)
		/// @return New Posit containing the absolute value of the initial Posit
		constexpr Posit abs()  const { return from_sraw(pcabs(v));  }

		/// @brief Compute the sign opposition of the Posit (L1)
		/// @return New Posit with the opposite sign of the initial Posit
		constexpr Posit neg()  const { return from_sraw(-v); }; 

		constexpr Posit operator-() const { return neg(); } 

		/// @brief Digital negation operator that calls the Posit reciprocate function
		/// @return Returns the reciprocate of the current Posit
		constexpr Posit operator~() const { return reciprocate(); } 

	    /// returns true if the value is NaN
	    constexpr bool is_nan() const { return PT::withnan && v == PT::POSIT_NAN; } 
	    /// returns true if it is any infinity
		constexpr bool is_infinity() const { return (PT::withsignedinf && (v == PT::POSIT_PINF || v == PT::POSIT_NINF))||(PT::withsignedinf && v == PT::POSIT_INF); }
		/// returns true for zero
		constexpr bool is_zero() const { return v == 0; }
		/// returns true for one
		constexpr bool is_one() const { return v == PT::POSIT_ONE; }

		/// @brief Returns previous (clockwise in the Posit ring) regular Posit number
		constexpr Posit prev() const { return from_sraw(v > PT::POSIT_MAXPOS || v <= PT::POSIT_MINNEG ? v : v-1); }
		
		/// @brief Returns next (anti-clockwise in the Posit ring) regular Posit number
		constexpr Posit next() const { return from_sraw(v < PT::POSIT_MINNEG || v >= PT::POSIT_MAXPOS ? v : v+1); }

		/// @brief Check if the Posit is in in the unit interval [0,1]  (Level 1)
		/// @return True if Posit in the interval [0,1]
		constexpr bool is_unitary_range() const { return PT::is_sphere_range(v); }

		/// @brief Check if the Posit is in in the unit interval [-1,1]  (Level 1)
		/// @return True if Posit in the interval [-1,1]
		constexpr bool is_sphere_range() const { return PT::is_sphere_range(v); }

		/// @brief Sum and assignment operator
		/// @param a Right-hand-side Posit in the sum
		/// @return This instance of Posit after the sum
		Posit& operator+=(const Posit &a) { Posit r = *this+a; v = r.v; return *this; }

		/// @brief Subtraction and assignment operator
		/// @param a Right-hand-side Posit in the subtraction
		/// @return This instance of Posit after the subtraction
		Posit& operator-=(const Posit &a) { Posit r = *this-a; v = r.v; return *this; }

		/// @brief Multiplication and assignment operator
		/// @param a Right-hand-side Posit in the multiplication
		/// @return This instance of Posit after the multiplication		
		CONSTEXPR14 Posit & operator*= (const Posit & b)
		{
			*this = pack_posit<T,totalbits,esbits,FT,positspec>(to_backend()*b.to_backend());
			return *this;
		}

		/// @brief Division and assignment operator
		/// @param a Right-hand-side Posit in the division
		/// @return This instance of Posit after the division		
	    Posit & operator/= (const Posit & a) { auto x = *this / a; v = x.v; return *this; }

		/**
		 * @brief Returns 1/x as Level 1.
		 * Special Cases:
		 *	 zero should give positive infinity, if present, nan otherwise;
		 *	 nan  should be nan, if present;
		 *	 +inf/-inf should give zero, if present;
		 *	 before0,after0 should give infinity, if present.
		 *
		 * @return constexpr Posit 
		 */
		constexpr Posit reciprocate() const 
		{
			return is_infinity() ? zero() : (v == PT::POSIT_NAN ? nan() : (v == 0 ? pinf() : from_sraw(PT::reciprocate(v))));
		}	    


		static constexpr Posit zero() { return from_sraw(0); }
		static constexpr Posit inf() { return from_sraw(PT::POSIT_PINF); }
		static constexpr Posit pinf() { return from_sraw(PT::POSIT_PINF); }
		static constexpr Posit ninf() { return from_sraw(PT::POSIT_NINF); }
		static constexpr Posit max() { return from_sraw(PT::POSIT_MAXPOS); }
		static constexpr Posit min() { return from_sraw(PT::POSIT_AFTER0); }
		static constexpr Posit lowest() { return from_sraw(PT::POSIT_MINNEG); }

		static constexpr Posit nan() { return from_sraw(PT::POSIT_NAN); }
		static constexpr Posit infinity() { return from_sraw(PT::POSIT_PINF); }
		static constexpr Posit one() { return from_sraw(PT::POSIT_ONE); }
		static constexpr Posit two() { return from_sraw(PT::POSIT_TWO); }
		static constexpr Posit mone() { return from_sraw(PT::POSIT_MONE); }
		static constexpr Posit onehalf() { return from_sraw(PT::POSIT_HALF); }

		/**
		 * @brief computes the dot product between posit values vectors	 
		 *
		 * @todo decide if resulting type is Posit or another on
		 * @tparam It1 type of first iterator
		 * @tparam It2 type of second iterator
		 * @param begin1 first iterator
		 * @param begin2 second iterator
		 * @param N length of vectors
		 */
	    template <class It1, class It2>
	    static CONSTEXPR14 Posit dot(It1 begin1, It2 begin2, int N)
	    {
	    	return math::dot(begin1,begin2,N,Posit(),Posit());
	    }

		/// @todo only for non float backend
		constexpr uint32_t as_float_bin() const { return to_backend().template pack_xfloati<single_trait>(); }



		constexpr operator BackendT() const { return to_backend(); }


		constexpr operator float() const { return to_backend(); }
		constexpr operator double() const { return to_backend(); }
		
		constexpr operator short int() const { return (int)to_backend();}
		constexpr operator int() const { return to_backend(); }
		constexpr operator long() const { return to_backend(); }
		constexpr operator long long int() const { return to_backend();}		

		constexpr operator short unsigned int() const { return (unsigned int)to_backend();}
		constexpr operator unsigned int() const { return to_backend();}
		constexpr operator long unsigned int() const { return to_backend();}		
		constexpr operator long long unsigned int() const { return to_backend();}		


		constexpr operator bool() const { return !is_zero();}
		constexpr operator std::complex<float>() const {return std::complex<float>((float)to_backend());}
		constexpr operator std::complex<double>() const {return std::complex<float>((float)to_backend());}
		constexpr operator unsigned char() const { return (int)to_backend();}
		constexpr operator signed char() const { return (int)to_backend();}

		
		constexpr bool operator==(double d) {
			return d == (double)*this;
		}

		constexpr bool operator==(float f) {
			return f == (float)*this;
		}	

		constexpr bool operator==(Posit p) {
			return v == p.v;
		}			
					
		
		/// @brief Natural logarithm of the Posit
		/// @return Posit
		constexpr Posit log() const { return Posit(std::log((double)*this)); }

		/// @brief Pseudosigmoid without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit
		constexpr Posit pseudosigmoid() const { 
            //return from_sraw((v + 127 +2) *  64/256);
            return from_sraw((v+(PT::POSIT_INVERTBIT << 1) + 2) >> 2);
            //return from_sraw((PT::POSIT_INVERTBIT+(v >> 1))>>1); 
        };

		/// @brief Alternative Pseudosigmoid without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit    	
		constexpr Posit pseudosigmoidalt() const { 
            using POSIT_UTYPE = typename PT::POSIT_UTYPE;
            POSIT_UTYPE uv = (POSIT_UTYPE)v;
            return from_sraw( (T)((uv ^ PT::POSIT_SIGNMASK) >> 2));
        };

		/// @brief Pseudo-ELU without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit    
		constexpr Posit elu() const { return from_sraw((PT::POSIT_INVERTBIT+(v >> 1))>>1); };

		/// @brief Pseudo-SoftPlus without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit    
		Posit pseudosoftplus() const { return *this > Posit(3) ? *this : (-*this).pseudosigmoid().reciprocate().log(); };

		/// @brief Pseudo-generalized TANH without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit   
		constexpr Posit pseudotanh(int k) const { return Posit(k)*((Posit(k)*from_sraw(v)).pseudosigmoid())-Posit(k/2);}

		/// @brief Pseudo-Hyperbolic Tangent without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit   
		constexpr Posit pseudotanh() const { return (twice().pseudosigmoid()).twice()-one();}


		/// @brief Pseudo-Hyperbolic Tangent without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit  
		constexpr Posit fastTanh() const { 
		   return (v < 0)? this->twice().pseudosigmoid().twice().one_minus_ur().neg():this->twice().neg().pseudosigmoid().twice().one_minus_ur();
		}

		/// @brief Pseudo-ELU without decoding the Posit; only works for 0-bit exponent posits
		/// @return Posit   
		constexpr Posit fastELU() const {
			return (v >= 0)? *this: this->neg().pseudosigmoid().reciprocate().half().one_minus_ur().twice().neg();
		}

		/// @brief 1's complement without decoding the posit; only works for 0-bit exponent posits in the unary interval
		/// @return Posit   
		constexpr Posit one_minus_ur() const { return PT::POSIT_ESP_SIZE == 0 ? from_sraw(PT::POSIT_INVERTBIT-v) : one()-*this; }
	};


	#ifdef POSIT_VALUE_LOGGER
		template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
		Logger<double> Posit<T,totalbits,esbits,FT,positspec>::_logger;
	#endif


	
	/// @brief Operator<< for Posit 
	/// @tparam T Posit holding type
	/// @tparam FT Posit backend type
	/// @tparam totalbits Posit number of bits
	/// @tparam esbits Posit maximum exponent bits
	/// @tparam positspec Posit specifics for special numbers
	/// @param ons std::ostream &
	/// @param o Posit
	/// @return std::ostream &
	template <class T, int totalbits, int esbits, class FT, PositSpec positspec>
	std::ostream & operator << (std::ostream & ons, Posit<T,totalbits,esbits,FT,positspec> const & o)
	{
		// mapping std iostream flags as Posit printing
		// - as float
		// - as raw
		// - as parsed
		auto af = ons.flags();
		auto f = af & std::ios::basefield;
		ons.setf(std::ios::dec, std::ios::basefield);
		if(f == std::ios::hex)
		{
			ons << std::hex << o.v;
		}
		else if(f == std::ios::dec)
		{
			ons << (float)o;
		}
		else if(f == std::ios::oct)
		{
			//ons << o.to_backend();
		}	
		// reset
		ons.setf(af & std::ios::basefield, std::ios::basefield);
		return ons;
	}

	/// @brief Negation of the Posit 
	/// @tparam T Posit holding type
	/// @tparam FT Posit backend type
	/// @tparam totalbits Posit number of bits
	/// @tparam esbits Posit maximum exponent bits
	/// @tparam positspec Posit specifics for special numbers
	/// @param x Posit
	/// @return Posit
	template <class T, int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr Posit<T,totalbits,esbits,FT,positspec> neg(Posit<T,totalbits,esbits,FT,positspec> x) { return -x; }

	/// @brief Reciprocate of the Posit 
	/// @tparam T Posit holding type
	/// @tparam FT Posit backend type
	/// @tparam totalbits Posit number of bits
	/// @tparam esbits Posit maximum exponent bits
	/// @tparam positspec Posit specifics for special numbers
	/// @param x Posit
	/// @return Posit
	template <class T, int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr Posit<T,totalbits,esbits,FT,positspec> reciprocate(Posit<T,totalbits,esbits,FT,positspec> x) { return ~x; }

	template <class T, int hbits,int ebits, bool zeroes>
	struct msb_exp {};

	template <class T, int hbits,int ebits>
	struct msb_exp<T,hbits,ebits,true>
	{
		 constexpr T operator()(T) const
		{
			return 0;
		}
	};

	template <class T, int hbits,int ebits>
	struct msb_exp<T,hbits,ebits,false>
	{
	 constexpr T operator()(T exp) const
		{
			return exp << (hbits-ebits);
		}

	};

	/**
	 * @brief Unpacks Posit
	 * 
	 * @tparam T 
	 * @tparam totalbits 
	 * @tparam esbits 
	 * @tparam FT 
	 * @tparam positspec 
	 * @return UnpackedPosit 
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 auto Posit<T,totalbits,esbits,FT,positspec>::unpack_low() const -> UnpackedPosit
	{
	    using PT=PositTrait<T,totalbits,esbits,positspec>;
	    using POSIT_UTYPE = typename PT::POSIT_UTYPE;
	    //using POSIT_STYPE = typename PT::POSIT_STYPE;

		if(is_infinity()) // infinity
	    {
	    	return UnpackedPosit(NumberType::Infinity, v < 0);
	    }
	    else if(is_nan())
	   	{	
	    	return UnpackedPosit(NumberType::NaN);
	   	}	
	    else if(v == 0)
	    	return UnpackedPosit(NumberType::Zero);
		else
		{
	        //constexpr int POSIT_RS_MAX = PT::POSIT_SIZE-1-esbits;

			//r.type = BackendT::Regular;
			bool negativeSign = (v & PT::POSIT_SIGNBIT) != 0;
			//std::cout << "unpacking " << std::bitset<sizeof(T)*8>(pa) << " abs " << std::bitset<sizeof(T)*8>(pcabs(pa)) << " r.negativeSign? " << r.negativeSign << std::endl;
	        T pa = negativeSign ? -v : v;
			//	std::cout << "after " << std::hex << pa << std::endl;

	        POSIT_UTYPE pars1 = pa << (PT::POSIT_EXTRA_BITS+1); // MSB: RS ES FS MSB
	        auto q = PT::decode_posit_rs(pars1);
	        int reg = q.first;
	        int rs = q.second;
	        POSIT_UTYPE pars2 = pars1 << rs; // MSB: ES FS
	        POSIT_UTYPE exp = bitset_leftmost_get_const<POSIT_UTYPE,esbits>()(pars2); //        bitset_leftmost_getT(pars,esbits);
	        POSIT_UTYPE pars = pars2 << esbits; // MSB: FS left aligned in T

	        return UnpackedPosit(negativeSign,reg,exp,pars);
		}
	}

	/**
	 * @brief Packs into Posit
	 * 
	 * @tparam T 
	 * @tparam totalbits 
	 * @tparam esbits 
	 * @tparam FT 
	 * @tparam positspec 
	 * @param x 
	 * @return Posit 
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 auto Posit<T,totalbits,esbits,FT,positspec>::pack_low(UnpackedPosit x) -> Posit
	{
		using PP=Posit<T,totalbits,esbits,FT,positspec>;
	    using PT=typename Posit<T,totalbits,esbits,FT,positspec>::PT;
	    using POSIT_UTYPE = typename PT::POSIT_UTYPE;
	    using POSIT_STYPE = typename PT::POSIT_STYPE;
        constexpr POSIT_UTYPE maxShift = sizeof(POSIT_UTYPE)*8 - 1;
	    switch(x.type)
		{
			case NumberType::Infinity:
				// if infinity is missing return nan
				return positspec != PositSpec::WithNan ? (x.negativeSign ? PP::ninf(): PP::pinf()): PP::nan();
			case NumberType::Zero:
				return PP(typename PP::DeepInit(),0);
			case NumberType::NaN:
				// if nan is missing return infinity
				return PT::withsignedinf ? PP::pinf() : PP::nan();
			default:
				break;
		}

		auto exp = x.exp;
		auto reg = x.regime;

	    // for reg>=0: 1 0[reg+1] => size is reg+2 
	    // for reg <0: 0[-reg] 0  => size is reg+1
	    auto rs = -reg+1 > reg+2 ? -reg+1:reg+2; //std::max(-reg + 1, reg + 2);  MSVC issue
	    auto es = (totalbits-rs-1) < esbits ? (totalbits-rs-1): esbits; //std::min((int)(totalbits-rs-1),(int)esbits);  MSVC issue
	    
        auto rses_shift = std::min<POSIT_UTYPE>(maxShift,rs+es+1);
	    POSIT_UTYPE regbits = reg < 0 ? (PT::POSIT_HOLDER_MSB >> -reg) : (PT::POSIT_MASK << (PT::POSIT_HOLDER_SIZE-(reg+1))); // reg+1 bits on the left
		POSIT_UTYPE eexp = msb_exp<POSIT_UTYPE,PT::POSIT_HOLDER_SIZE,esbits,(esbits == 00)>()(exp);
		POSIT_UTYPE fraction =  x.fraction;
        POSIT_UTYPE pf = (rses_shift == maxShift)? 0: fraction >> rses_shift;
        POSIT_UTYPE pf_rem = (pf == 0)? 0:fraction % (1u<<rses_shift);
        POSIT_UTYPE pe = (eexp >> (rs+1));
        POSIT_UTYPE pr = (regbits>>1) >> (sizeof(T)*8-totalbits);
        POSIT_STYPE p = pf | pe | pr;

	    auto retp =  PP(typename PP::DeepInit(),x.negativeSign ? -p : p);

        constexpr POSIT_UTYPE sticky = sizeof(POSIT_UTYPE)*4-1;

        if(pf_rem >= sticky) return retp.next();
        return retp;
	}

	/**
	 * @brief Computes half as Level 1 for es=0, Level 2 for es != 0
	 * 
	 * @tparam T 
	 * @tparam totalbits 
	 * @tparam esbits 
	 * @tparam FT 
	 * @tparam positspec 
	 * @return Posit<T,totalbits,esbits,FT,positspec> 
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 auto Posit<T,totalbits,esbits,FT,positspec>::half() const -> Posit<T,totalbits,esbits,FT,positspec>
	{
		if(esbits == 0)
		{
			return from_sraw(PT::fast_half(v));
		}
		else
		{
			UnpackedPosit q = unpack_low();
			if(q.type == NumberType::Regular)
			{
				// +- 2^(R expmax + E) 1.xyz  == +- 2^(exp) 1.xyz
				// where xyz are decimal digits
				// 1.xyz / 2     => 0.1xyz ==> just exp--
				//
				// exp-- mean E-- if E s not null
				// otherwise R-- and exp 
				if(q.exp == 0)
				{
					q.regime--; // will it undrflow?
					q.exp = PT::POSIT_REG_SCALE-1; // maximum exponent
				}
				else
				{
					q.exp--;
				}
				return pack_low(q);
			}
			else
			{
				return *this;
			}
		}
	}

	/**
	 * @brief Computes twice as Level 1 for es=0, Level 2 for es != 0
	 * 
	 * Level 1 for zero esponentbits
	 *
	 * +- 2^(R) 1.xyz  == +- 2^(exp) 1.xyz 
	 * 1.xyz * 2     => 1.0xyz ==> just R++ meaning shift R left up to overflow
	 *
	 * Level 2 otherwise
	 *
	 * +- 2^(R expmax + E) 1.xyz  == +- 2^(exp) 1.xyz 
	 * 1.xyz * 2     => 1.0xyz ==> just exp++ 
	 * exp++ ==> E++ if E not maximal
	 * otherwise R++ and exp zero
	 * 
	 * @tparam T 
	 * @tparam totalbits 
	 * @tparam esbits 
	 * @tparam FT 
	 * @tparam positspec 
	 * @return Posit<T,totalbits,esbits,FT,positspec> 
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 auto Posit<T,totalbits,esbits,FT,positspec>::twice() const -> Posit<T,totalbits,esbits,FT,positspec>
	{
		if(esbits == 0)
		{
			return from_sraw(PT::fast_twice(v));
		}
		else
		{
			UnpackedPosit q = unpack_low();
			if(q.type == NumberType::Regular)
			{

				if(q.exp == PT::POSIT_REG_SCALE-1)
				{
					q.regime++; // will it overflo??
					q.exp = 0; // maximum exponent
				}
				else
				{
					q.exp++;
				}
				return pack_low(q);
			}
			else
			{
				return *this;
			}
		}
	}


	/**
	 * @brief Packs from Backend to Posit passing throught the Unpacked form
	 * 
	 * @tparam T 
	 * @tparam totalbits 
	 * @tparam esbits 
	 * @tparam FT 
	 * @tparam positspec 
	 * @param x 
	 * @return CONSTEXPR14 pack_posit 
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> pack_posit(const typename Posit<T,totalbits,esbits,FT,positspec>::BackendT & x)
	{
		using PP=Posit<T,totalbits,esbits,FT,positspec>;
		using BE=typename Posit<T,totalbits,esbits,FT,positspec>::BackendT;

		if constexpr (std::is_base_of<HwBaseBackend, BE>::value)
			return PP::from_sraw(x.v);
		else
			return PP::pack_low(PP::unpacked_full2low(x));
	}
	
	/**
	 * @brief Unpack from Posit to Backend passing throught the Unpacked form
	 * 
	 * @tparam T 
	 * @tparam totalbits 
	 * @tparam esbits 
	 * @tparam FT 
	 * @tparam positspec 
	 * @param x 
	 * @return CONSTEXPR14 unpack_posit 
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 auto unpack_posit(const Posit<T,totalbits,esbits,FT,positspec> & p) -> typename Posit<T,totalbits,esbits,FT,positspec>::BackendT 
	{
		using PP=Posit<T,totalbits,esbits,FT,positspec>;

		using BE=typename Posit<T,totalbits,esbits,FT,positspec>::BackendT;

		if constexpr (std::is_base_of<HwBaseBackend,BE >::value)
			return BE({},p.v);
		else
		return PP::unpacked_low2full(p.unpack_low());
	}

	/**
	 * @brief Absolute value of Posit
	 * 
	 * @tparam T 
	 * @tparam totalbits 
	 * @tparam esbits 
	 * @tparam FT 
	 * @tparam positspec 
	 * @param z 
	 * @return CONSTEXPR14 abs 
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	inline CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> __abs(const Posit<T,totalbits,esbits,FT,positspec> & z) 
	{
		return Posit<T,totalbits,esbits,FT,positspec>::from_sraw(pcabs(z.v));
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator == (const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & u) 
	{
		return a.comparable(u) && a.v == u.v;
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator == (const Posit<T,totalbits,esbits,FT,positspec> & a, const double & d)
	{
		return d == (double)a;
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator == (const Posit<T,totalbits,esbits,FT,positspec> & a, const float & f)  
	{
		return f == (float)a;
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator != (const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & u)  
	{
		return a.comparable(u) && a.v != u.v;
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator < (const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & u)   
	{
		return a.comparable(u) && a.v < u.v;
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator <= (const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & u)  
	{
		return a.comparable(u) && a.v <= u.v;
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator > (const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & u)  
	{
		return a.comparable(u) && a.v > u.v;
	}

	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	constexpr bool operator >= (const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & u)  
	{
		return a.comparable(u) && a.v >= u.v;
	}

	/// @brief Division operator between two Posits
	/// @param a dividend Posit
	/// @param b divider Posit
	/// @return Division of a and b
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> operator/(const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & b)  { 
		return pack_posit<T,totalbits,esbits,FT,positspec> (a.to_backend()/b.to_backend()); 
	}

	/// @brief Addition operator between two Posits
	/// @param a addend Posit
	/// @param b addend Posit
	/// @return Addition of a and b
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
    CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> operator+(const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & b)
    {
        auto ab = a.to_backend();
        auto bb = b.to_backend();
        auto rb = ab+bb;
        auto res = pack_posit<T,totalbits,esbits,FT,positspec>(rb);
        return a.is_zero() ? b : b.is_zero() ? a: res;
    }

	/// @brief Subtraction operator between two Posits
	/// @param a Addend Posit
	/// @param b Subtracting Posit
	/// @return Subtraction of a and b
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> operator-(const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & b)
	{
		return a + (-b);
	}

	/// @brief Multiplication operator between two Posits
	/// @param a multiplicand Posit
	/// @param b multiplicand Posit
	/// @return Multiplication of a and b
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> operator*(const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & b) 
	{
		return pack_posit<T,totalbits,esbits,FT,positspec>(a.to_backend()*b.to_backend());
	}

	/// @brief Fused multiply and add operation between two posits without intermediate re-packing
	/// @param a Multiplicand posit 
	/// @param b Multiplicand posit
	/// @param c Accumulator posit
	/// @return a * b + c
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	CONSTEXPR14 Posit<T,totalbits,esbits,FT,positspec> fma(const Posit<T,totalbits,esbits,FT,positspec> & a, const Posit<T,totalbits,esbits,FT,positspec> & b, const Posit<T,totalbits,esbits,FT,positspec> & c)
	{
		return pack_posit<T,totalbits,esbits,FT,positspec>(a.to_backend()*b.to_backend()+c.to_backend());
	}

	/**
	 * @brief Relu Activation function (Level 1)
	 * @return 0 for negative and not nan, otherwise identity
	 */
	template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
	Posit<T,totalbits,esbits,FT,positspec> relu(const Posit<T,totalbits,esbits,FT,positspec> & a)
	{
		return a.is_nan() || a.v >= 0 ? a : Posit<T,totalbits,esbits,FT,positspec>::zero();
	}

}



/// Standardized Posit configurations in the RISC-V RV64Xposit extension
using P8 = posit::Posit<int8_t, 8 , 0, uint_fast32_t, posit::PositSpec::WithNan>;
using P16_0 = posit::Posit<int16_t, 16 , 0, uint_fast32_t, posit::PositSpec::WithNan>;
using P16_1 = posit::Posit<int16_t, 16 , 1, uint_fast32_t, posit::PositSpec::WithNan>;
using P8fx = posit::Posit<int8_t, 8 , 0, posit::BackendFixed<fixedtrait<int16_t,16>,int16_t>, posit::PositSpec::WithNan>;
using P16_0fx = posit::Posit<int16_t, 16 , 0, posit::BackendFixed<fixedtrait<int32_t,32>,int32_t>, posit::PositSpec::WithNan>;
using P16_0fxq = posit::Posit<int16_t, 16 , 0, posit::BackendFixed<fixedtrait<int64_t,64>,int64_t>, posit::PositSpec::WithNan>; // 16,0 with doubled backend, for quire support
using P16_1fx = posit::Posit<int16_t, 16 , 1, posit::BackendFixed<fixedtrait<int64_t,64>,int32_t >, posit::PositSpec::WithNan>;


