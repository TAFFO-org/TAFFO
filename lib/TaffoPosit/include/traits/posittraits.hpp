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

    /**
	 * @enum PositSpec
	 * @brief Specifies how a family of posit numbers behaves with respect to special numbers
	 * The fundamental special number is the opposite of zero in the posit wheel.
	 *
	 * Signless infinity has been removed
	 */
	enum PositSpec { 
		WithNone, /// never consider special numbers. The top value is never present
		WithNan, /// one top Nan (or Nar using posit spec terminology)
		WithNanInfs, /// one top Nan and two signed Infinites (for more float compatibility)
		WithInfs /// one top Nan and two signed Infinites (for more float compatibility)
	}; 

	/**
	 * @brief Trait for describing the posit specifier based on the family 
	 */
	template <PositSpec x>
	struct PositSpecTrait
	{

	};

	/**
	 * @brief Specialization without special numbers
	 */
	template <>
	struct PositSpecTrait<WithNone>
	{
		static constexpr bool withnan = false;
		static constexpr bool withinf = false;
		static constexpr bool withsignedinf = false;
	};

	/**
	 * @brief Specialization with NaN
	 */
	template <>
	struct PositSpecTrait<WithNan>
	{
		static constexpr bool withnan = true;
		static constexpr bool withinf = false;
		static constexpr bool withsignedinf = false;
	};

	/**
	 * @brief Specialization with Nan on top and infinity on side
	 */
	template <>
	struct PositSpecTrait<WithNanInfs>
	{
		static constexpr bool withnan = true;
		static constexpr bool withinf = true;
		static constexpr bool withsignedinf = true;
	};

	/**
	 * @brief Specialization with Nan on top and infinity on side
	 */
	template <>
	struct PositSpecTrait<WithInfs>
	{
		static constexpr bool withnan = false;
		static constexpr bool withinf = true;
		static constexpr bool withsignedinf = true;
	};

	template<int x>
	struct PositZero
	{

	};

	template<>
	struct PositZero<0>
	{
		using type = bool;
	};

	template<>
	struct PositZero<1>
	{
		using type = int;
	};

	/**
	 * @brief Trait used to describe the families of posit numbers
	 * The trait provides constants for constructing the posit representation and several constants of the posit expressed in the signed integer.
	 * In addition some fundamental constexpr functions are provided
	 * 
	 * @tparam T the signed integer holding the posit
	 * @tparam totalbits the number of bits of the posit less or equal then the bitsize of T. Data will be RIGHT aligned
	 * @tparam esbits the maximum number of bits for the exponent
	 * @tparam positspec_ the enumeration that controls the special numbers in a Posit. @see PositSpec
	 */
    
    template <class T, int totalbits, int esbits, PositSpec positspec_ >
	struct PositTrait
	{
		static_assert(std::is_signed<T>::value,"required signed T");
		static_assert(sizeof(T)*8 >= totalbits,"required enough storage T for provided bits  SREF");
		static_assert(esbits <= totalbits-3,"esbits should be at most N-3 for the cases [x01,E] and [x10,E]");

		using POSIT_STYPE = T;
		using POSIT_UTYPE = typename std::make_unsigned<T>::type; /// unsigned version of T
		static constexpr PositSpec positspec = positspec_;
		static constexpr bool withnan = PositSpecTrait<positspec_>::withnan;
		static constexpr bool withinf = PositSpecTrait<positspec_>::withinf;
		static constexpr bool withsignedinf = PositSpecTrait<positspec_>::withsignedinf;

		/// type necessary to hold the unpacked exponent
		using exponenttype = typename std::conditional<(totalbits+esbits >= sizeof(T)*8),typename  nextinttype<T>::type,T>::type;

		///@{ 
		/// @name Helper Sizes	
		static constexpr POSIT_UTYPE POSIT_MAXREGIME_BITS = totalbits-1; /// maximum number of regime bits (all except sign)
		static constexpr POSIT_UTYPE POSIT_HOLDER_SIZE = sizeof(T)*8;
		static constexpr POSIT_UTYPE POSIT_SIZE = totalbits;
		static constexpr POSIT_UTYPE POSIT_ESP_SIZE = esbits;
		static constexpr POSIT_UTYPE POSIT_EXTRA_BITS = POSIT_HOLDER_SIZE-totalbits; /// non posit bits in holder 
		///@}

		///@{ 
		/// @name Specific Bits
		static constexpr POSIT_UTYPE POSIT_ONEHELPER = 1;
		static constexpr POSIT_UTYPE POSIT_INVERTBIT = (POSIT_ONEHELPER<<(totalbits-2)); /// bit for 1/x
		static constexpr POSIT_UTYPE POSIT_MSB = POSIT_ONEHELPER<<(totalbits-1); /// most significant bit of posit: (0[*] 1 0[totalbits-1])
		static constexpr POSIT_UTYPE POSIT_SIGNBIT = POSIT_MSB; // sign bit in posit is MSB
		static constexpr POSIT_UTYPE POSIT_HOLDER_MSB = POSIT_ONEHELPER<<(POSIT_HOLDER_SIZE-1); /// most significant bit of holding type: (1 0[*])
		static constexpr POSIT_STYPE POSIT_REG_SCALE = 1<<esbits;
		///@}

		///@{ 
		/// @name Masks
	    static constexpr POSIT_UTYPE POSIT_ESP_MASK = (POSIT_ONEHELPER<< esbits)-1; /// mask for exponent: (0[*] 1[esbits])
	    static constexpr POSIT_UTYPE POSIT_MASK = ((POSIT_MSB-1)|(POSIT_MSB)); /// all posit bits to one on the right: (0[*] 1[totalbits])
		static constexpr POSIT_STYPE POSIT_MASK_NOSIGN = (POSIT_MASK >> 1); /// (0[*] 1[totalbits-1])
		static constexpr POSIT_UTYPE POSIT_SIGNMASK = ~POSIT_MASK_NOSIGN; /// mask for bits of signed part (1[*] 0[totalbits-1])
		///@}
        static constexpr POSIT_UTYPE POSIT_TWICEMASK = (POSIT_INVERTBIT | (POSIT_ONEHELPER << totalbits - 3) ) << 1;
		///@{ 
		/// @name Special Values	
		static constexpr POSIT_STYPE _POSIT_TOP = (POSIT_STYPE)POSIT_HOLDER_MSB; /// 1 0[*]
		static constexpr POSIT_STYPE _POSIT_TOPLEFT = (POSIT_STYPE)(POSIT_SIGNMASK+1); /// 1[*] 0[totlabits-2] 1
		static constexpr POSIT_STYPE _POSIT_TOPRIGHT = -_POSIT_TOPLEFT; /// 0[*] 1[totalbits-1]

		static constexpr POSIT_STYPE POSIT_INF =  withinf ? _POSIT_TOP : 0; /// right of top if present
		static constexpr POSIT_STYPE POSIT_PINF =  withsignedinf ? _POSIT_TOPRIGHT: POSIT_INF; /// right of top if present
		static constexpr POSIT_STYPE POSIT_NINF =  withsignedinf ? _POSIT_TOPLEFT : 0; /// top or left if present
		static constexpr POSIT_STYPE POSIT_NAN  = withnan ? _POSIT_TOP : 0;  /// infinity in withnan=false otherwise it is truly nan
		static constexpr POSIT_STYPE POSIT_ONE = POSIT_INVERTBIT; /// invert bit IS positive one
		static constexpr POSIT_STYPE POSIT_MONE = -POSIT_ONE ; /// trivially minus one
		static constexpr POSIT_STYPE POSIT_TWO = (POSIT_INVERTBIT | (POSIT_INVERTBIT>>(1+esbits))); /// 2.0
		static constexpr POSIT_STYPE POSIT_HALF = -(POSIT_STYPE)(POSIT_TWO ^ POSIT_SIGNMASK); /// 0.5
		static constexpr POSIT_STYPE POSIT_MAXPOS = _POSIT_TOPRIGHT - (withsignedinf ? 1 : 0); /// max value below Infinity: 1[holder-total] 1 0[total-1]
		static constexpr POSIT_STYPE POSIT_MINNEG = _POSIT_TOPLEFT - (withsignedinf ? 1 : 0); 	/// min value above -Infinity // 0[holder-total] 0 1[total-1]
		static constexpr POSIT_STYPE POSIT_AFTER0 = 1; /// right to 0:  minimal number above zero
		static constexpr POSIT_STYPE POSIT_BEFORE0 = -POSIT_AFTER0; /// left to 0: smallest number before zero

		using positzero = typename PositZero<esbits == 0>::type;
		///@}

		/// maxexponent that can be constructed
		static constexpr exponenttype maxexponent() { return withsignedinf ? POSIT_REG_SCALE * (totalbits - 3) : POSIT_REG_SCALE * (totalbits - 2); }
		/// minimumexponent
		static constexpr exponenttype minexponent() { return (-((exponenttype)POSIT_REG_SCALE) * (totalbits - 2)) ; }

		static constexpr bool posvalue_has_only_regime(POSIT_STYPE x)
		{
			return (x <= POSIT_ONE ? is_power_of_two(x) : is_power_of_two((POSIT_UTYPE)_POSIT_TOP-(POSIT_UTYPE)x));
		}
		/**
		 * @brief decoding of posit into Regime balue and number of remaining bits
		 * 
		 * @param pars input posit
		 * @return decode_posit_rs 
		 */
	    static CONSTEXPR14 std::pair<int,int> decode_posit_rs(T pars)
	    {  
	        const bool x = (pars & POSIT_HOLDER_MSB) != 0; // marker bit for > 1
	        int aindex = x ? (~pars == 0 ? POSIT_MAXREGIME_BITS : findbitleftmostC((POSIT_UTYPE)~pars)) : (pars == 0 ? POSIT_MAXREGIME_BITS : findbitleftmostC((POSIT_UTYPE)pars)); // index is LAST with !x
	        int index = aindex; // aindex > POSIT_SIZE  ? POSIT_SIZE : aindex;
	        int reg = x ? index-1 : -index;
	        int rs =  (int)POSIT_MAXREGIME_BITS < index+1 ? POSIT_MAXREGIME_BITS : index+1; //std::min((int)POSIT_MAXREGIME_BITS,index+1);
	        return {reg,rs};
	    }

		/**
		 * @brief Given a full exponent decomposes it into regime part and base exponent
		 * 
		 * @param eexponent 
		 * @return constexpr std::pair<POSIT_STYPE,POSIT_UTYPE> 
		 */
	    static constexpr std::pair<POSIT_STYPE,POSIT_UTYPE> split_reg_exp(exponenttype eexponent)
	    {
	        return {eexponent >> POSIT_ESP_SIZE, eexponent & POSIT_ESP_MASK };
	    }

	    /**
	     * @brief Computes the inverted position of the givem posit using a binary property of the wheel.
	     * Correct for non special values. Requires check for: zero, nan, and the adjacent values if signed infinities
	     *
		 * @param value
		 * @return 1/x for most values except: 
	     */
	    static CONSTEXPR14 POSIT_STYPE reciprocate(POSIT_STYPE x)
	    {
	    	bool s = false;
	    	POSIT_STYPE r;
	    	if(x < 0)
	    	{
	    		s = true;
	    		x = -x;
	    	}
	    	if(posvalue_has_only_regime(x))
	    	{
	    		r = (POSIT_STYPE)((POSIT_UTYPE)(POSIT_SIGNBIT)-(POSIT_UTYPE)x) ;
	    	}
	    	else
	    	{
	    		r = x ^ (~POSIT_SIGNBIT);
	    	}

	    	return s ? -r : r;
	    }

		/**
		 * @brief Merges regime and exponent
		 */
	    static constexpr exponenttype join_reg_exp(POSIT_STYPE reg, POSIT_UTYPE exp)
	    {
	    	return (((exponenttype)reg) * (1<<POSIT_ESP_SIZE))|exp;
	    }

        
	    /**
	     * @brief fast twice for E=0
	     */	    
	    //template <typename std::enable_if<std::is_same<positzero, PositZero<1>::type >::value, int>::type = 0>
	    static constexpr POSIT_STYPE fast_twice(POSIT_STYPE x)
	    {
            POSIT_STYPE s = -POSIT_STYPE(x < 0);
            POSIT_STYPE av = pabs(x);
            POSIT_STYPE X_invbit = av & POSIT_INVERTBIT;
            POSIT_STYPE Xs = av << 1;
            POSIT_STYPE Xs_invbit = Xs & POSIT_INVERTBIT;
            POSIT_STYPE x_ge1 = Xs >> 1;
            POSIT_STYPE x_lthalf = Xs << 1;
            POSIT_STYPE x_lt1 = bitwise_ite<POSIT_STYPE>(Xs_invbit == 0,x_lthalf,Xs ^ POSIT_TWICEMASK);
            POSIT_STYPE Y1 = bitwise_ite<POSIT_STYPE>(X_invbit == 0,x_lt1,x_ge1);
            Y1 = (POSIT_STYPE)((POSIT_UTYPE)Y1 >> 1);
            return (Y1 ^ s) - s;
			//return pcabs(x) < POSIT_MAXPOS ? x << 1 : x;
	    }

	    /**
	     * @brief fast half for E=0
	     */	    
	    //template <typename std::enable_if<POSIT_ESP_SIZE == 0, int>::type = 0>
	    static constexpr POSIT_STYPE fast_half(POSIT_STYPE x)
	    {
            POSIT_STYPE s = -POSIT_STYPE(x < 0);
            POSIT_STYPE av = pabs(x);
            POSIT_STYPE X_invbit = av & POSIT_INVERTBIT;
            POSIT_STYPE Xs = av << 1;
            POSIT_STYPE Xs_invbit = Xs & POSIT_INVERTBIT;
            POSIT_STYPE x_ge2 = Xs << 1;
            POSIT_STYPE x_lt1 = (POSIT_STYPE)((POSIT_UTYPE)Xs >> 1);
            POSIT_STYPE x_ge1 = bitwise_ite<POSIT_STYPE>(Xs_invbit == 0,Xs ^ POSIT_TWICEMASK,x_ge2);
            POSIT_STYPE Y1 = bitwise_ite<POSIT_STYPE>(X_invbit == 0,x_lt1,x_ge1);
            Y1 = (POSIT_STYPE)((POSIT_UTYPE)Y1 >> 1);
            return (Y1 ^ s) - s;
			//return pcabs(x) < POSIT_MAXPOS ? x >> 1 : x;
	    }

	    /**
	     * Complement of number in unitary range
	     */
	    //template <typename std::enable_if<POSIT_ESP_SIZE == 0, int>::type = 0>
		static constexpr POSIT_STYPE fast_one_minus_ur(POSIT_STYPE v) 
		{ 
			return POSIT_INVERTBIT-v;
		}
		/**
		 * \brief returns true if in the unit interval [0,1]
		 */
		static constexpr bool is_unitary_range(POSIT_STYPE v) { return (v & (POSIT_SIGNBIT|POSIT_INVERTBIT)) == POSIT_INVERTBIT; }

		/**
		 * \brief returns true if in the sphere interval [-1,1]
		 */
		static constexpr bool is_sphere_range(POSIT_STYPE v) { return (v & POSIT_INVERTBIT) == POSIT_INVERTBIT; };

	};
}