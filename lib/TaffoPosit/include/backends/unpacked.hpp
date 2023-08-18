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

    template <class PT, class Target>
	struct PositPacker;

    template <class T>
	struct is_posit_backend;

    template <class PT>
	struct UnpackedPosit;

    /**
	 * @brief Unpacked representaiton of the Posit with (sign,regime,exponent,fraction)
	 * 
	 * @tparam PT is the posit trait
	 */
	template <class PT>
	struct UnpackedPosit
	{
		constexpr UnpackedPosit(NumberType t): type(t), negativeSign(false), regime(0),exp(0),fraction(0) {}
		constexpr UnpackedPosit(NumberType t, bool anegativeSign): type(t), negativeSign(anegativeSign), regime(0),exp(0),fraction(0) {}
		constexpr UnpackedPosit(bool n, typename PT::POSIT_STYPE r, typename PT::POSIT_UTYPE e, typename PT::POSIT_UTYPE f):
			 type(NumberType::Regular),negativeSign(n), regime(r), exp(e), fraction(f) {}

		NumberType type; /// type of posit special number
		bool negativeSign; // for Regular and Infinity
		typename PT::POSIT_STYPE regime; /// decoded regime value
		typename PT::POSIT_UTYPE exp;    /// decoded exponent value (positive)
		typename PT::POSIT_UTYPE fraction; /// decoded fraciton left aligned with the leading 1
	};

    /// @brief Helper class to pack and unpack posit from/to backend
    /// @tparam PT Posit type
    /// @tparam FT Fraction type
    template <class PT, class FT>
	struct PositPacker<PT,Unpacked<FT,typename PT::exponenttype> >
	{
		using BE = Unpacked<FT,typename PT::exponenttype>;
		using UP = UnpackedPosit<PT>;

		/// @brief Construct the posit Backend from an UnpackedPosit instance 
		/// @param up Unpacked Posit instance
		/// @return Backend type instance
		static CONSTEXPR14 BE unpacked_to_backend(UP up)
		{
			using BE = Unpacked<FT,typename PT::exponenttype>;
		    BE r;
		    r.type = up.type;
		    r.negativeSign = up.negativeSign;

		    if(up.type == NumberType::Regular)
		    {
		        r.fraction = cast_msb<typename PT::POSIT_UTYPE,PT::POSIT_HOLDER_SIZE,FT,BE::FT_bits>()(up.fraction);
		        r.exponent = PT::join_reg_exp(up.regime,up.exp);
		    }
			return r;
		}

		/// @brief Construct an Unpacked posit instance from a Backend instance
		/// @param b Backend instance
		/// @return UnpackedPosit
		static CONSTEXPR14 UP backend_to_unpacked(BE b)
		{
			if(b.type == NumberType::Regular)
			{
				auto eexponent = clamp<decltype(b.exponent)>(b.exponent,PT::minexponent(),PT::maxexponent()); // no overflow
				auto rr = PT::split_reg_exp(eexponent);
				auto frac = cast_msb<FT,sizeof(FT)*8,typename PT::POSIT_UTYPE,sizeof(typename PT::POSIT_UTYPE)*8>()(b.fraction);
				return UP(b.negativeSign,rr.first,rr.second,frac);
			}
			else
			{
				return UP(b.type,b.negativeSign);
			}		
		}		
	};

    template <class FT, class ET>
	struct is_posit_backend<Unpacked<FT,ET> >: public std::true_type {};
}