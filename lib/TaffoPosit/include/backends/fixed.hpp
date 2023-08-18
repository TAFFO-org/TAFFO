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
     * @tparam FT is the Fixed Trait <T,N,F>
     */
    template <class PT, class Target>
	struct PositPacker;

    template <class T>
	struct is_posit_backend;

    template <class PT>
	struct UnpackedPosit;

	/// @brief Fixed Backend for Posit
	/// @tparam FT FixedTrait class to specify fixed-point number of bits
	/// @tparam ET ExponentType used to construct the underlying UnpackedPosit
	template <class FT, class ET>
	struct BackendFixed
	{
		using ft = FT;
		using value_t = typename ft::value_t;
		using OBE = Unpacked<uint32_t,ET>;
		struct single_tag{};
		struct raw_tag{};

		BackendFixed() : v(0) {}
		BackendFixed(raw_tag, value_t x) : v(x) {}
		BackendFixed(single_tag, uint32_t x): BackendFixed(OBE::template make_floati<single_trait>(x)) {}
		BackendFixed(int x) : BackendFixed(OBE(x))  {}
		BackendFixed(long int x) : BackendFixed(OBE(x))  {}
		BackendFixed(long long int x) : BackendFixed(OBE(x))  {}

		BackendFixed(float x) : BackendFixed(OBE(x)) {}
		BackendFixed(double x) : BackendFixed(OBE(x))  {}
		BackendFixed(OBE u): v(u.template pack_xfixed<FT>()) {}

		template <class XFT>
		static BackendFixed make_floati(typename XFT::holder_t x) { return BackendFixed(OBE::template make_floati<XFT>(x)); }

		static BackendFixed fromraw(value_t v) { return BackendFixed(raw_tag(),v); }

		operator OBE () const { return OBE::template make_fixed<FT>(v); }
		constexpr operator float () const { return OBE::template make_fixed<FT>(v);}
		constexpr operator double () const { return OBE::template make_fixed<FT>(v);}
		constexpr operator int () const { return OBE::template make_fixed<FT>(v);}
		constexpr operator long int () const { return OBE::template make_fixed<FT>(v);}
		constexpr operator long long int () const { return OBE::template make_fixed<FT>(v);}

	 	BackendFixed operator + (BackendFixed o) const { return fromraw(v+o.v); }
		BackendFixed operator * (BackendFixed o) const { return fromraw(v*o.v); }
	 	BackendFixed operator - (BackendFixed o) const { return fromraw(v-o.v); }
	 	BackendFixed operator /(BackendFixed o) const { return fromraw(v/o.v); }

		value_t v;
	};

	template <class B, class ET>
	struct is_posit_backend<BackendFixed<B,ET> >: public std::true_type
	{
	};

	template <class FT, class ET>
	std::ostream & operator<<(std::ostream &ons, BackendFixed<FT,ET> const &o) 
    {
    	ons << "fix(" << o.v << " = " << (float)o << ")";
    	return ons;
    }

	template <class PT, class FT>
	struct PositPacker<PT,BackendFixed<FT,typename PT::exponenttype> >
	{
		using BE = BackendFixed<FT,typename PT::exponenttype>;
		using OBE = typename BE::OBE;
		using UP = UnpackedPosit<PT>;

		static CONSTEXPR14 BE unpacked_to_backend(UP up)
		{
			return BE::fromraw(PositPacker<PT,OBE>::unpacked_to_backend(up).template pack_xfixed<FT>());
		}

		static CONSTEXPR14 UP backend_to_unpacked(BE b)
		{
			return PositPacker<PT,OBE>::backend_to_unpacked(OBE::template make_fixed<FT>(b.v));
		}
	};
}