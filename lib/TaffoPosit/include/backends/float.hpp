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

	/// @brief Floating-point like backend
	/// @tparam B floating-point type (can be float or double)
	template <class B>
	struct BackendFloat
	{
		using bt = B;
		using OBE = Unpacked<uint64_t,int>;
		struct single_tag{};

		BackendFloat() {}
		BackendFloat(single_tag, uint32_t x): v(uint32_to_float(x)) {}
		BackendFloat(int x) : v(x) {}
		BackendFloat(long x) : v(x) {}
		BackendFloat(long long int x) : v(x) {}
		BackendFloat(unsigned int x) : v(x) {}
		BackendFloat(unsigned long x) : v(x) {}
		BackendFloat(unsigned long long int x) : v(x) {}
		BackendFloat(float x) : v(x) {}
		BackendFloat(double x) : v(x) {}

		template <class FT>
		static BackendFloat make_floati(typename FT::holder_t x) { return BackendFloat(OBE::make_floati<FT>(x)); }

		constexpr operator float () const {return (float)v;}
		constexpr operator double () const {return (double)v;}
		constexpr operator int () const {return (int)v;}
		constexpr operator long int () const {return (long)v;}
		constexpr operator long long int () const {return (long long int)v;}
		constexpr operator unsigned int () const {return (unsigned int)v;}
		constexpr operator unsigned long int () const {return (unsigned long int)v;}
		constexpr operator unsigned long long int () const {return (unsigned long long int)v;}

	 	BackendFloat operator + (BackendFloat o) const { return BackendFloat(v+o.v); }
		BackendFloat operator * (BackendFloat o) const { return BackendFloat(v*o.v); }
	 	BackendFloat operator - (BackendFloat o) const { return BackendFloat(v-o.v); }
	 	BackendFloat operator / (BackendFloat o) const { return BackendFloat(v/o.v); }
		bt v;
	};


	template <class B>
	struct is_posit_backend<BackendFloat<B> >: public std::true_type
	{
	};

	template <class PT, class FT>
	struct PositPacker<PT,BackendFloat<FT> >
	{
		using BE = BackendFloat<FT>;
		using OBE = Unpacked<uint64_t,typename PT::exponenttype>;
		using UP = UnpackedPosit<PT>;

		static CONSTEXPR14 BE unpacked_to_backend(UP up)
		{
			return BE((FT)PositPacker<PT,OBE>::unpacked_to_backend(up));
		}

		static CONSTEXPR14 UP backend_to_unpacked(BE b)
		{
			return PositPacker<PT,OBE>::backend_to_unpacked(OBE(b.v));
		}
	};

	template <class B>
	std::ostream & operator<<(std::ostream &ons, BackendFloat<B> const &o) 
    {
    	ons << "float(" << o.v << ")";
    	return ons;
    }






}