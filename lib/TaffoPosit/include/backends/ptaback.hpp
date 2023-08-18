/**
 * Copyright (C) 2017-2023 Emanuele Ruffaldi, Federico Rossi
 * 
 * Distributed under the terms of the BSD 3-Clause License.  
 * 
 * (See accompanying file LICENSE)
 * 
 * --
 */
#ifdef _PTA_
#include <pta_zcu.h>

namespace posit {

    template <class T>
	struct is_posit_backend;

	template <class T,class PositEmu, class PtaCfg>
	struct PtaBack: public HwBaseBackend
	{
		struct single_tag{};

		PtaBack() {}
		PtaBack(single_tag, T x): v(x) {}

		explicit PtaBack(int x) {v=PositEmu(x).v;}
		explicit PtaBack(long x) {v=PositEmu(x).v;}
		explicit PtaBack(float x) {v=PositEmu(x).v;}
		explicit PtaBack(double x) {v=PositEmu(x).v;}

		constexpr operator float () const {return (float)PositEmu::from_sraw(v);}
		constexpr operator double () const {return (double)PositEmu::from_sraw(v);}
		constexpr operator int () const {return (int)PositEmu::from_sraw(v);}
		constexpr operator long () const {return (long)PositEmu::from_sraw(v);}

	 	PtaBack operator + (PtaBack o) const { 
			sformat_ r, a(o.v), b(v);
			pta_add(&r,&(a),&(b), PtaCfg::unit, A_POSIT | B_POSIT | C_POSIT | R_POSIT);
			return PtaBack{{},T(r)}; 
		}

		PtaBack operator * (PtaBack o) const {
			sformat_ r, a(o.v), b(v);
			pta_mul(&r,&(a),&(b), PtaCfg::unit, A_POSIT | B_POSIT | C_POSIT | R_POSIT);
			return PtaBack{{},T(r)}; 
		}
	 	PtaBack operator - (PtaBack o) const {
			sformat_ r, a(o.v), b(v);
			pta_sub(&r,&(a),&(b), PtaCfg::unit, A_POSIT | B_POSIT | C_POSIT | R_POSIT);
			return PtaBack{{},T(r)}; 
		}
	 	PtaBack operator / (PtaBack o) const {
			sformat_ r, a(o.v), b(v);
			pta_div(&r,&(a),&(b), PtaCfg::unit, A_POSIT | B_POSIT | C_POSIT | R_POSIT);
			return PtaBack{{},T(r)}; 
		}
		T v;

        static bool initialize() {
            int descriptor = init_pta();
            return descriptor > 0;
        }

	};

    template <class T, class PositEmu, class PtaCfg>
	struct is_posit_backend<PtaBack<T,PositEmu, PtaCfg> >: public std::true_type
	{
	};
    
}
#endif