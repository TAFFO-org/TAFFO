/**
 * Copyright (C) 2017-2019 Emanuele Ruffaldi
 * Distributed under the terms of the BSD 3-Clause License.  
 * 
 * (See accompanying file LICENSE)
 * 
 * --
 */

#pragma once
#include "posit.h"
#ifdef POSIT_EIGEN
#include <Eigen/Core>
#endif POSIT_EIGEN

using namespace posit;
namespace Eigen {

/**
 * @brief Template Trait that allows to use posit in Eigen
 */
template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
struct NumTraits<Posit<T,totalbits,esbits,FT, positspec> >
 :  GenericNumTraits<Posit<T,totalbits,esbits,FT,positspec> > // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef Posit<T,totalbits,esbits,FT,positspec> P;

  typedef P Real;
  typedef P NonInteger;
  typedef P Nested;

  //static inline Real epsilon() { return 0; }
  //static inline Real dummy_precision() { return 0; }
  //static inline Real digits10() { return 0; }
  // highest() and lowest() functions returning the highest and lowest possible values respectively.
  // An epsilon() function which, unlike std::numeric_limits::epsilon(), it returns a Real instead of a T.
  // digits10() function returning the number of decimal digits that can be represented without change. This is the analogue of std::numeric_limits<T>::digits10 which is used as the default implementation if specialized.
  

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};



template <class FT, class ET>
struct NumTraits<Unpacked<FT,ET> >
 :  GenericNumTraits<Unpacked<FT,ET> > // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef Unpacked<FT,ET> P;

  typedef P Real;
  typedef P NonInteger;
  typedef P Nested;

  //static inline Real epsilon() { return 0; }
  //static inline Real dummy_precision() { return 0; }
  //static inline Real digits10() { return 0; }
  // highest() and lowest() functions returning the highest and lowest possible values respectively.
  // An epsilon() function which, unlike std::numeric_limits::epsilon(), it returns a Real instead of a T.
  // digits10() function returning the number of decimal digits that can be represented without change. This is the analogue of std::numeric_limits<T>::digits10 which is used as the default implementation if specialized.
  

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

namespace internal {

  template <class T,int totalbits, int esbits, class FT, PositSpec positspec>
      inline typename Posit<T,totalbits,esbits,FT,positspec>::UnpackedT cast(const Posit<T,totalbits,esbits,FT,positspec>& x)
    { return x.unpack(); }
}





}

/* REGISTER FOR CMATH FUNCTIONS OVERLOAD */

#define REGISTER_CMATH_FUN(F) \
  template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec> \
  constexpr posit::Posit<T,totalbits,esbits,FT,positspec> F(const posit::Posit<T,totalbits,esbits,FT,positspec>& a) \
  { \
      using PP = posit::Posit<T,totalbits,esbits,FT,positspec>; \
      return (PP)std::F((float)a); \
  } \

#define REGISTER_CMATH_FUN2(F) \
  template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec> \
  constexpr posit::Posit<T,totalbits,esbits,FT,positspec> F(const posit::Posit<T,totalbits,esbits,FT,positspec>& a,const posit::Posit<T,totalbits,esbits,FT,positspec>& b) \
  { \
      using PP = posit::Posit<T,totalbits,esbits,FT,positspec>; \
      return (PP)std::F((float)a,(float)b); \
  } \

#define REGISTER_CMATH_FUNB(F) \
  template <class T,int totalbits, int esbits, class FT, posit::PositSpec positspec> \
  constexpr bool F(const posit::Posit<T,totalbits,esbits,FT,positspec>& a) \
  { \
      using PP = posit::Posit<T,totalbits,esbits,FT,positspec>; \
      return std::F((float)a); \
  } \  

namespace posit {
  REGISTER_CMATH_FUN(exp);
  REGISTER_CMATH_FUN(log);
  REGISTER_CMATH_FUN(log1p);
  REGISTER_CMATH_FUN(sqrt);
  REGISTER_CMATH_FUN(ceil);
  REGISTER_CMATH_FUN(acos);
  REGISTER_CMATH_FUN(acosh);
  REGISTER_CMATH_FUN(asin);
  REGISTER_CMATH_FUN(asinh);
  REGISTER_CMATH_FUN(atan);
  REGISTER_CMATH_FUN(atanh);
  REGISTER_CMATH_FUN(cos);
  REGISTER_CMATH_FUN(cosh);
  REGISTER_CMATH_FUN(expm1);
  REGISTER_CMATH_FUN(sin);
  REGISTER_CMATH_FUN(sinh);
  REGISTER_CMATH_FUN(tan);
  REGISTER_CMATH_FUN(tanh);
  REGISTER_CMATH_FUNB(isnan);
  REGISTER_CMATH_FUN(floor);
  REGISTER_CMATH_FUNB(isinf);
  REGISTER_CMATH_FUNB(isfinite);
  REGISTER_CMATH_FUN2(pow);
  REGISTER_CMATH_FUN2(fmod);
}
	