//===-- AffineForms.h - Classes related to error propagation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of various classes that perform
/// error propagation with affine forms.
///
//===----------------------------------------------------------------------===//

#ifndef ERRORPROPAGATOR_AFFINE_FORMS_H
#define ERRORPROPAGATOR_AFFINE_FORMS_H

#include <cmath>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "errorprop"

#define DEFAULT_NOISE_SIZE 2

namespace ErrorProp
{

/// An interval of values of type T.
template <typename T>
struct Interval {
  T Min;
  T Max;

  Interval()
      : Min(0), Max(0) {}

  Interval(const T MinValue, const T MaxValue)
      : Min(MinValue), Max(MaxValue)
  {
    assert((std::isnan(Min) && std::isnan(Max)) || Min <= Max && "Interval bounds inconsistent.");
  }

  bool operator==(const Interval<T> &O) const
  {
    return this->Min == O.Min && this->Max == O.Max;
  }
};

/// Base class for noise terms.
///
/// It handles the identification of each noise term as a symbolic value.
struct NoiseTermBase {
public:
  typedef unsigned long SymbolT;

  SymbolT Symbol; ///< Noise symbol identifier.

  bool operator<(const NoiseTermBase &Other) const
  {
    return this->Symbol < Other.Symbol;
  }

protected:
  /// Construct a NoiseTerm with a new unique symbolic value.
  NoiseTermBase()
      : Symbol(nextSymId()) {}

  /// Construct a NoiseTerm with the given symbolic value.
  NoiseTermBase(const SymbolT NoiseSymbol)
      : Symbol(NoiseSymbol)
  {
    if (SymId <= NoiseSymbol) {
      SymId = NoiseSymbol + 1;
    }
  }

  static SymbolT SymId;

  static SymbolT nextSymId()
  {
    return SymId++;
  }
};

/// A noise term for affine arithmetic.
///
/// It represents an error term as a magnitude multiplied by a symbolic value.
template <typename T>
struct NoiseTerm : NoiseTermBase {
public:
  T Magnitude; ///< Magnitude of the noise term.

  /// Constructs a NoiseTerm of magnitude 0 with a new unique symbolic value.
  NoiseTerm()
      : Magnitude(0) {}

  /// Constructs a NoiseTerm of magnitude NoiseMagnitude
  /// with a new unique symbolic value.
  NoiseTerm(const T NoiseMagnitude)
      : Magnitude(NoiseMagnitude) {}

  /// Constructs a NoiseTerm of magnitude NoiseMagnitude
  /// with the given symbolic value.
  NoiseTerm(const SymbolT NoiseSymbol, const T NoiseMagnitude)
      : NoiseTermBase(NoiseSymbol), Magnitude(NoiseMagnitude) {}

  /// Check whether two NoiseTerms have the same symbolic value
  /// (i.e. they are comparable).
  bool matches(const NoiseTerm<T> &O) const
  {
    return this->Symbol == O.Symbol;
  }

  NoiseTerm<T> &operator+=(const NoiseTerm<T> &O)
  {
    assert(this->matches(O) && "Only matching NoiseTerms can be summed.");
    this->Magnitude += O.Magnitude;
    return *this;
  }

  NoiseTerm<T> operator+(const NoiseTerm<T> &O) const
  {
    return NoiseTerm<T>(*this) += O;
  }

  NoiseTerm<T> operator-() const
  {
    return NoiseTerm<T>(this->Symbol, -this->Magnitude);
  }

  NoiseTerm<T> &operator-=(const NoiseTerm<T> &O)
  {
    return *this += -O;
  }

  NoiseTerm<T> operator-(const NoiseTerm<T> O) const
  {
    return *this + (-O);
  }
};

/// An affine form representing a number of type with error terms
///
/// A number is represented as
/// x = x0 + x1*eps1 + x2 * eps2 + ... + xn * epsn
/// See de Figueiredo et al., Affine arithmetic: concepts and applications.
template <typename T>
class AffineForm
{
public:
  /// Construct an AffineForm with value 0 and no error terms.
  AffineForm() : X0(0), Xi() {}

  /// Construct an AffineForm with value CentralValue and no error terms.
  AffineForm(const T CentralValue) : X0(CentralValue), Xi() {}

  /// Construct an AffineForm with value CentralValue,
  /// and a single error term with magnitude NoiseMagnitude.
  AffineForm(const T CentralValue, const T NoiseMagnitude)
      : X0(CentralValue), Xi(1, NoiseTerm<T>(NoiseMagnitude)) {}

  /// Construct an AffineForm with value CentralValue,
  /// and the noise terms contained in NoiseTerms,
  /// which must be sorted.
  AffineForm(const T CentralValue,
             llvm::SmallVectorImpl<NoiseTerm<T>> &&NoiseTerms)
      : X0(CentralValue), Xi(std::move(NoiseTerms))
  {

    assert(std::is_sorted(Xi.begin(), Xi.end()) && "NoiseTerm Ids must be sorted.");
  }

  /// Construct an AffineForm with value CentralValue,
  /// and the noise terms contained in NoiseTerms,
  /// after sorting them.
  AffineForm(const T CentralValue, const llvm::ArrayRef<NoiseTerm<T>> &NoiseTerms)
      : X0(CentralValue), Xi(NoiseTerms.begin(), NoiseTerms.end())
  {
    std::sort(Xi.begin(), Xi.end());
  }

  ///
  /// Construct an AffineForm by converting an Interval.
  /// The central value is the average between the interval bounds,
  /// and a noise term equal to the distance between the central value
  /// and the bounds is introduced.
  ///
  explicit AffineForm(const Interval<T> &Range)
  {
    X0 = (Range.Min + Range.Max) / 2;
    Xi.push_back(NoiseTerm<T>(Range.Max - X0));
  }

  ///
  /// Converts this AffineForm into an Interval.
  ///
  Interval<T> toInterval() const
  {
    T Rad = this->noiseTermsAbsSum();
    return Interval<T>(X0 - Rad, X0 + Rad);
  }

  T getCentralValue() const
  {
    return this->X0;
  }

  void setCentralValue(T NX0)
  {
    this->X0 = NX0;
  }

  T noiseTermsAbsSum() const
  {
    T Rad = 0;
    for (const NoiseTerm<T> &NT : Xi) {
      Rad += std::abs(NT.Magnitude);
    }

    return Rad;
  }

  /// Return an AffineForm with the same central value
  /// and a single noise term which is the sum of the old ones.
  AffineForm<T> flattenNoiseTerms() const
  {
    return AffineForm<T>(this->X0, this->noiseTermsAbsSum());
  }

  AffineForm<T> &operator+=(const AffineForm<T> &O)
  {
    this->X0 = this->X0 + O.X0;
    this->Xi = mergeSumNoiseTerms(O);
    return *this;
  }

  AffineForm<T> operator+(const AffineForm<T> &O) const
  {
    return AffineForm<T>(*this) += O;
  }

  AffineForm<T> operator-() const
  {
    llvm::SmallVector<NoiseTerm<T>, DEFAULT_NOISE_SIZE> NXi;
    NXi.reserve(Xi.size());

    for (const NoiseTerm<T> &NT : Xi) {
      // Change sign to all noise terms
      NXi.push_back(-NT);
    }
    // Change sign to central value
    return AffineForm<T>(-this->X0, std::move(NXi));
  }

  AffineForm<T> &operator-=(const AffineForm<T> &O)
  {
    return *this += -O;
  }

  AffineForm<T> operator-(const AffineForm<T> &O) const
  {
    return *this + (-O);
  }

  AffineForm<T> &operator*=(const AffineForm<T> &O)
  {
    // Compute new noise terms as (x0*yi + y0*xi)*epsi
    // and add approximation error if necessary
    this->Xi = mergeMulNoiseTerms(O);

    // Product of central values
    this->X0 *= O.X0;

    return *this;
  }

  AffineForm<T> operator*(const AffineForm<T> &O) const
  {
    return AffineForm<T>(*this) *= O;
  }

  AffineForm<T> &operator/=(const AffineForm<T> &O)
  {
    return *this *= O.inverse();
  }

  AffineForm<T> operator/(const AffineForm<T> &O) const
  {
    return AffineForm<T>(*this) /= O;
  }

  /// Compute 1/this.
  AffineForm<T> inverse(bool ErrorOnly = false) const
  {
    if (Xi.empty()) {
      // No noise terms
      return AffineForm<T>(1 / X0);
    }

    // If there are noise terms, we compute a linear approximation
    // similar to the implementation of rosa by Darulova et al.

    // We have f(x) = f(xu) + f'(xu)(x - xu)
    // or      f(x) = f(xl) + f'(xl)(x - xl)
    // with f(x) = 1/x
    // but we use f'(xu) and we take the average of the rest.

    Interval<T> X = this->toInterval();

    T xl = X.Min;
    T xu = X.Max;

    // First derivative of 1/x computed in xu (f'(xu))
    T d1ox = static_cast<T>(-1) / (xu * xu);

    T rmin = (static_cast<T>(1) / xl) - d1ox * xl;
    T rmax = (static_cast<T>(1) / xu) - d1ox * xu;
    T ravg = (rmin + rmax) / static_cast<T>(2);
    // New central value
    T NX0 = ravg + d1ox * this->X0;

    // New noise terms
    noiseContainer NXi;
    NXi.reserve(this->Xi.size() + !ErrorOnly);
    // Multiply old noise terms by f'(x)
    for (const NoiseTerm<T> &NT : this->Xi) {
      NXi.push_back(NoiseTerm<T>(NT.Symbol, NT.Magnitude * d1ox));
    }

    if (!ErrorOnly) {
      T error = (rmax - rmin) / static_cast<T>(2);
      NXi.push_back(NoiseTerm<T>(error));
    }

    return AffineForm<T>(NX0, std::move(NXi));
  }

  AffineForm<T> scalarMultiply(T x) const
  {
    noiseContainer NXi;
    NXi.reserve(this->Xi.size());
    for (const NoiseTerm<T> &NT : this->Xi) {
      NXi.push_back(NoiseTerm<T>(NT.Symbol, NT.Magnitude * x));
    }
    return AffineForm<T>(this->X0 * x, std::move(NXi));
  }

protected:
  typedef llvm::SmallVector<NoiseTerm<T>, DEFAULT_NOISE_SIZE> noiseContainer;

  T X0;              ///< Central value.
  noiseContainer Xi; ///< Noise terms.

  /// Merge the noise terms of this and O by summing those that are matching.
  noiseContainer mergeSumNoiseTerms(const AffineForm<T> &O) const
  {
    assert(std::is_sorted(this->Xi.begin(), this->Xi.end()) && "NoiseTerm Ids must be sorted.");
    assert(std::is_sorted(O.Xi.begin(), O.Xi.end()) && "NoiseTerm Ids must be sorted.");

    noiseContainer NXi;
    NXi.reserve(this->Xi.size() + O.Xi.size());

    auto ThisIt = this->Xi.begin();
    auto ThisItEnd = this->Xi.end();

    auto OtherIt = O.Xi.begin();
    auto OtherItEnd = O.Xi.end();

    while (ThisIt != ThisItEnd && OtherIt != OtherItEnd) {
      const NoiseTerm<T> &TTerm = *ThisIt;
      const NoiseTerm<T> &OTerm = *OtherIt;
      if (TTerm < OTerm) {
        NXi.push_back(TTerm);
        ++ThisIt;
      } else if (OTerm < TTerm) {
        NXi.push_back(OTerm);
        ++OtherIt;
      } else {
        NXi.push_back(TTerm + OTerm);
        ++ThisIt;
        ++OtherIt;
      }
    }

    if (ThisIt != ThisItEnd) {
      assert(OtherIt == OtherItEnd && "All Other's NoiseTerms must have been processed.");
      NXi.append(ThisIt, ThisItEnd);
    } else if (OtherIt != OtherItEnd) {
      assert(ThisIt == ThisItEnd && "All NoiseTerms of this must have been processed.");
      NXi.append(OtherIt, OtherItEnd);
    }

    assert(std::is_sorted(NXi.begin(), NXi.end()) && "NoiseTerm Ids must be sorted.");

    return std::move(NXi);
  }

  /// Merge the noise terms of this and O by summing those that are matching,
  /// after having multiplied those of this by the central value of O
  /// and vice versa (part of the implementation of multiplication).
  noiseContainer mergeMulNoiseTerms(const AffineForm<T> &O) const
  {
    assert(std::is_sorted(this->Xi.begin(), this->Xi.end()) && "NoiseTerm Ids must be sorted.");
    assert(std::is_sorted(O.Xi.begin(), O.Xi.end()) && "NoiseTerm Ids must be sorted.");

    noiseContainer NXi;
    NXi.reserve(this->Xi.size() + O.Xi.size());

    auto ThisIt = this->Xi.begin();
    auto ThisItEnd = this->Xi.end();

    auto OtherIt = O.Xi.begin();
    auto OtherItEnd = O.Xi.end();

    while (ThisIt != ThisItEnd && OtherIt != OtherItEnd) {
      const NoiseTerm<T> &TTerm = *ThisIt;
      const NoiseTerm<T> &OTerm = *OtherIt;
      NoiseTerm<T> NewTerm(TTerm);
      if (TTerm < OTerm) {
        NewTerm.Magnitude = O.X0 * TTerm.Magnitude;
        NXi.push_back(NewTerm);
        ++ThisIt;
      } else if (OTerm < TTerm) {
        NewTerm.Magnitude = this->X0 * OTerm.Magnitude;
        NXi.push_back(NewTerm);
        ++OtherIt;
      } else {
        NewTerm.Magnitude = O.X0 * TTerm.Magnitude + this->X0 * OTerm.Magnitude;
        NXi.push_back(NewTerm);
        ++ThisIt;
        ++OtherIt;
      }
    }

    // We must now process all remaining noise terms
    // in the AffineForm that is not empty.
    auto RestIt = ThisIt;
    auto RestItEnd = ThisItEnd;
    T RX0 = O.X0;
    if (OtherIt != OtherItEnd) {
      assert(ThisIt == ThisItEnd && "All NoiseTerms of this must have been processed.");
      RestIt = OtherIt;
      RestItEnd = OtherItEnd;
      RX0 = this->X0;
    }

    for (; RestIt != RestItEnd; ++RestIt) {
      const NoiseTerm<T> RTerm(*RestIt);
      NXi.push_back(NoiseTerm<T>(RTerm.Symbol, RTerm.Magnitude * RX0));
    }

    // Add approximation error, if non-zero
    if (!this->Xi.empty() && !O.Xi.empty()) {
      NXi.push_back(NoiseTerm<T>(this->noiseTermsAbsSum() * O.noiseTermsAbsSum()));
    }

    assert(std::is_sorted(NXi.begin(), NXi.end()) && "NoiseTerm Ids must be sorted.");

    return std::move(NXi);
  }
};

/// Compute the errors of a variable with range R and errors E
/// after being passed as a parameter to function F, whose derivative is dF.
/// This variant maximizes decreasing derivatives.
template <typename T, typename FunDer>
AffineForm<T>
LinearErrorApproximationDecr(FunDer dF, const Interval<T> &R, const AffineForm<T> &E)
{
  T X = std::min(R.Min, R.Max);
  T dFx = dF(X);

  LLVM_DEBUG(llvm::dbgs() << "(R = [" << static_cast<double>(R.Min)
                          << ", " << static_cast<double>(R.Max)
                          << "], dFx = " << static_cast<double>(dFx)
                          << ", E = " << static_cast<double>(E.noiseTermsAbsSum())
                          << ") ");
  return E.scalarMultiply(dFx);
}

/// Compute the errors of a variable with range R and errors E
/// after being passed as a parameter to function F, whose derivative is dF.
/// This variant maximizes increasing derivatives.
template <typename T, typename FunDer>
AffineForm<T>
LinearErrorApproximationIncr(FunDer dF, const Interval<T> &R, const AffineForm<T> &E)
{
  T X = std::max(R.Min, R.Max);
  T dFx = dF(X);

  LLVM_DEBUG(llvm::dbgs() << "(R = [" << static_cast<double>(R.Min)
                          << ", " << static_cast<double>(R.Max)
                          << "], dFx = " << static_cast<double>(dFx) << ") ");
  return E.scalarMultiply(dFx);
}


} // end namespace ErrorProp

#endif // ERRORPROPAGATOR_AFFINE_FORMS_H
