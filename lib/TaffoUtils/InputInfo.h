//===-- InputInfo.h - Data Structures for Input Info Metadata ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Various data structures that support in-memory representation of
/// input info metadata.
///
//===----------------------------------------------------------------------===//

#ifndef TAFFO_INPUT_INFO_H
#define TAFFO_INPUT_INFO_H

#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <memory>
#include <sstream>

namespace mdutils
{

#define FIXP_TYPE_FLAG "fixp"
#define FLOAT_TYPE_FLAG "float"

/// Info about a data type for numerical computations.
class TType
{
public:
  enum TTypeKind {
    K_FPType,
    K_FloatType
  };

  TType(TTypeKind K) : Kind(K) {}

  virtual double getRoundingError() const = 0;

  /// Safe approximation of the minimum value representable with this Type.
  virtual llvm::APFloat getMinValueBound() const = 0;

  /// Safe approximation of the maximum value representable with this Type.
  virtual llvm::APFloat getMaxValueBound() const = 0;

  virtual llvm::MDNode *toMetadata(llvm::LLVMContext &C) const = 0;

  virtual TType *clone() const = 0;

  virtual ~TType() = default;

  static std::unique_ptr<TType> createFromMetadata(llvm::MDNode *MDN);

  static bool isTTypeMetadata(llvm::Metadata *MD);

  virtual std::string toString() const
  {
    return "TType";
  };

  virtual bool operator==(const TType &b) const
  {
    return Kind == b.Kind;
  }

  TTypeKind getKind() const { return Kind; }

private:
  const TTypeKind Kind;
};

/// A Fixed Point Type.
/// Contains bit width, number of fractional bits of the format
/// and whether it is signed or not.
class FPType : public TType
{
public:
  FPType(unsigned Width, unsigned PointPos, bool Signed = true)
      : TType(K_FPType), Width((Signed) ? -Width : Width), PointPos(PointPos) {}

  FPType(int Width, unsigned PointPos)
      : TType(K_FPType), Width(Width), PointPos(PointPos) {}

  double getRoundingError() const override;

  llvm::APFloat getMinValueBound() const override;

  llvm::APFloat getMaxValueBound() const override;

  llvm::MDNode *toMetadata(llvm::LLVMContext &C) const override;

  unsigned getWidth() const { return std::abs(Width); }

  int getSWidth() const { return Width; }

  unsigned getPointPos() const { return PointPos; }

  bool isSigned() const { return Width < 0; }

  virtual TType *clone() const override
  {
    return new FPType(Width, PointPos);
  };

  static bool isFPTypeMetadata(llvm::MDNode *MDN);

  static std::unique_ptr<FPType> createFromMetadata(llvm::MDNode *MDN);

  virtual std::string toString() const override
  {
    std::stringstream stm;
    if (Width < 0)
      stm << "s";
    else
      stm << "u";
    stm << std::abs(Width) - PointPos << "_" << PointPos << "fixp";
    return stm.str();
  };

  virtual bool operator==(const TType &b) const override
  {
    if (!TType::operator==(b))
      return false;
    const FPType *b2 = llvm::cast<FPType>(&b);
    return Width == b2->Width && PointPos == b2->PointPos;
  }

  static bool classof(const TType *T) { return T->getKind() == K_FPType; }

protected:
  int Width;         ///< Width of the format (in bits), negative if signed.
  unsigned PointPos; ///< Number of fractional bits.
};


/// A Floating Point Type.
/// Contains the particular type of floating point used, that must be supported by LLVM
class FloatType : public TType
{
public:
  enum FloatStandard {
    Float_half = 0,  /*16-bit floating-point value*/
    Float_float,     /*32-bit floating-point value*/
    Float_double,    /*64-bit floating-point value*/
    Float_fp128,     /*128-bit floating-point value (112-bit mantissa)*/
    Float_x86_fp80,  /*80-bit floating-point value (X87)*/
    Float_ppc_fp128, /*128-bit floating-point value (two 64-bits)*/
    Float_bfloat
  };

  static const llvm::Type::TypeID llvmFloatTypes[7];

  static std::string getFloatStandardName(FloatStandard standard);

  static std::string getFloatStandardName(llvm::Type::TypeID typeId);

  static llvm::Type::TypeID getLLVMTypeID(FloatType::FloatStandard);

  static FloatType::FloatStandard getFloatStandard(llvm::Type::TypeID typeId);

  static FloatType::FloatStandard getFloatStandard(const std::string& typeIdStr);

  FloatType(FloatStandard standard, double greatestNumber)
      : TType(K_FloatType), standard(standard), greatestNunber(greatestNumber) {}
  FloatType(llvm::Type::TypeID TyId, double greatestNumber)
      : TType(K_FloatType), greatestNunber(greatestNumber) {
    switch (TyId) {
      case llvm::Type::HalfTyID: standard = Float_half; break;
      case llvm::Type::FloatTyID: standard = Float_float; break;
      case llvm::Type::DoubleTyID: standard = Float_double; break;
      case llvm::Type::FP128TyID: standard = Float_fp128; break;
      case llvm::Type::X86_FP80TyID: standard = Float_x86_fp80; break;
      case llvm::Type::PPC_FP128TyID: standard = Float_ppc_fp128; break;
      case llvm::Type::BFloatTyID: standard = Float_bfloat; break;
      default:
        llvm_unreachable("invalid type id for FloatType's constructor");
    }
  }

  double getRoundingError() const override;

  llvm::APFloat getMinValueBound() const override;

  llvm::APFloat getMaxValueBound() const override;

  int getP() const;

  llvm::Type::TypeID getLLVMTypeID() const;

  llvm::MDNode *toMetadata(llvm::LLVMContext &C) const override;

  virtual TType *clone() const override
  {
    return new FloatType(standard, greatestNunber);
  };

  static bool isFloatTypeMetadata(llvm::MDNode *MDN);

  static std::unique_ptr<FloatType> createFromMetadata(llvm::MDNode *MDN);

  std::string toString() const override
  {
    std::stringstream stm;
    stm << getFloatStandardName(standard);
    stm << "_float";
    return stm.str();
  };

  virtual bool operator==(const TType &b) const override
  {
    if (!TType::operator==(b))
      return false;
    const auto *b2 = llvm::cast<FloatType>(&b);
    return standard == b2->getStandard();
  }

  static bool classof(const TType *T) { return T->getKind() == K_FloatType; }

  FloatStandard getStandard() const
  {
    return standard;
  }

  double getGreatestNumber() const
  {
    return greatestNunber;
  }

protected:
  FloatStandard standard;

  // This is only used to understand the maximum error that this type can generate
  // As during the DTA pass we assign each Type looking at its range, it is "free" (as in free beer)
  double greatestNunber;
};

struct Range {
public:
  double Min;
  double Max;

  Range() : Min(0.0), Max(0.0) {}

  Range(double Min, double Max) : Min(Min), Max(Max) {}

  Range(Range &r) : Min(r.Min), Max(r.Max) {}

  std::string toString() const
  {
    std::stringstream sstm;
    sstm << "[" << Min << ", " << Max << "]";
    return sstm.str();
  }

  llvm::MDNode *toMetadata(llvm::LLVMContext &C) const;

  static std::unique_ptr<Range> createFromMetadata(llvm::MDNode *MDN);

  static bool isRangeMetadata(llvm::Metadata *MD);
};

std::unique_ptr<double> CreateInitialErrorFromMetadata(llvm::MDNode *MDN);

llvm::MDNode *InitialErrorToMetadata(double Error);

bool IsInitialErrorMetadata(llvm::Metadata *MD);

class MDInfo
{
public:
  enum MDInfoKind {
    K_Struct,
    K_Field
  };

  MDInfo(MDInfoKind K) : Kind(K) {}

  virtual llvm::MDNode *toMetadata(llvm::LLVMContext &C) const = 0;

  virtual MDInfo *clone() const = 0;

  virtual ~MDInfo() = default;

  MDInfoKind getKind() const { return Kind; }

  virtual std::string toString() const
  {
    return "MDInfo";
  };

  virtual bool getEnableConversion() const = 0;

private:
  const MDInfoKind Kind;
};

/// Structure containing pointers to Type, Range, and initial Error
/// of an LLVM Value.
struct InputInfo : public MDInfo {
  std::shared_ptr<TType> IType;
  std::shared_ptr<Range> IRange;
  std::shared_ptr<double> IError;
  bool IEnableConversion;
  bool IFinal;

  InputInfo()
      : MDInfo(K_Field), IType(nullptr), IRange(nullptr), IError(nullptr), IEnableConversion(false),
        IFinal(false) {}

  InputInfo(std::shared_ptr<TType> T, std::shared_ptr<Range> R, std::shared_ptr<double> Error)
      : MDInfo(K_Field), IType(T), IRange(R), IError(Error), IEnableConversion(false), IFinal(false) {}

  InputInfo(std::shared_ptr<TType> T, std::shared_ptr<Range> R, std::shared_ptr<double> Error, bool EnC,
            bool IsFinal = false)
      : MDInfo(K_Field), IType(T), IRange(R), IError(Error), IEnableConversion(EnC), IFinal(IsFinal) {}

  InputInfo(InputInfo& II) = default;

  virtual MDInfo *clone() const override
  {
    std::shared_ptr<TType> NewIType(IType.get() ? IType->clone() : nullptr);
    std::shared_ptr<Range> NewIRange(IRange.get() ? new Range(*IRange) : nullptr);
    std::shared_ptr<double> NewIError(IError.get() ? new double(*IError) : nullptr);
    return new InputInfo(NewIType, NewIRange, NewIError, IEnableConversion, IFinal);
  }

  llvm::MDNode *toMetadata(llvm::LLVMContext &C) const override;

  static bool isInputInfoMetadata(llvm::Metadata *MD);

  InputInfo &operator=(const InputInfo &O)
  {
    assert(this->getKind() == O.getKind());
    this->IType = O.IType;
    this->IRange = O.IRange;
    this->IError = O.IError;
    this->IEnableConversion = O.IEnableConversion;
    this->IFinal = O.IFinal;
    return *this;
  };

  virtual std::string toString() const override
  {
    std::stringstream sstm;
    sstm << "scalar(";
    bool first = true;
    if (IType.get()) {
      first = false;
      sstm << "type(" << IType->toString() << ")";
    }
    if (IRange.get()) {
      if (!first)
        sstm << " ";
      else
        first = false;
      sstm << "range(" << IRange->Min << ", " << IRange->Max << ")";
    }
    if (IError.get()) {
      if (!first)
        sstm << " ";
      sstm << "error(" << *IError << ")";
    }
    if (!IEnableConversion) {
      if (!first)
        sstm << " ";
      sstm << "disabled";
    }
    if (IFinal) {
      if (!first)
        sstm << " ";
      sstm << "final";
    }
    sstm << ")";
    return sstm.str();
  };

  bool getEnableConversion() const override
  {
    return IEnableConversion;
  };

  bool isFinal() const { return IFinal; }

  static bool classof(const MDInfo *M) { return M->getKind() == K_Field; }
};

class StructInfo : public MDInfo
{
private:
  typedef llvm::SmallVector<std::shared_ptr<MDInfo>, 4U> FieldsType;
  FieldsType Fields;

  bool _getEnableConversion(llvm::SmallPtrSetImpl<const StructInfo *> &visited) const
  {
    visited.insert(this);
    for (auto field : Fields) {
      if (!field.get())
        continue;
      if (StructInfo *si = llvm::dyn_cast<StructInfo>(field.get())) {
        if (visited.count(si) > 0)
          continue;
        if (!si->_getEnableConversion(visited))
          return false;
      } else {
        if (!field->getEnableConversion())
          return false;
      }
    }
    return true;
  };

public:
  typedef FieldsType::iterator iterator;
  typedef FieldsType::const_iterator const_iterator;
  typedef FieldsType::size_type size_type;

  StructInfo(int size)
      : MDInfo(K_Struct), Fields(size, nullptr) {}

  StructInfo(const llvm::ArrayRef<std::shared_ptr<MDInfo>> SInfos)
      : MDInfo(K_Struct), Fields(SInfos.begin(), SInfos.end()) {}

  iterator begin() { return Fields.begin(); }

  iterator end() { return Fields.end(); }

  const_iterator begin() const { return Fields.begin(); }

  const_iterator end() const { return Fields.end(); }

  size_type size() const { return Fields.size(); }

  MDInfo *getField(size_type I) const { return Fields[I].get(); }

  void setField(size_type I, std::shared_ptr<MDInfo> F) { Fields[I] = F; }

  std::shared_ptr<MDInfo> getField(size_type I) { return Fields[I]; }

  /** Builds a StructInfo with the recursive structure of the specified
   *  LLVM Type. All non-struct struct members are set to nullptr.
   *  @returns Either a StructInfo, or nullptr if the type does not
   *    contain any structure. */
  static std::shared_ptr<StructInfo> constructFromLLVMType(llvm::Type *t,
                                                           llvm::SmallDenseMap<llvm::Type *, std::shared_ptr<StructInfo>> *recursionMap = nullptr)
  {
    std::unique_ptr<llvm::SmallDenseMap<llvm::Type *, std::shared_ptr<StructInfo>>> _recursionMap;
    if (!recursionMap) {
      _recursionMap.reset(new llvm::SmallDenseMap<llvm::Type *, std::shared_ptr<StructInfo>>());
      recursionMap = _recursionMap.get();
    }

    auto rec = recursionMap->find(t);
    if (rec != recursionMap->end()) {
      return rec->getSecond();
    }

    int c = t->getNumContainedTypes();

    if (c == 0 || t->isFunctionTy()) {
      recursionMap->insert({t, nullptr});
      return nullptr;
    }

    if (t->isStructTy()) {
      FieldsType fields;
      std::shared_ptr<StructInfo> res = std::make_shared<StructInfo>(StructInfo(c));
      recursionMap->insert({t, res});
      for (int i = 0; i < c; i++) {
        res->getField(i) = StructInfo::constructFromLLVMType(t->getContainedType(i), recursionMap);
      }
      return res;
    }

    return StructInfo::constructFromLLVMType(t->getContainedType(0), recursionMap);
  }

  std::shared_ptr<MDInfo> resolveFromIndexList(llvm::Type *type, llvm::ArrayRef<unsigned> indices)
  {
    llvm::Type *resolvedType = type;
    std::shared_ptr<MDInfo> resolvedInfo(this);
    for (unsigned idx : indices) {
      if (resolvedInfo.get() == nullptr)
        break;
      if (resolvedType->isStructTy()) {
        resolvedType = resolvedType->getContainedType(idx);
        resolvedInfo = llvm::cast<StructInfo>(resolvedInfo.get())->getField(idx);
      } else {
        resolvedType = resolvedType->getContainedType(idx);
      }
    }
    return resolvedInfo;
  }

  virtual MDInfo *clone() const override
  {
    FieldsType newFields;
    for (std::shared_ptr<MDInfo> oldF : Fields) {
      if (oldF.get())
        newFields.push_back(std::shared_ptr<MDInfo>(oldF->clone()));
      else
        newFields.push_back(std::shared_ptr<MDInfo>(nullptr));
    }
    return new StructInfo(newFields);
  }

  virtual std::string toString() const override
  {
    std::stringstream sstm;
    sstm << "struct(";
    bool first = true;
    for (std::shared_ptr<MDInfo> i : Fields) {
      if (!first)
        sstm << ", ";
      if (i.get()) {
        sstm << i->toString();
      } else {
        sstm << "void()";
      }
      first = false;
    }
    sstm << ")";
    if (!getEnableConversion()) {
      if (!first)
        sstm << " ";
      sstm << "disabled ";
    }
    return sstm.str();
  };

  bool getEnableConversion() const override
  {
    llvm::SmallPtrSet<const StructInfo *, 1> visited;
    return _getEnableConversion(visited);
  };

  llvm::MDNode *toMetadata(llvm::LLVMContext &C) const override;

  static bool classof(const MDInfo *M) { return M->getKind() == K_Struct; }
};


/// Struct containing info about a possible comparison error.
struct CmpErrorInfo {
public:
  double MaxTolerance; ///< Maximum error tolerance for this comparison.
  bool MayBeWrong;     ///< True if this comparison may be wrong due to propagated errors.

  CmpErrorInfo(double MaxTolerance, bool MayBeWrong = true)
      : MaxTolerance(MaxTolerance), MayBeWrong(MayBeWrong) {}

  llvm::MDNode *toMetadata(llvm::LLVMContext &C) const;

  static std::unique_ptr<CmpErrorInfo> createFromMetadata(llvm::MDNode *MDN);
};


bool IsNullInputInfoField(llvm::Metadata *MD);

llvm::MDNode *createDoubleMDNode(llvm::LLVMContext &C, double Value);

double retrieveDoubleMDNode(llvm::MDNode *MDN);

} // end namespace mdutils

#endif
