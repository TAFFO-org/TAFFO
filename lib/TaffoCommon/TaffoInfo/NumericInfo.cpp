#include "NumericInfo.hpp"

using namespace llvm;
using namespace taffo;

double FixedPointInfo::getRoundingError() const { return std::ldexp(1.0, -this->getFractionalBits()); }

APFloat FixedPointInfo::getMinValueBound() const {
  if (isSigned())
    return APFloat(std::ldexp(-1.0, getBits() - getFractionalBits() - 1));
  return APFloat(0.0);
}

APFloat FixedPointInfo::getMaxValueBound() const {
  int MaxIntExp = isSigned() ? getBits() - 1 : getBits();
  double MaxIntPlus1 = std::ldexp(1.0, MaxIntExp);
  double MaxInt = MaxIntPlus1 - 1.0;
  if (MaxInt == MaxIntPlus1)
    MaxInt = std::nextafter(MaxInt, 0.0);
  return APFloat(std::ldexp(MaxInt, -getFractionalBits()));
}

bool FixedPointInfo::operator==(const NumericTypeInfo& other) const {
  if (!NumericTypeInfo::operator==(other))
    return false;
  const auto* otherFixpInfo = cast<FixedPointInfo>(&other);
  return sign == otherFixpInfo->sign && bits == otherFixpInfo->bits && fractionalBits == otherFixpInfo->fractionalBits;
}

std::shared_ptr<NumericTypeInfo> FixedPointInfo::clone() const {
  return std::make_shared<FixedPointInfo>(sign, bits, fractionalBits);
}

std::string FixedPointInfo::toString() const {
  std::stringstream ss;
  ss << (sign ? "s" : "u") << bits - fractionalBits << "_" << fractionalBits << "fixp";
  return ss.str();
}

json FixedPointInfo::serialize() const {
  json j;
  j["kind"] = "FixedPoint";
  j["signed"] = sign;
  j["bits"] = bits;
  j["fractionalBits"] = fractionalBits;
  return j;
}

void FixedPointInfo::deserialize(const json& j) {
  assert(j["kind"] == "FixedPoint");
  sign = j["signed"].get<bool>();
  bits = j["bits"].get<int>();
  fractionalBits = j["fractionalBits"].get<unsigned>();
}

std::string FloatingPointInfo::getFloatStandardName(FloatStandard standard) {
  switch (standard) {
  case Float_half: /*16-bit floating-point value*/ return "Float_half";
  case Float_float: /*32-bit floating-point value*/ return "Float_float";
  case Float_double: /*64-bit floating-point value*/ return "Float_double";
  case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/ return "Float_fp128";
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/ return "Float_x86_fp80";
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/ return "Float_ppc_fp128";
  case Float_bfloat: /*bfloat floating point value)*/ return "Float_bfloat";
  }
  llvm_unreachable("[TAFFO] Unknown FloatType standard!");
}

// FIXME: this can give incorrect results if used in corner cases
double FloatingPointInfo::getRoundingError() const {
  int p = getP();

  // Computing the exponent value
  double k = floor(log2(this->greatestNumber));

  // given that epsilon is the maximum error achievable given a certain amount of bit in mantissa (p) on the mantissa
  // itself it will be multiplied by the exponent, that will be at most 2^k BTW we are probably carrying some type of
  // error here Hehehe Complete formula -> epsilon * exponent_value that is (beta/2)*(b^-p)     *     b^k thus (beta/2)
  // b*(k-p) given beta = 2 on binary machines (so I hope the target one is binary too...)
  return exp2(k - p);
}

// FIXME: some values are not computed correctly because we cannot!
APFloat FloatingPointInfo::getMinValueBound() const {
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/ return APFloat::getLargest(APFloat::IEEEhalf(), true);
  case Float_float: /*32-bit floating-point value*/ return APFloat::getLargest(APFloat::IEEEsingle(), true);
  case Float_double: /*64-bit floating-point value*/ return APFloat::getLargest(APFloat::IEEEdouble(), true);
  case Float_fp128:     /*128-bit floating-point value (112-bit mantissa)*/
    return APFloat::getLargest(APFloat::IEEEquad(), true);
  case Float_x86_fp80:  /*80-bit floating-point value (X87)*/
    return APFloat::getLargest(APFloat::x87DoubleExtended(), true);
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    return APFloat::getLargest(APFloat::PPCDoubleDouble(), true);
  case Float_bfloat: /*bfloat floating point value)*/ return APFloat::getLargest(APFloat::BFloat(), true);
  }
  llvm_unreachable("[TAFFO] Unknown FloatType standard!");
}

// FIXME: some values are not computed correctly because we cannot!
APFloat FloatingPointInfo::getMaxValueBound() const {
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/ return APFloat::getLargest(APFloat::IEEEhalf(), false);
  case Float_float: /*32-bit floating-point value*/ return APFloat::getLargest(APFloat::IEEEsingle(), false);
  case Float_double: /*64-bit floating-point value*/ return APFloat::getLargest(APFloat::IEEEdouble(), false);
  case Float_fp128:     /*128-bit floating-point value (112-bit mantissa)*/
    return APFloat::getLargest(APFloat::IEEEquad(), false);
  case Float_x86_fp80:  /*80-bit floating-point value (X87)*/
    return APFloat::getLargest(APFloat::x87DoubleExtended(), false);
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    return APFloat::getLargest(APFloat::PPCDoubleDouble(), false);
  case Float_bfloat: /*bfloat floating point value)*/ return APFloat::getLargest(APFloat::BFloat(), false);
  }

  llvm_unreachable("Unknown limit for this float type");
}

// This function will return the number of bits in the mantissa
int FloatingPointInfo::getP() const {
  // The plus 1 is due to the fact that there is always an implicit 1 stored (the d_0 value)
  // Therefore, we have actually 1 bit more wrt the ones stored
  int p;
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/ p = APFloat::semanticsPrecision(APFloat::IEEEhalf()); break;
  case Float_float: /*32-bit floating-point value*/ p = APFloat::semanticsPrecision(APFloat::IEEEsingle()); break;
  case Float_double: /*64-bit floating-point value*/ p = APFloat::semanticsPrecision(APFloat::IEEEdouble()); break;
  case Float_fp128:    /*128-bit floating-point value (112-bit mantissa)*/
    p = APFloat::semanticsPrecision(APFloat::IEEEquad());
    break;
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/
    // But in this case, it has a fractional part of 63 and an "integer" part of 1, total 64 for the significand
    p = APFloat::semanticsPrecision(APFloat::x87DoubleExtended());
    break;
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/
    p = APFloat::semanticsPrecision(APFloat::PPCDoubleDouble());
    break;
  case Float_bfloat:    /*128-bit floating-point value (two 64-bits)*/
    p = APFloat::semanticsPrecision(APFloat::BFloat());
    break;
  }
  return p;
}

Type::TypeID FloatingPointInfo::getLLVMTypeID() const {
  switch (this->getStandard()) {
  case Float_half: /*16-bit floating-point value*/ return Type::TypeID::HalfTyID;
  case Float_float: /*32-bit floating-point value*/ return Type::TypeID::FloatTyID;
  case Float_double: /*64-bit floating-point value*/ return Type::TypeID::DoubleTyID;
  case Float_fp128: /*128-bit floating-point value (112-bit mantissa)*/ return Type::TypeID::FP128TyID;
  case Float_x86_fp80: /*80-bit floating-point value (X87)*/ return Type::TypeID::X86_FP80TyID;
  case Float_ppc_fp128: /*128-bit floating-point value (two 64-bits)*/ return Type::TypeID::PPC_FP128TyID;
  case Float_bfloat: /*bfloat floating point value)*/ return Type::TypeID::BFloatTyID;
  }
  llvm_unreachable("[TAFFO] Unknown FloatType standard!");
}

bool FloatingPointInfo::operator==(const NumericTypeInfo& other) const {
  if (!NumericTypeInfo::operator==(other))
    return false;
  const auto* b2 = cast<FloatingPointInfo>(&other);
  return standard == b2->getStandard();
}

std::shared_ptr<NumericTypeInfo> FloatingPointInfo::clone() const {
  return std::make_shared<FloatingPointInfo>(standard, greatestNumber);
}

std::string FloatingPointInfo::toString() const {
  std::stringstream ss;
  ss << getFloatStandardName(standard);
  ss << "_float";
  return ss.str();
}

json FloatingPointInfo::serialize() const {
  json j;
  j["kind"] = "FloatingPoint";
  j["standard"] = standard;
  j["greatestNumber"] = serializeDouble(greatestNumber);
  return j;
}

void FloatingPointInfo::deserialize(const json& j) {
  assert(j["kind"] == "FloatingPoint");
  standard = static_cast<FloatStandard>(j["standard"].get<int>());
  greatestNumber = deserializeDouble(j["greatestNumber"]);
}
