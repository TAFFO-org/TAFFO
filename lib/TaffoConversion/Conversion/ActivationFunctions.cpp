#include "../ConversionPass.hpp"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using namespace tda;
using namespace taffo;

namespace {

constexpr unsigned LUT_BITS = 10;
constexpr unsigned LUT_SIZE = 1u << LUT_BITS;

using ActivationEvaluator = long double (*)(long double);

/*
 * Recupera il FixedPointInfo prodotto dalle analisi TAFFO.
 *
 * Non usare std::dynamic_pointer_cast: TAFFO viene compilato
 * con -fno-rtti.
 *
 * std::dynamic_ptr_cast è l'utility utilizzata internamente
 * dal progetto.
 */
static std::shared_ptr<FixedPointInfo> getFixedPointInfoForValue(TaffoInfo& taffoInfo, Value* value) {
  if (!value)
    return nullptr;

  if (!taffoInfo.hasValueInfo(*value))
    return nullptr;

  std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*value));

  if (!scalarInfo)
    return nullptr;

  if (!scalarInfo->numericType)
    return nullptr;

  return std::dynamic_ptr_cast<FixedPointInfo>(scalarInfo->numericType);
}

/*
 * Costruisce un ConversionScalarType fixed-point a partire
 * dal FixedPointInfo prodotto dalla DTA.
 *
 * infoValue:
 *   valore da cui recuperare il FixedPointInfo.
 *
 * transparentValue:
 *   valore da cui recuperare la struttura del tipo originario.
 */
static std::unique_ptr<ConversionScalarType>
makeFixedConversionType(TaffoInfo& taffoInfo, Value* infoValue, Value* transparentValue) {
  std::shared_ptr<FixedPointInfo> fixedInfo = getFixedPointInfoForValue(taffoInfo, infoValue);

  if (!fixedInfo || !transparentValue)
    return nullptr;

  TransparentType* transparentType = taffoInfo.getOrCreateTransparentType(*transparentValue);

  if (!transparentType)
    return nullptr;

  auto result = std::make_unique<ConversionScalarType>(*transparentType, fixedInfo.get());

  if (!result->isFixedPoint())
    return nullptr;

  return result;
}

/*
 * Codifica un tipo fixed-point nel nome della LUT.
 *
 * Esempi:
 *
 *   signed, 32 bit, frac=30    -> s32_fp30
 *   unsigned, 16 bit, frac=15  -> u16_fp15
 *   signed, 32 bit, frac=-2    -> s32_fm2
 */
static std::string encodeFixedType(const ConversionScalarType& type) {
  std::string result = type.isSigned() ? "s" : "u";

  result += std::to_string(type.getBits());

  const int fractionalBits = type.getFractionalBits();

  if (fractionalBits >= 0) {
    result += "_fp";
    result += std::to_string(fractionalBits);
  }
  else {
    result += "_fm";
    result += std::to_string(-fractionalBits);
  }

  return result;
}

/*
 * Quantizza un valore reale nel fixed-point di output.
 *
 * Il vecchio codice TAFFOMath usava rmTowardNegative.
 * std::floor replica tale direzione di arrotondamento.
 */
static APInt quantizeFixedPoint(long double realValue, const ConversionScalarType& outputType) {
  const int outputBitsInt = outputType.getBits();

  const unsigned outputBits = static_cast<unsigned>(outputBitsInt);

  const int outputFractionalBits = outputType.getFractionalBits();

  long double scaledValue = std::ldexp(realValue, outputFractionalBits);

  scaledValue = std::floor(scaledValue);

  if (outputType.isSigned()) {
    /*
     * Intervallo signed:
     *
     * [-2^(N-1), 2^(N-1) - 1]
     */
    const long double minimumRaw = -std::ldexp(1.0L, outputBitsInt - 1);

    const long double upperExclusive = std::ldexp(1.0L, outputBitsInt - 1);

    if (scaledValue <= minimumRaw)
      return APInt::getSignedMinValue(outputBits);

    if (scaledValue >= upperExclusive)
      return APInt::getSignedMaxValue(outputBits);

    const int64_t rawValue = static_cast<int64_t>(scaledValue);

    return APInt(outputBits, static_cast<uint64_t>(rawValue), true);
  }

  /*
   * Intervallo unsigned:
   *
   * [0, 2^N - 1]
   */
  if (scaledValue <= 0.0L)
    return APInt(outputBits, 0, false);

  const long double upperExclusive = std::ldexp(1.0L, outputBitsInt);

  if (scaledValue >= upperExclusive)
    return APInt::getMaxValue(outputBits);

  const uint64_t rawValue = static_cast<uint64_t>(scaledValue);

  return APInt(outputBits, rawValue, false);
}

static long double evaluateTanh(long double x) { return std::tanh(x); }

/*
 * Sigmoid numericamente stabile.
 */
static long double evaluateSigmoid(long double x) {
  if (x >= 0.0L) {
    const long double expNegative = std::exp(-x);

    return 1.0L / (1.0L + expNegative);
  }

  const long double expPositive = std::exp(x);

  return expPositive / (1.0L + expPositive);
}

/*
 * Genera oppure recupera la LUT globale relativa a:
 *
 * - activation;
 * - tipo fixed-point di input;
 * - tipo fixed-point di output;
 * - dimensione della LUT.
 */
static GlobalVariable* getOrCreateActivationLUT(Module* module,
                                                LLVMContext& context,
                                                StringRef activationName,
                                                const ConversionScalarType& inputType,
                                                const ConversionScalarType& outputType,
                                                ActivationEvaluator evaluator) {
  if (!module)
    return nullptr;

  const int inputBitsInt = inputType.getBits();

  const int outputBitsInt = outputType.getBits();

  if (inputBitsInt < static_cast<int>(LUT_BITS))
    return nullptr;

  if (outputBitsInt <= 0 || outputBitsInt > 64)
    return nullptr;

  const unsigned inputBits = static_cast<unsigned>(inputBitsInt);

  auto* outputLLVMType = dyn_cast<IntegerType>(outputType.toScalarLLVMType(context));

  if (!outputLLVMType)
    return nullptr;

  const unsigned shiftAmount = inputBits - LUT_BITS;

  const int internalFractionalBits = inputType.getFractionalBits() - static_cast<int>(shiftAmount);

  const std::string lutName = "__taffo_" + activationName.str() + "_" + encodeFixedType(inputType) + "_to_"
                            + encodeFixedType(outputType) + "_lut1024";

  ArrayType* lutType = ArrayType::get(outputLLVMType, LUT_SIZE);

  GlobalVariable* lutVariable = module->getNamedGlobal(lutName);

  if (lutVariable) {
    if (lutVariable->getValueType() != lutType) {
      errs() << "[TAFFO LUT] global " << lutName << " già presente con tipo incompatibile\n";

      return nullptr;
    }

    if (lutVariable->hasInitializer())
      return lutVariable;
  }

  /*
   * Generazione compile-time della LUT.
   */
  std::vector<Constant*> lutValues;
  lutValues.reserve(LUT_SIZE);

  for (unsigned index = 0; index < LUT_SIZE; ++index) {
    /*
     * L'indice rappresenta il bit pattern del tipo
     * fixed-point interno largo 10 bit.
     *
     * Caso signed:
     *
     * 0...511     -> 0...511
     * 512...1023  -> -512...-1
     */
    APInt internalPattern(LUT_BITS, index);

    int64_t internalRawValue;

    if (inputType.isSigned())
      internalRawValue = internalPattern.getSExtValue();
    else
      internalRawValue = static_cast<int64_t>(internalPattern.getZExtValue());

    /*
     * Conversione fixed-point -> reale:
     *
     * realInput =
     *   internalRawValue *
     *   2^(-internalFractionalBits)
     */
    const long double realInput = std::ldexp(static_cast<long double>(internalRawValue), -internalFractionalBits);

    const long double realOutput = evaluator(realInput);

    const APInt fixedOutput = quantizeFixedPoint(realOutput, outputType);

    lutValues.push_back(ConstantInt::get(context, fixedOutput));
  }

  Constant* initializer = ConstantArray::get(lutType, lutValues);

  if (!lutVariable) {
    lutVariable = new GlobalVariable(*module, lutType, true, GlobalValue::InternalLinkage, initializer, lutName);
  }
  else {
    lutVariable->setInitializer(initializer);

    lutVariable->setConstant(true);

    lutVariable->setLinkage(GlobalValue::InternalLinkage);
  }

  return lutVariable;
}

/*
 * Emette il codice LLVM per:
 *
 *   input
 *     ↓
 *   shift di inputBits - LUT_BITS
 *     ↓
 *   trunc a LUT_BITS
 *     ↓
 *   zext dell'indice
 *     ↓
 *   GEP + load
 */
static Value* emitActivationLUTLookup(IRBuilder<NoFolder>& builder,
                                      Value* fixedOperand,
                                      const ConversionScalarType& inputType,
                                      GlobalVariable* lutVariable,
                                      StringRef instructionPrefix) {
  if (!fixedOperand || !lutVariable)
    return nullptr;

  auto* inputLLVMType = dyn_cast<IntegerType>(fixedOperand->getType());

  if (!inputLLVMType)
    return nullptr;

  const int inputBitsInt = inputType.getBits();

  if (inputBitsInt < static_cast<int>(LUT_BITS))
    return nullptr;

  const unsigned inputBits = static_cast<unsigned>(inputBitsInt);

  if (inputLLVMType->getBitWidth() != inputBits) {
    errs() << "[TAFFO LUT] bit-width LLVM dell'input = " << inputLLVMType->getBitWidth()
           << ", tipo fixed-point = " << inputBits << "\n";

    return nullptr;
  }

  const unsigned shiftAmount = inputBits - LUT_BITS;

  Value* reducedOperand = fixedOperand;

  if (shiftAmount > 0) {
    Value* shiftValue = ConstantInt::get(inputLLVMType, shiftAmount);

    if (inputType.isSigned())
      reducedOperand = builder.CreateAShr(fixedOperand, shiftValue, Twine(instructionPrefix) + ".shifted");
    else
      reducedOperand = builder.CreateLShr(fixedOperand, shiftValue, Twine(instructionPrefix) + ".shifted");
  }

  IntegerType* indexPatternType = IntegerType::get(fixedOperand->getContext(), LUT_BITS);

  Value* indexPattern = reducedOperand;

  if (inputBits != LUT_BITS)
    indexPattern = builder.CreateTrunc(reducedOperand, indexPatternType, Twine(instructionPrefix) + ".pattern");

  /*
   * Reinterpretazione del pattern come indice unsigned:
   *
   * i10 1111111111 -> i32 1023
   */
  Value* lutIndex = builder.CreateZExt(indexPattern, builder.getInt32Ty(), Twine(instructionPrefix) + ".index");

  auto* lutType = dyn_cast<ArrayType>(lutVariable->getValueType());

  if (!lutType)
    return nullptr;

  Type* elementType = lutType->getElementType();

  Value* elementPointer =
    builder.CreateGEP(lutType, lutVariable, {builder.getInt32(0), lutIndex}, Twine(instructionPrefix) + ".gep");

  return builder.CreateLoad(elementType, elementPointer, Twine(instructionPrefix) + ".value");
}

} // namespace

Value* ConversionPass::createTanh(CallBase* call) {

  if (!call || call->arg_empty())
    return unsupported;

  IRBuilder<NoFolder> builder(call);

  LLVMContext& context = call->getContext();

  Value* originalOperand = call->getArgOperand(0);

  Function* calledFunction = call->getCalledFunction();

  /*
   * INPUT TYPE
   *
   * Il vecchio branch recuperava il fixed-point
   * dall'argomento formale della funzione.
   */
  std::unique_ptr<ConversionScalarType> ownedInputType;

  if (calledFunction && !calledFunction->arg_empty())
    ownedInputType = makeFixedConversionType(taffoInfo, calledFunction->getArg(0), originalOperand);

  /*
   * Fallback: FixedPointInfo associato direttamente
   * all'actual operand.
   */
  if (!ownedInputType)
    ownedInputType = makeFixedConversionType(taffoInfo, originalOperand, originalOperand);

  const ConversionScalarType* inputType = ownedInputType.get();

  /*
   * Fallback sul ConversionType tradizionale.
   */
  if (!inputType) {
    const auto* candidate = taffoConvInfo.getNewOrOldType<ConversionScalarType>(originalOperand);

    if (candidate && candidate->isFixedPoint())
      inputType = candidate;
  }

  /*
   * OUTPUT TYPE
   *
   * Il vecchio branch recuperava il tipo di ritorno
   * dalla call site.
   */
  std::unique_ptr<ConversionScalarType> ownedOutputType = makeFixedConversionType(taffoInfo, call, call);

  /*
   * Fallback sull'informazione associata alla funzione.
   */
  if (!ownedOutputType && calledFunction)
    ownedOutputType = makeFixedConversionType(taffoInfo, calledFunction, call);

  const ConversionScalarType* outputType = ownedOutputType.get();

  if (!outputType) {
    const auto* candidate = taffoConvInfo.getNewOrOldType<ConversionScalarType>(call);

    if (candidate && candidate->isFixedPoint())
      outputType = candidate;
  }

  if (!inputType) {
    errs() << "[TANH] unsupported: tipo fixed-point "
              "dell'input non trovato\n";

    return unsupported;
  }

  if (!outputType) {
    errs() << "[TANH] unsupported: tipo fixed-point "
              "dell'output non trovato\n";

    return unsupported;
  }

  const int inputBits = inputType->getBits();

  const int outputBits = outputType->getBits();

  if (inputBits < static_cast<int>(LUT_BITS)) {
    errs() << "[TANH] unsupported: input con meno "
              "di 10 bit\n";

    return unsupported;
  }

  if (outputBits <= 0 || outputBits > 64) {
    errs() << "[TANH] unsupported: bit-width "
              "dell'output non supportata\n";

    return unsupported;
  }

  Value* fixedOperand = getConvertedOperand(originalOperand, *inputType, call, ConvTypePolicy::ForceHint);

  if (!fixedOperand) {
    errs() << "[TANH] getConvertedOperand "
              "ha restituito nullptr\n";

    return nullptr;
  }

  GlobalVariable* lutVariable =
    getOrCreateActivationLUT(call->getModule(), context, "tanh", *inputType, *outputType, evaluateTanh);

  if (!lutVariable) {
    errs() << "[TANH] impossibile creare la LUT\n";

    return nullptr;
  }

  Value* result = emitActivationLUTLookup(builder, fixedOperand, *inputType, lutVariable, "tanh.lut");

  if (!result) {
    errs() << "[TANH] impossibile generare "
              "il lookup della LUT\n";

    return nullptr;
  }

  setConversionResultInfo(result, call, outputType);

  return result;
}

Value* ConversionPass::createReLU(CallBase* call) {

  if (!call || call->arg_empty())
    return unsupported;

  IRBuilder<NoFolder> builder(call);

  ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(call);

  auto* newConvType = valueConvInfo->getNewOrOldType<ConversionScalarType>();

  if (!newConvType || !newConvType->isFixedPoint()) {
    errs() << "[RELU] unsupported: tipo fixed-point "
              "non disponibile\n";

    return unsupported;
  }

  Value* originalOperand = call->getArgOperand(0);

  Value* convertedOperand = getConvertedOperand(originalOperand, *newConvType, call, ConvTypePolicy::ForceHint);

  if (!convertedOperand)
    return nullptr;

  auto* llvmType = dyn_cast<IntegerType>(convertedOperand->getType());

  if (!llvmType)
    return nullptr;

  Value* zero = ConstantInt::get(llvmType, 0);

  Value* isPositive = builder.CreateICmpSGT(convertedOperand, zero, "relu.positive");

  Value* result = builder.CreateSelect(isPositive, convertedOperand, zero, "relu.value");

  setConversionResultInfo(result, call, newConvType);

  return result;
}

Value* ConversionPass::createSigmoid(CallBase* call) {

  if (!call || call->arg_empty())
    return unsupported;

  IRBuilder<NoFolder> builder(call);

  LLVMContext& context = call->getContext();

  Value* originalOperand = call->getArgOperand(0);

  Function* calledFunction = call->getCalledFunction();

  /*
   * INPUT TYPE
   */
  std::unique_ptr<ConversionScalarType> ownedInputType;

  if (calledFunction && !calledFunction->arg_empty())
    ownedInputType = makeFixedConversionType(taffoInfo, calledFunction->getArg(0), originalOperand);

  if (!ownedInputType)
    ownedInputType = makeFixedConversionType(taffoInfo, originalOperand, originalOperand);

  const ConversionScalarType* inputType = ownedInputType.get();

  if (!inputType) {
    const auto* candidate = taffoConvInfo.getNewOrOldType<ConversionScalarType>(originalOperand);

    if (candidate && candidate->isFixedPoint())
      inputType = candidate;
  }

  /*
   * OUTPUT TYPE
   */
  std::unique_ptr<ConversionScalarType> ownedOutputType = makeFixedConversionType(taffoInfo, call, call);

  if (!ownedOutputType && calledFunction)
    ownedOutputType = makeFixedConversionType(taffoInfo, calledFunction, call);

  const ConversionScalarType* outputType = ownedOutputType.get();

  if (!outputType) {
    const auto* candidate = taffoConvInfo.getNewOrOldType<ConversionScalarType>(call);

    if (candidate && candidate->isFixedPoint())
      outputType = candidate;
  }

  if (!inputType) {
    errs() << "[SIGMOID] unsupported: tipo fixed-point "
              "dell'input non trovato\n";

    return unsupported;
  }

  if (!outputType) {
    errs() << "[SIGMOID] unsupported: tipo fixed-point "
              "dell'output non trovato\n";

    return unsupported;
  }

  const int inputBits = inputType->getBits();

  const int outputBits = outputType->getBits();

  if (inputBits < static_cast<int>(LUT_BITS)) {
    errs() << "[SIGMOID] unsupported: input con meno "
              "di 10 bit\n";

    return unsupported;
  }

  if (outputBits <= 0 || outputBits > 64) {
    errs() << "[SIGMOID] unsupported: bit-width "
              "dell'output non supportata\n";

    return unsupported;
  }

  Value* fixedOperand = getConvertedOperand(originalOperand, *inputType, call, ConvTypePolicy::ForceHint);

  if (!fixedOperand) {
    errs() << "[SIGMOID] getConvertedOperand "
              "ha restituito nullptr\n";

    return nullptr;
  }

  GlobalVariable* lutVariable =
    getOrCreateActivationLUT(call->getModule(), context, "sigmoid", *inputType, *outputType, evaluateSigmoid);

  if (!lutVariable) {
    errs() << "[SIGMOID] impossibile creare la LUT\n";

    return nullptr;
  }

  Value* result = emitActivationLUTLookup(builder, fixedOperand, *inputType, lutVariable, "sigmoid.lut");

  if (!result) {
    errs() << "[SIGMOID] impossibile generare "
              "il lookup della LUT\n";

    return nullptr;
  }

  setConversionResultInfo(result, call, outputType);

  return result;
}
