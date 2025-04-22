#pragma once

#include "Debug/Logger.hpp"
#include "FixedPointType.hpp"
#include "PtrCasts.hpp"
#include "TaffoInfo/ConversionInfo.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "Types/TransparentType.hpp"
#include "Types/TypeUtils.hpp"

#include "llvm/Support/Casting.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>

#include <memory>

#define DEBUG_TYPE "taffo-conversion"
extern llvm::cl::opt<unsigned int> MaxTotalBitsConv;
extern llvm::cl::opt<unsigned int> MinQuotientFrac;

STATISTIC(FixToFloatCount,
          "Number of generic fixed point to floating point "
          "value conversion operations inserted");
STATISTIC(FloatToFixCount,
          "Number of generic floating point to fixed point "
          "value conversion operations inserted");
STATISTIC(FixToFloatWeight,
          "Number of generic fixed point to floating point "
          "value conversion operations inserted,"
          " weighted by the loop depth");
STATISTIC(FloatToFixWeight,
          "Number of generic floating point to fixed point "
          "value conversion operations inserted,"
          " weighted by the loop depth");
STATISTIC(FallbackCount, "Number of instructions not replaced by a fixed-point-native equivalent");
STATISTIC(ConversionCount, "Number of instructions affected by taffo");
STATISTIC(MetadataCount, "Number of valid Metadata found");
STATISTIC(FunctionCreated, "Number of fixed point function inserted");

/* flags in conversionPool; actually not valid LLVM values but dummy pointers */
extern llvm::Value* ConversionError;
extern llvm::Value* Unsupported;

namespace taffo {

struct PHIInfo {
  llvm::Value* placeh_noconv;
  llvm::Value* placeh_conv;
};

class Conversion : public llvm::PassInfoMixin<Conversion> {
public:
  llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);
};

struct FloatToFixed {
  /** Map from original values to converted values.
   *  Values not to be converted do not appear in the map.
   *  Values which have not been converted successfully are mapped to
   *  one of two sentinel values, ConversionError or Unsupported. */
  llvm::DenseMap<llvm::Value*, llvm::Value*> convertedValues;

  /** Map from original function (as cloned by Initializer)
   *  to function cloned by this pass in order to change argument
   *  and return values */
  llvm::DenseMap<llvm::Function*, llvm::Function*> functionPool;

  /* to not be accessed directly, use valueInfo() */
  llvm::DenseMap<llvm::Value*, std::shared_ptr<ConversionInfo>> conversionInfo;

  llvm::ValueMap<llvm::PHINode*, PHIInfo> phiReplacementData;

  llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& AM);
  void readGlobalMetadata(llvm::Module& m, llvm::SmallVectorImpl<llvm::Value*>& res, bool functionAnnotation = false);
  void readLocalMetadata(llvm::Function& f, llvm::SmallVectorImpl<llvm::Value*>& res, bool onlyArguments = false);
  void readAllLocalMetadata(llvm::Module& m, llvm::SmallVectorImpl<llvm::Value*>& res);
  bool parseMetaData(llvm::SmallVectorImpl<llvm::Value*>* variables,
                     std::shared_ptr<taffo::ValueInfo> fpInfo,
                     llvm::Value* instr);
  void removeNoFloatTy(llvm::SmallVectorImpl<llvm::Value*>& res);
  void printAnnotatedObj(llvm::Module& m);
  void openPhiLoop(llvm::PHINode* phi);
  void closePhiLoops();
  bool isKnownConvertibleWithIncompleteMetadata(llvm::Value* V);
  void sortQueue(std::vector<llvm::Value*>& vals);
  void cleanup(const std::vector<llvm::Value*>& queue);
  void cleanUpOriginalFunctions(llvm::Module& m);
  void insertOpenMPIndirection(llvm::Module& m);
  void propagateCall(std::vector<llvm::Value*>& vals, llvm::SmallVectorImpl<llvm::Value*>& global, llvm::Module& m);
  llvm::Function* createFixFun(llvm::CallBase* call, bool* old);
  void printConversionQueue(const std::vector<llvm::Value*>& vals);
  void performConversion(llvm::Module& m, std::vector<llvm::Value*>& q);
  llvm::Value* convertSingleValue(llvm::Module& m, llvm::Value* val, std::shared_ptr<FixedPointType>& fixpt);

  llvm::Value* createPlaceholder(llvm::Type* type, llvm::BasicBlock* where, llvm::StringRef name);

  enum class TypeMatchPolicy {
    RangeOverHintMaxFrac = 0, /// Minimize extra conversions
    RangeOverHintMaxInt,
    HintOverRangeMaxFrac,     /// Create new type different than the hint if hint
                              /// does not fit value
    HintOverRangeMaxInt,
    ForceHint                 /// Always use the hint for the type
  };

  bool isMaxFracPolicy(TypeMatchPolicy tmp) {
    return tmp == TypeMatchPolicy::RangeOverHintMaxFrac || tmp == TypeMatchPolicy::HintOverRangeMaxFrac;
  }

  bool isMaxIntPolicy(TypeMatchPolicy tmp) {
    return tmp == TypeMatchPolicy::RangeOverHintMaxInt || tmp == TypeMatchPolicy::HintOverRangeMaxInt;
  }

  bool isHintPreferredPolicy(TypeMatchPolicy tmp) {
    return tmp == TypeMatchPolicy::HintOverRangeMaxInt || tmp == TypeMatchPolicy::HintOverRangeMaxFrac
        || tmp == TypeMatchPolicy::ForceHint;
  }

  /* convert* functions return nullptr if the conversion cannot be
   * recovered, and Unsupported to trigger the fallback behavior */
  llvm::Constant* convertConstant(llvm::Constant* flt, std::shared_ptr<FixedPointType>& fixpt, TypeMatchPolicy typepol);
  llvm::Constant* convertGlobalVariable(llvm::GlobalVariable* glob, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Constant*
  convertConstantExpr(llvm::ConstantExpr* cexp, std::shared_ptr<FixedPointType>& fixpt, TypeMatchPolicy typepol);
  llvm::Constant* convertConstantAggregate(llvm::ConstantAggregate* cag,
                                           std::shared_ptr<FixedPointType>& fixpt,
                                           TypeMatchPolicy typepol);
  llvm::Constant* convertConstantDataSequential(llvm::ConstantDataSequential*,
                                                const std::shared_ptr<FixedPointScalarType>&);
  template <class T>
  llvm::Constant* createConstantDataSequential(llvm::ConstantDataSequential*, const std::shared_ptr<FixedPointType>&);
  llvm::Constant*
  convertLiteral(llvm::ConstantFP* flt, llvm::Instruction*, std::shared_ptr<FixedPointType>&, TypeMatchPolicy typepol);
  bool convertAPFloat(llvm::APFloat, llvm::APSInt&, llvm::Instruction*, const std::shared_ptr<FixedPointScalarType>&);
  llvm::Value* convertInstruction(llvm::Module& m, llvm::Instruction* val, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertAlloca(llvm::AllocaInst* alloca, const std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertLoad(llvm::LoadInst* load, std::shared_ptr<FixedPointType>& fixpt, llvm::Module& m);
  llvm::Value* convertStore(llvm::StoreInst* load, llvm::Module& m);
  llvm::Value* convertGep(llvm::GetElementPtrInst* gep, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertExtractValue(llvm::ExtractValueInst* exv, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertInsertValue(llvm::InsertValueInst* inv, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertPhi(llvm::PHINode* load, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertSelect(llvm::SelectInst* sel, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertCall(llvm::CallBase* call, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertRet(llvm::ReturnInst* ret, std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertBinOp(llvm::Instruction* instr, const std::shared_ptr<FixedPointScalarType>& fixpt);
  llvm::Value* convertUnaryOp(llvm::Instruction* instr, const std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* convertCmp(llvm::FCmpInst* fcmp);
  llvm::Value* convertCast(llvm::CastInst* cast, const std::shared_ptr<FixedPointType>& fixpt);
  llvm::Value* fallback(llvm::Instruction* unsupp, std::shared_ptr<FixedPointType>& fixpt);

  /* OpenCL support */
  bool isSupportedOpenCLFunction(llvm::Function* F);
  llvm::Value* convertOpenCLCall(llvm::CallBase* C);
  void cleanUpOpenCLKernelTrampolines(llvm::Module* M);

  /*Cuda support */
  bool isSupportedCudaFunction(llvm::Function* F);
  llvm::Value* convertCudaCall(llvm::CallBase* C);

  /* Math intrinsic support */
  bool isSupportedMathIntrinsicFunction(llvm::Function* F);
  llvm::Value* convertMathIntrinsicFunction(llvm::CallBase* C, const std::shared_ptr<FixedPointScalarType>& fixpt);

  /** Returns if a function is a library function which shall not
   *  be cloned.
   *  @param f The function to check */
  bool isSpecialFunction(const llvm::Function* f) {
    llvm::StringRef fName = f->getName();
    return fName.starts_with("llvm.") || f->empty();
  }

  /** Returns the converted Value matching a non-converted Value.
   *  @param val The non-converted value to match.
   *  @returns nullptr if the value has not been converted properly,
   *    the converted value if the original value was converted,
   *    or the original value itself if it does not require conversion. */
  llvm::Value* matchOp(llvm::Value* val) {
    llvm::Value* res = convertedValues[val];
    return res == ConversionError ? nullptr : (res ? res : val);
  }

  /** Returns a fixed point Value from any Value, whether it should be
   *  converted or not.
   *  @param val The non-converted value. Must be of a primitive floating-point
   *    non-reference LLVM type (in other words, ints, pointers, arrays, struct
   * are not allowed); use matchOp() for values of those types.
   *  @param iofixpt A reference to a fixed point type. On input,
   *    it must contain the preferred fixed point type required
   *    for the returned Value. On output, it will contain the
   *    actual fixed point type of the returned Value (which may or
   *    may not be different than the input type).
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns A fixed point value corresponding to val or nullptr if
   *    val was to be converted but its conversion failed. */
  llvm::Value* translateOrMatchOperand(llvm::Value* val,
                                       std::shared_ptr<FixedPointType>& iofixpt,
                                       llvm::Instruction* ip = nullptr,
                                       TypeMatchPolicy typepol = TypeMatchPolicy::RangeOverHintMaxFrac,
                                       bool wasHintForced = false);

  /** Returns a fixed point Value from any Value, whether it should be
   *  converted or not, if possible.
   *  @param val The non-converted value.
   *  @param iofixpt A reference to a fixed point type. On input,
   *    it must contain the preferred fixed point type required
   *    for the returned Value. On output, it will contain the
   *    actual fixed point type of the returned Value (which may or
   *    may not be different than the input type).
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns A fixed point value corresponding to val or nullptr if
   *    val was to be converted but its conversion failed. */
  llvm::Value* translateOrMatchAnyOperand(llvm::Value* val,
                                          std::shared_ptr<FixedPointType>& iofixpt,
                                          llvm::Instruction* ip = nullptr,
                                          TypeMatchPolicy typepol = TypeMatchPolicy::RangeOverHintMaxFrac) {
    auto& taffoInfo = TaffoInfo::getInstance();
    llvm::Value* res;
    if (val->getType()->getNumContainedTypes() > 0) {
      if (llvm::Constant* cst = llvm::dyn_cast<llvm::Constant>(val)) {
        res = convertConstant(cst, iofixpt, typepol);
        taffoInfo.setTransparentType(*res, TransparentTypeFactory::create(res->getType()));
      }
      else {
        res = matchOp(val);
        if (res) {
          if (typepol == TypeMatchPolicy::ForceHint)
            assert(*iofixpt == *getFixpType(res) && "type mismatch on reference Value");
          else
            *iofixpt = *getFixpType(res);
        }
      }
    }
    else {
      res = translateOrMatchOperand(val, iofixpt, ip, typepol);
    }
    return res;
  }

  /** Returns a fixed point Value of a specified fixed point type from any
   *  Value, whether it should be converted or not.
   *  @param val The non-converted value. Must be of a primitive floating-point
   *    non-reference LLVM type (in other words, ints, pointers, arrays, struct
   * are not allowed); use matchOp() for values of those types.
   *  @param fixpt The fixed point type of the value returned.
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns A fixed point value corresponding to val of type fixpt
   *    or nullptr if val was to be converted but its conversion failed.  */
  llvm::Value* translateOrMatchOperandAndType(llvm::Value* val,
                                              const std::shared_ptr<FixedPointType>& fixpt,
                                              llvm::Instruction* ip = nullptr) {
    std::shared_ptr<FixedPointType> iofixpt = fixpt->clone();
    return translateOrMatchOperand(val, iofixpt, ip, TypeMatchPolicy::ForceHint);
  }

  /** Returns a fixed point Value of a specified fixed point type from any
   *  Value, whether it should be converted or not, if possible.
   *  @param val The non-converted value.
   *  @param fixpt The fixed point type of the value returned.
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns A fixed point value corresponding to val of type fixpt
   *    or nullptr if val was to be converted but its conversion failed.
   *    An assertion is raised if the value cannot be converted to
   *    the specified type (for example if it is a pointer)  */
  llvm::Value* translateOrMatchAnyOperandAndType(llvm::Value* val,
                                                 const std::shared_ptr<FixedPointType>& fixpt,
                                                 llvm::Instruction* ip = nullptr) {
    std::shared_ptr<FixedPointType> iofixpt = fixpt->clone();
    return translateOrMatchAnyOperand(val, iofixpt, ip, TypeMatchPolicy::ForceHint);
  }

  bool hasConvertedValue(llvm::Value* value) { return convertedValues.contains(value); }

  llvm::Value* fallbackMatchValue(llvm::Value* value,
                                  const std::shared_ptr<TransparentType>& origType,
                                  llvm::Instruction* insertionPoint = nullptr) {
    Logger& logger = log();

    llvm::Value* fallBackValue = convertedValues.at(value);
    assert(fallBackValue != nullptr && "Value was converted to a nullptr");

    LLVM_DEBUG(
      auto indenter = logger.getIndenter();
      indenter.increaseIndent();
      logger << "[FallbackMatchingValue] ";
      if (fallBackValue != nullptr) {
        logger.logValue(value);
        logger << " was converted to ";
        logger.logValue(fallBackValue);
        logger << "\n";
      });

    if (fallBackValue == ConversionError) {
      LLVM_DEBUG(llvm::dbgs() << "error: bail out reverse match of " << *value << "\n");
      return nullptr;
    }

    LLVM_DEBUG(llvm::dbgs() << "hasInfo " << hasConversionInfo(fallBackValue) << "\n";);
    if (!hasConversionInfo(fallBackValue))
      return fallBackValue;
    LLVM_DEBUG(llvm::dbgs() << "Info noTypeConversion " << getConversionInfo(fallBackValue)->noTypeConversion << "\n";);
    if (getConversionInfo(fallBackValue)->noTypeConversion)
      return fallBackValue;

    if (!insertionPoint) {
      // argument is not an instruction, insert it's convertion in the first basic block
      if (insertionPoint == nullptr && llvm::isa<llvm::Argument>(fallBackValue)) {
        auto arg = llvm::cast<llvm::Argument>(fallBackValue);
        insertionPoint = (&*(arg->getParent()->begin()->getFirstInsertionPt()));
      }

      assert(insertionPoint && "ip mandatory for non-instruction values");
    }

    if (origType->isFloatingPointType() && !origType->isPointerType())
      return genConvertFixToFloat(fallBackValue, getFixpType(fallBackValue), origType);
    return fallBackValue;
  }

  /** Generate code for converting the value of a scalar from floating point to
   *  fixed point.
   *  @param flt A floating point scalar value.
   *  @param fixpt The fixed point type of the output
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns The converted value. */
  llvm::Value* genConvertFloatToFix(llvm::Value* flt,
                                    const std::shared_ptr<FixedPointScalarType>& fixpt,
                                    llvm::Instruction* ip = nullptr);

  llvm::Value* genConvertFixToFloat(llvm::Value* fix,
                                    const std::shared_ptr<FixedPointType>& fixpt,
                                    const std::shared_ptr<TransparentType>& destt);

  /** Generate code for converting between two fixed point formats.
   *  @param flt A fixed point scalar value.
   *  @param scrt The fixed point type of the input
   *  @param destt The fixed point type of the output
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns The converted value. */
  llvm::Value* genConvertFixedToFixed(llvm::Value* fix,
                                      const std::shared_ptr<FixedPointScalarType>& srct,
                                      const std::shared_ptr<FixedPointScalarType>& destt,
                                      llvm::Instruction* ip = nullptr);

  /** Transforms a pre-existing LLVM type to a new LLVM
   *  type with integers instead of floating point depending on a
   *  fixed point type specification.
   *  @param fptype The original type
   *  @param baset The fixed point type
   *  @param hasfloats If non-null, points to a bool which, on return,
   *    will be true if at least one floating point type to transform to
   *    fixed point was encountered.
   *  @returns The new LLVM type.  */
  llvm::Type* getLLVMFixedPointTypeForFloatType(const std::shared_ptr<taffo::TransparentType>& srcType,
                                                const std::shared_ptr<FixedPointType>& baset,
                                                bool* hasfloats = nullptr);

  llvm::Instruction* getFirstInsertionPointAfter(llvm::Instruction* i) {
    llvm::Instruction* ip = i->getNextNode();
    if (!ip) {
      LLVM_DEBUG(llvm::dbgs() << "warning: getFirstInsertionPointAfter on a BB-terminating inst\n");
      return nullptr;
    }
    if (llvm::isa<llvm::PHINode>(ip))
      ip = ip->getParent()->getFirstNonPHI();
    return ip;
  }

  llvm::Type* getLLVMFixedPointTypeForFloatValue(llvm::Value* val);

  std::shared_ptr<ConversionInfo> newConversionInfo(llvm::Value* val) {
    LLVM_DEBUG(llvm::dbgs() << "new valueinfo for " << *val << "\n");
    auto vi = conversionInfo.find(val);
    if (vi == conversionInfo.end()) {
      conversionInfo[val] = std::make_shared<ConversionInfo>();
      return conversionInfo[val];
    }
    else {
      assert(false && "value already has info!");
    }
  }

  std::shared_ptr<ConversionInfo> demandConversionInfo(llvm::Value* val, bool* isNew = nullptr) {
    LLVM_DEBUG(llvm::dbgs() << "new valueinfo for " << *val << "\n");
    auto vi = conversionInfo.find(val);
    if (vi == conversionInfo.end()) {
      if (isNew)
        *isNew = true;
      conversionInfo[val] = std::make_shared<ConversionInfo>();
      return conversionInfo[val];
    }
    else {
      if (isNew)
        *isNew = false;
      return vi->getSecond();
    }
  }

  std::shared_ptr<ConversionInfo> getConversionInfo(llvm::Value* val) {
    auto vi = conversionInfo.find(val);
    if (vi == conversionInfo.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Requested info for " << *val << " which doesn't have it!!! ABORT\n");
      llvm_unreachable("PAAAANIC!! VALUE WITH NO INFO");
    }
    return vi->getSecond();
  }

  std::shared_ptr<FixedPointType> getFixpType(const llvm::Value* val) {
    auto vi = conversionInfo.find(val);
    assert(vi != conversionInfo.end() && "value with no info");
    return vi->getSecond()->fixpType->clone();
  }

  bool hasConversionInfo(const llvm::Value* val) const { return conversionInfo.find(val) != conversionInfo.end(); }

  bool isConvertedFixedPoint(llvm::Value* val) {
    auto& taffoInfo = TaffoInfo::getInstance();
    if (!hasConversionInfo(val))
      return false;
    std::shared_ptr<ConversionInfo> vi = getConversionInfo(val);
    if (vi->noTypeConversion)
      return false;
    if (vi->fixpType->isInvalid())
      return false;
    if (*taffoInfo.getTransparentType(*val) == *vi->origType)
      return false;
    return true;
  }

  bool isFloatingPointToConvert(llvm::Value* val) {
    if (llvm::isa<llvm::Argument>(val))
      // function arguments to be converted are substituted by placeholder
      // values in the function cloning stage.
      // Besides, they cannot be replaced without recreating the
      // function, thus they never fit the requirements for being
      // converted.
      return false;
    if (!hasConversionInfo(val))
      return false;
    std::shared_ptr<ConversionInfo> vi = getConversionInfo(val);
    if (vi->noTypeConversion)
      return false;
    if (vi->fixpType->isInvalid())
      return false;
    if (llvm::ReturnInst* ret = llvm::dyn_cast<llvm::ReturnInst>(val))
      val = ret->getReturnValue();
    llvm::Type* ty = getUnwrappedType(val);
    if (!ty->isStructTy() && !ty->isFloatTy())
      return false;
    return true;
  }

  llvm::Value* copyValueInfo(llvm::Value* dst, llvm::Value* src, std::shared_ptr<TransparentType> dstType = nullptr) {
    using namespace llvm;
    using namespace taffo;
    auto& taffoInfo = TaffoInfo::getInstance();
    if (taffoInfo.hasValueInfo(*src)) {
      std::shared_ptr<ValueInfo> dstInfo = taffoInfo.getValueInfo(*src)->clone();
      taffoInfo.setValueInfo(*dst, dstInfo);
    }
    if (dstType)
      taffoInfo.setTransparentType(*dst, dstType);
    else
      taffoInfo.setTransparentType(*dst, TransparentTypeFactory::create(dst->getType()));

    // TODO check old impl because I don't know what this does
    /*if (openMPIndirectMD) {
      if (auto *to = dyn_cast<Instruction>(dst)) {
        to->setMetadata(INDIRECT_METADATA, openMPIndirectMD);
      }
    }*/

    return dst;
  }

  void updateFPTypeMetadata(llvm::Value* v, bool isSigned, int fractionalBits, int bits) {
    using namespace taffo;
    std::shared_ptr<ValueInfo> valueInfo = TaffoInfo::getInstance().getValueInfo(*v);
    std::shared_ptr<ScalarInfo> scalarInfo = std::dynamic_ptr_cast_or_null<ScalarInfo>(valueInfo);
    if (!scalarInfo)
      return;
    std::shared_ptr<ScalarInfo> newScalarInfo = std::static_ptr_cast<ScalarInfo>(scalarInfo->clone());
    newScalarInfo->numericType.reset(new FixedPointInfo(isSigned, bits, fractionalBits));
    TaffoInfo::getInstance().setValueInfo(*v, newScalarInfo);
  }

  void updateConstTypeMetadata(llvm::Value* v, unsigned opIdx, const std::shared_ptr<FixedPointType>& type) {
    using namespace llvm;
    using namespace taffo;
    Instruction* i = dyn_cast<Instruction>(v);
    // TODO: handle case when IRBuilder does constant folding, and v is a constant.
    if (!i)
      return;
    Value* op = i->getOperand(opIdx);
    if (!isa<Constant>(op))
      return;
    TaffoInfo& taffoInfo = TaffoInfo::getInstance();
    if (!taffoInfo.hasValueInfo(*op))
      return;
    if (std::shared_ptr<ScalarInfo> opScalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*op))) {
      std::shared_ptr<FixedPointScalarType> scalarType = std::static_ptr_cast<FixedPointScalarType>(type);
      opScalarInfo->numericType = std::make_shared<FixedPointInfo>(
        scalarType->isSigned(), scalarType->getBits(), scalarType->getFractionalBits());
    }
  }

  int getLoopNestingLevelOfValue(llvm::Value* v);

  template <class T>
  llvm::Constant* createConstantDataSequentialFP(llvm::ConstantDataSequential* cds,
                                                 const std::shared_ptr<FixedPointType>& fixpt);

  bool associateFixFormat(const std::shared_ptr<taffo::ScalarInfo>& II, std::shared_ptr<FixedPointType>& iofixpt);

  void convertIndirectCalls(llvm::Module& m);

  void handleKmpcFork(llvm::CallInst* patchedDirectCall, llvm::Function* indirectFunction);

private:
  llvm::ModuleAnalysisManager* MAM;
  const llvm::DataLayout* ModuleDL;
};

llvm::Value* adjustBufferSize(
  llvm::Value* OrigSize, llvm::Type* OldTy, llvm::Type* NewTy, llvm::Instruction* IP, bool Tight = false);

} // namespace taffo

#undef DEBUG_TYPE
