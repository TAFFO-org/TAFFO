#pragma once

#include "Debug/Logger.hpp"
#include "SerializationUtils.hpp"
#include "TaffoInfo/PhiInfo.hpp"
#include "TaffoInfo/TaffoConvInfo.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "TaffoInfo/ValueConvInfo.hpp"
#include "TransparentType.hpp"
#include "TypeDeductionAnalysis.hpp"
#include "Types/ConversionType.hpp"
#include "Types/TypeUtils.hpp"
#include "Utils/PtrCasts.hpp"

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
#include <llvm/Support/Casting.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>

#include <memory>
#include <sstream>

#define DEBUG_TYPE "taffo-conv"
extern llvm::cl::opt<unsigned> maxTotalBitsConv;
extern llvm::cl::opt<unsigned> minQuotientFrac;

STATISTIC(fixToFloatCount, "Number of generic fixed point to floating point value conversion operations inserted");
STATISTIC(floatToFixCount, "Number of generic floating point to fixed point value conversion operations inserted");
STATISTIC(fallbackCount, "Number of instructions not replaced by a fixed-point-native equivalent");
STATISTIC(conversionCount, "Number of instructions affected by taffo");
STATISTIC(valueInfoCount, "Number of valid valueInfo found");
STATISTIC(functionCreated, "Number of fixed point function inserted");

// Not valid LLVM values but dummy pointer
extern llvm::Value* unsupported;

namespace taffo {

class ConversionPass : public llvm::PassInfoMixin<ConversionPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);

  /** Map from original values to converted values.
   *  Values not to be converted do not appear in the map.
   *  Values which have not been converted successfully are mapped to
   *  one of two sentinel values, ConversionError or Unsupported. */
  llvm::DenseMap<llvm::Value*, llvm::Value*> convertedValues;

  /** Map from original function (as cloned by Initializer)
   *  to function cloned by this pass in order to change argument
   *  and return values */
  llvm::DenseMap<llvm::Function*, llvm::Function*> functionPool;

  llvm::DenseMap<llvm::PHINode*, PhiInfo> phiNodeInfo;

  void buildGlobalConvInfo(llvm::Module& m, llvm::SmallVectorImpl<llvm::Value*>& values);
  void buildLocalConvInfo(llvm::Function& f, llvm::SmallVectorImpl<llvm::Value*>& values, bool argsOnly = false);
  void buildAllLocalConvInfo(llvm::Module& m, llvm::SmallVectorImpl<llvm::Value*>& values);
  bool buildConvInfo(llvm::SmallVectorImpl<llvm::Value*>* convQueue, llvm::Value* value);
  bool isAlwaysConvertible(llvm::Value* value);

  void createConversionQueue(std::vector<llvm::Value*>& values);
  void propagateCalls(std::vector<llvm::Value*>& convQueue,
                      const llvm::SmallVectorImpl<llvm::Value*>& globalValues,
                      llvm::Module& m);
  llvm::Function* createConvertedFunctionForCall(llvm::CallBase* call, bool* alreadyHandledNewF);
  void openPhiLoop(llvm::PHINode* phi);
  void closePhiLoops();

  void printConversionQueue(const std::vector<llvm::Value*>& queue);
  void cleanup(const std::vector<llvm::Value*>& queue);
  void cleanUpOriginalFunctions(llvm::Module& m);

  using MemoryAllocationsVec = std::vector<std::pair<llvm::Instruction*, tda::TransparentType*>>;
  MemoryAllocationsVec collectMemoryAllocations(llvm::Module& m);
  void adjustSizeOfMemoryAllocations(llvm::Module& m,
                                     const MemoryAllocationsVec& oldMemoryAllocations,
                                     const MemoryAllocationsVec& newMemoryAllocations);
  llvm::Value* adjustMemoryAllocationSize(llvm::Value* oldSizeValue,
                                          const std::shared_ptr<tda::TransparentType>& oldAllocatedType,
                                          const std::shared_ptr<tda::TransparentType>& newAllocatedType,
                                          llvm::Instruction* insertionPoint);

  void performConversion(const std::vector<llvm::Value*>& queue);
  llvm::Value* convert(llvm::Value* value, std::unique_ptr<ConversionType>* resConvType);

  llvm::Value* createPlaceholder(llvm::Type* type, llvm::BasicBlock* where, llvm::StringRef name);

  class ConvTypePolicy : tda::Printable {
  public:
    enum Policy {
      RangeOverHint = 0, // Convert based on value range when needed/possible
      ForceHint          // Always convert to the hint type
    };

    ConvTypePolicy(const Policy p)
    : policy(p) {}

    bool operator==(const ConvTypePolicy& other) const { return policy == other.policy; }

    std::string toString() const override {
      switch (policy) {
      case RangeOverHint: return "RangeOverHint";
      case ForceHint:     return "ForceHint";
      default:            llvm_unreachable("Invalid policy");
      }
    }

  private:
    Policy policy;
  };

  // Convert functions return:
  // - nullptr if the conversion cannot be recovered
  // - unsupported to trigger the fallback behavior

  // Constants

  llvm::Constant* convertConstant(llvm::Constant* constant,
                                  const ConversionType& convType,
                                  const ConvTypePolicy& policy,
                                  std::unique_ptr<ConversionType>* resConvType = nullptr);
  llvm::Constant* convertGlobalVariable(llvm::GlobalVariable* globalVariable,
                                        const ConversionType& convType,
                                        std::unique_ptr<ConversionType>* resConvType = nullptr);
  llvm::Constant* convertConstantFloat(llvm::ConstantFP* floatConst,
                                       const ConversionScalarType& convType,
                                       const ConvTypePolicy& policy,
                                       std::unique_ptr<ConversionType>* resConvType = nullptr);
  llvm::Constant* convertConstantAggregate(llvm::ConstantAggregate* constantAggregate,
                                           const ConversionType& convType,
                                           std::unique_ptr<ConversionType>* resConvType = nullptr);
  llvm::Constant* convertConstantDataSequential(llvm::ConstantDataSequential*,
                                                const ConversionScalarType& convType,
                                                std::unique_ptr<ConversionType>* resConvType = nullptr);
  template <class T>
  llvm::Constant* createConstantDataSequentialFixedPoint(llvm::ConstantDataSequential* cds,
                                                         const ConversionScalarType& convType,
                                                         std::unique_ptr<ConversionType>* resConvType = nullptr);
  template <class T>
  llvm::Constant* createConstantDataSequentialFloat(llvm::ConstantDataSequential* cds,
                                                    const ConversionScalarType& convType,
                                                    std::unique_ptr<ConversionType>* resConvType = nullptr);
  llvm::Constant* convertConstantExpr(llvm::ConstantExpr* constantExpr,
                                      const ConversionType& convType,
                                      const ConvTypePolicy& policy,
                                      std::unique_ptr<ConversionType>* resConvType = nullptr);
  void convertAPFloat(llvm::APFloat floatValue, llvm::APSInt& fixedPointValue, const ConversionScalarType& convType);

  // Instructions

  llvm::Value* convertInstruction(llvm::Instruction* inst);
  llvm::Value* convertAlloca(llvm::AllocaInst* alloca);
  llvm::Value* convertLoad(llvm::LoadInst* load);
  llvm::Value* convertStore(llvm::StoreInst* load);
  llvm::Value* convertGep(llvm::GetElementPtrInst* gep);
  llvm::Value* convertExtractValue(llvm::ExtractValueInst* extractValue);
  llvm::Value* convertInsertValue(llvm::InsertValueInst* insertValue);
  llvm::Value* convertPhi(llvm::PHINode* phi);
  llvm::Value* convertSelect(llvm::SelectInst* select);
  llvm::Value* convertCall(llvm::CallBase* call);
  llvm::Value* convertRet(llvm::ReturnInst* ret);
  llvm::Value* convertBinOp(llvm::Instruction* inst, const ConversionScalarType& convType);
  llvm::Value* convertFAdd(llvm::Instruction* inst, const ConversionScalarType& convType);
  llvm::Value* convertFSub(llvm::Instruction* inst, const ConversionScalarType& convType);
  llvm::Value* convertFRem(llvm::Instruction* inst, const ConversionScalarType& convType);
  llvm::Value* convertFMul(llvm::Instruction* inst, const ConversionScalarType& convType);
  llvm::Value* convertFDiv(llvm::Instruction* inst, const ConversionScalarType& convType);
  llvm::Value* convertUnaryOp(llvm::Instruction* inst);
  llvm::Value* convertCmp(llvm::FCmpInst* fcmp);
  llvm::Value* convertCast(llvm::CastInst* cast);
  llvm::Value* fallback(llvm::Instruction* inst);

  // OpenCL support

  bool isSupportedOpenCLFunction(llvm::Function* F);
  llvm::Value* convertOpenCLCall(llvm::CallBase* C);
  void cleanUpOpenCLKernelTrampolines(llvm::Module* M);

  // Cuda support

  bool isSupportedCudaFunction(llvm::Function* F);
  llvm::Value* convertCudaCall(llvm::CallBase* C);

  // Math intrinsic support

  bool isSupportedMathIntrinsicFunction(llvm::Function* F);
  llvm::Value* convertMathIntrinsicFunction(llvm::CallBase* call);

  ValueConvInfo* setConversionResultInfoCommon(llvm::Value* resultValue,
                                               llvm::Value* oldValue = nullptr,
                                               const ConversionType* resultConvType = nullptr) {
    if (!oldValue)
      oldValue = resultValue;
    ValueConvInfo* oldConvInfo = taffoConvInfo.getOrCreateValueConvInfo(oldValue);
    ConversionType* oldConvType = oldConvInfo->getCurrentType();
    if (!resultConvType)
      resultConvType = oldConvType;
    ValueConvInfo* resConvInfo;
    if (oldValue != resultValue) {
      if (taffoInfo.hasValueInfo(*oldValue))
        taffoInfo.setValueInfo(*resultValue, taffoInfo.getValueInfo(*oldValue));
      taffoInfo.setTransparentType(*resultValue, resultConvType->toTransparentType()->clone());
      // If missing, create valueConvInfo with the same oldType as oldConvType but adapted to the new transparent type
      resConvInfo = taffoConvInfo.getOrCreateValueConvInfo(resultValue, oldConvType);
    }
    else {
      // If missing, create valueConvInfo from scratch
      resConvInfo = taffoConvInfo.getOrCreateValueConvInfo(resultValue);
    }
    return resConvInfo;
  }

  void setConstantConversionResultInfo(llvm::Value* resultValue,
                                       llvm::Value* oldValue = nullptr,
                                       const ConversionType* resultConvType = nullptr,
                                       std::unique_ptr<ConversionType>* resConvTypeOwner = nullptr) {
    ValueConvInfo* resConvInfo = setConversionResultInfoCommon(resultValue, oldValue, resultConvType);
    if (!resConvInfo->isConstant() && resultConvType) {
      resConvInfo->setNewType(resultConvType->clone());
      resConvInfo->setConverted();
    }
    if (resConvTypeOwner && resultConvType)
      *resConvTypeOwner = resultConvType->clone();
  }

  void setConversionResultInfo(llvm::Value* resultValue,
                               llvm::Value* oldValue = nullptr,
                               const ConversionType* resultConvType = nullptr) {
    ValueConvInfo* resConvInfo = setConversionResultInfoCommon(resultValue, oldValue, resultConvType);
    if (resultConvType) {
      resConvInfo->setNewType(resultConvType->clone());
      resConvInfo->setConverted();
    }
  }

  void logFunctionSignature(llvm::Function* fun) {
    tda::Logger& logger = tda::log();
    logger.log(*taffoConvInfo.getCurrentType(fun), tda::Logger::Cyan) << " " << fun->getName().str() << "(";
    for (auto iter : llvm::enumerate(fun->args())) {
      if (iter.index() != 0)
        logger << ", ";
      logger.log(*taffoConvInfo.getCurrentType(&iter.value()),
                 iter.index() % 2 == 0 ? tda::Logger::Cyan : tda::Logger::Blue);
    }
    logger << ")";
  }

  /** Returns if a function is a library function which shall not
   *  be cloned.
   *  @param f The function to check */
  bool isSpecialFunction(const llvm::Function* f) {
    llvm::StringRef fName = f->getName();
    return fName.starts_with("llvm.") || f->empty();
  }

  llvm::Value* getConvertedOperand(llvm::Value* value,
                                   const ConversionType& convType,
                                   llvm::Instruction* insertionPoint = nullptr,
                                   const ConvTypePolicy& policy = ConvTypePolicy::RangeOverHint,
                                   std::unique_ptr<ConversionType>* resConvType = nullptr);

  /** Returns a fixed point Value from any Value, whether it should be
   *  converted or not, if possible.
   *  @param val The non-converted value.
   *  @param convType A reference to a fixed point type. On input,
   *    it must contain the preferred fixed point type required
   *    for the returned Value. On output, it will contain the
   *    actual fixed point type of the returned Value (which may or
   *    may not be different from the input type).
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns A fixed point value corresponding to val or nullptr if
   *    val was to be converted but its conversion failed. */
  llvm::Value* translateOrMatchAnyOperand(llvm::Value* val,
                                          const ConversionScalarType& convType,
                                          llvm::Instruction* ip = nullptr,
                                          ConvTypePolicy policy = ConvTypePolicy::RangeOverHint) {
    llvm::Value* res;
    if (val->getType()->getNumContainedTypes() > 0) {
      if (auto* constant = llvm::dyn_cast<llvm::Constant>(val))
        res = convertConstant(constant, convType, policy);
      else {
        res = convertedValues.at(val);
        if (policy == ConvTypePolicy::ForceHint)
          assert(convType == *taffoConvInfo.getNewType(res) && "type mismatch on reference Value");
      }
    }
    else
      res = getConvertedOperand(val, convType, ip, policy);
    return res;
  }

  /** Returns a fixed point Value of a specified fixed point type from any
   *  Value, whether it should be converted or not, if possible.
   *  @param val The non-converted value.
   *  @param convType The fixed point type of the value returned.
   *  @param ip The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns A fixed point value corresponding to val of type convType
   *    or nullptr if val was to be converted but its conversion failed.
   *    An assertion is raised if the value cannot be converted to
   *    the specified type (for example if it is a pointer)  */
  llvm::Value* translateOrMatchAnyOperandAndType(llvm::Value* val,
                                                 const ConversionScalarType& convType,
                                                 llvm::Instruction* ip = nullptr) {
    return translateOrMatchAnyOperand(val, convType, ip, ConvTypePolicy::ForceHint);
  }

  bool hasConvertedValue(llvm::Value* value) { return convertedValues.contains(value); }

  /** Generate code for converting between two fixed point formats.
   *  @param flt A fixed point scalar value.
   *  @param scrt The fixed point type of the input
   *  @param destt The fixed point type of the output
   *  @param insertionPoint The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns The converted value. */
  llvm::Value* genConvertConvToConv(llvm::Value* src,
                                    const ConversionScalarType& srcConvType,
                                    const ConversionScalarType& dstConvType,
                                    const ConvTypePolicy& policy,
                                    llvm::Instruction* insertionPoint = nullptr);

  /** Generate code for converting the value of a scalar from floating point to fixed point.
   *  @param src A floating point scalar value.
   *  @param dstConvType The fixed point type of the output
   *  @param insertionPoint The instruction which will use the returned value.
   *    Used for placing generated fixed point runtime conversion code in
   *    case val was not to be converted statically. Not required if val
   *    is an instruction or a constant.
   *  @returns The converted value. */
  llvm::Value* genConvertFloatToConv(llvm::Value* src,
                                     const ConversionScalarType& dstConvType,
                                     llvm::Instruction* insertionPoint = nullptr);

  llvm::Value* genConvertConvToFloat(llvm::Value* src,
                                     const ConversionScalarType& srcConvType,
                                     const ConversionScalarType& dstConvType);

  llvm::Instruction* getFirstInsertionPointAfter(llvm::Value* value) {
    if (auto* arg = llvm::dyn_cast<llvm::Argument>(value))
      return &*arg->getParent()->getEntryBlock().getFirstInsertionPt();
    if (auto* inst = llvm::dyn_cast<llvm::Instruction>(value)) {
      llvm::Instruction* insertionPoint = inst->getNextNode();
      if (!insertionPoint) {
        LLVM_DEBUG(tda::log() << __FUNCTION__ << " called on a BB-terminating inst\n");
        return nullptr;
      }
      if (llvm::isa<llvm::PHINode>(insertionPoint))
        insertionPoint = insertionPoint->getParent()->getFirstNonPHI();
      return insertionPoint;
    }
    return nullptr;
  }

  bool isConvertedFixedPoint(llvm::Value* value) {
    if (!taffoConvInfo.hasValueConvInfo(value))
      return false;
    ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(value);
    if (valueConvInfo->isConversionDisabled())
      return false;
    ConversionType* convType = valueConvInfo->getNewType();
    if (*taffoInfo.getTransparentType(*value) == *convType->toTransparentType())
      return false;
    return true;
  }

  bool isFloatingPointToConvert(llvm::Value* value) {
    if (llvm::isa<llvm::Argument>(value))
      // Function arguments to be converted are substituted by placeholder values in the function cloning stage
      // Moreover, they cannot be replaced without recreating the function, thus never requiring conversion
      return false;
    if (!taffoConvInfo.hasValueConvInfo(value))
      return false;
    ValueConvInfo* valueConvInfo = taffoConvInfo.getValueConvInfo(value);
    if (valueConvInfo->isConversionDisabled())
      return false;
    if (auto* ret = llvm::dyn_cast<llvm::ReturnInst>(value))
      value = ret->getReturnValue();
    llvm::Type* type = getFullyUnwrappedType(value);
    if (!type->isStructTy() && !type->isFloatingPointTy())
      return false;
    return true;
  }

  llvm::Value* copyValueInfo(llvm::Value* dst, llvm::Value* src, tda::TransparentType* dstType = nullptr) {
    using namespace llvm;
    using namespace taffo;
    if (taffoInfo.hasValueInfo(*src)) {
      std::shared_ptr<ValueInfo> dstInfo = taffoInfo.getValueInfo(*src)->clone();
      taffoInfo.setValueInfo(*dst, dstInfo);
    }
    if (dstType)
      taffoInfo.setTransparentType(*dst, dstType->clone());
    else
      taffoInfo.setTransparentType(*dst, tda::TransparentTypeFactory::create(dst->getType()));

    // TODO check old impl because I don't know what this does
    /*if (openMPIndirectMD) {
      if (auto *to = dyn_cast<Instruction>(dst)) {
        to->setMetadata(INDIRECT_METADATA, openMPIndirectMD);
      }
    }*/

    return dst;
  }

  void updateNumericTypeInfo(llvm::Value* value, bool isSigned, int fractionalBits, int bits) {
    using namespace llvm;
    using namespace taffo;
    assert(!taffoInfo.getTransparentType(*value)->isStructTT());
    std::shared_ptr<ScalarInfo> scalarInfo;
    if (taffoInfo.hasValueInfo(*value))
      scalarInfo = std::dynamic_ptr_cast<ScalarInfo>(taffoInfo.getValueInfo(*value));
    else {
      scalarInfo = std::make_shared<ScalarInfo>();
      taffoInfo.setValueInfo(*value, scalarInfo);
    }
    scalarInfo->numericType = std::make_shared<FixedPointInfo>(isSigned, bits, fractionalBits);
  }

  void updateNumericTypeInfo(llvm::Value* value, const ConversionScalarType& convType) {
    updateNumericTypeInfo(value, convType.isSigned(), convType.getFractionalBits(), convType.getBits());
  }

  void convertIndirectCalls(llvm::Module& m);

  void handleKmpcFork(llvm::CallInst* patchedDirectCall, llvm::Function* indirectFunction);

private:
  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  TaffoConvInfo taffoConvInfo;
  const llvm::DataLayout* dataLayout = nullptr;
};

} // namespace taffo

#undef DEBUG_TYPE
