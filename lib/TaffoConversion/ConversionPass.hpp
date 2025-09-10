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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Support/CommandLine.h>

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

// Not valid LLVM value but dummy pointer
extern llvm::Value* unsupported;

namespace taffo {

class ConversionPass : public llvm::PassInfoMixin<ConversionPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);

private:
  class ConvTypePolicy : tda::Printable {
  public:
    enum Policy {
      RangeOverHint = 0, // Convert based on value range when needed and possible
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

  TaffoInfo& taffoInfo = TaffoInfo::getInstance();
  TaffoConvInfo taffoConvInfo;
  const llvm::DataLayout* dataLayout = nullptr;
  llvm::DenseMap<llvm::Value*, llvm::Value*> convertedValues;
  llvm::DenseMap<llvm::Function*, llvm::Function*> functionPool;
  llvm::DenseMap<llvm::PHINode*, PhiInfo> phiNodeInfo;

  bool hasConvertedValue(const llvm::Value* value) const { return convertedValues.contains(value); }

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

  void cleanup(const std::vector<llvm::Value*>& queue);
  void cleanUpOriginalFunctions(llvm::Module& m);

  using HeapAllocationsVec = std::vector<std::pair<llvm::Instruction*, tda::TransparentType*>>;
  HeapAllocationsVec collectHeapAllocations(llvm::Module& m);
  void adjustSizeOfHeapAllocations(llvm::Module& m,
                                   const HeapAllocationsVec& oldHeapAllocations,
                                   const HeapAllocationsVec& newHeapAllocations);
  llvm::Value* adjustHeapAllocationSize(llvm::Value* oldSizeValue,
                                        const tda::TransparentType* oldAllocatedType,
                                        const tda::TransparentType* newAllocatedType,
                                        llvm::Instruction* insertionPoint) const;

  llvm::Value* createPlaceholder(llvm::Type* type, llvm::BasicBlock* where, llvm::StringRef name) const;

  void performConversion(const std::vector<llvm::Value*>& queue);
  llvm::Value* convert(llvm::Value* value, std::unique_ptr<ConversionType>* resConvType);

  ValueConvInfo* setConversionResultInfoCommon(llvm::Value* resultValue,
                                               llvm::Value* oldValue = nullptr,
                                               const ConversionType* resultConvType = nullptr);
  void setConstantConversionResultInfo(llvm::Value* resultValue,
                                       llvm::Value* oldValue = nullptr,
                                       const ConversionType* resultConvType = nullptr,
                                       std::unique_ptr<ConversionType>* resConvTypeOwner = nullptr);
  void setConversionResultInfo(llvm::Value* resultValue,
                               llvm::Value* oldValue = nullptr,
                               const ConversionType* resultConvType = nullptr);

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
  llvm::Value* convertAtomicRMW(llvm::AtomicRMWInst* atomicRMW);
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

  // Indirect calls

  /// Retrieve the indirect calls converted into trampolines and re-use the original indirect functions.
  void convertIndirectCalls();
  void handleKmpcFork(llvm::CallBase* trampolineCall, llvm::Function* indirectFunction);

  /** Returns if a function is a library function which shall not be cloned.
   *  @param f The function to check */
  bool isSpecialFunction(const llvm::Function* f) { return f->getName().starts_with("llvm.") || f->empty(); }

  llvm::Value* getConvertedOperand(llvm::Value* value,
                                   const ConversionType& convType,
                                   llvm::Instruction* insertionPoint = nullptr,
                                   const ConvTypePolicy& policy = ConvTypePolicy::RangeOverHint,
                                   std::unique_ptr<ConversionType>* resConvType = nullptr);

  llvm::Value* genConvertConvToConv(llvm::Value* src,
                                    const ConversionScalarType& srcConvType,
                                    const ConversionScalarType& dstConvType,
                                    const ConvTypePolicy& policy,
                                    llvm::Instruction* insertionPoint = nullptr);
  llvm::Value* genConvertFloatToConv(llvm::Value* src,
                                     const ConversionScalarType& dstConvType,
                                     llvm::Instruction* insertionPoint = nullptr);
  llvm::Value* genConvertConvToFloat(llvm::Value* src,
                                     const ConversionScalarType& srcConvType,
                                     const ConversionScalarType& dstConvType);

  llvm::Instruction* getFirstInsertionPointAfter(llvm::Value* value) const;

  llvm::Value*
  copyValueInfo(llvm::Value* dst, const llvm::Value* src, const tda::TransparentType* dstType = nullptr) const;
  void updateNumericTypeInfo(llvm::Value* value, bool isSigned, int fractionalBits, int bits) const;
  void updateNumericTypeInfo(llvm::Value* value, const ConversionScalarType& convType) const;

  void printConversionQueue(const std::vector<llvm::Value*>& queue) const;
  void logFunctionSignature(llvm::Function* fun);
};

} // namespace taffo

#undef DEBUG_TYPE
