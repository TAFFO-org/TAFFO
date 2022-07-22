//===-- Metadata.h - Metadata Utilities -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declarations of utility functions that handle metadata in TAFFO.
///
//===----------------------------------------------------------------------===//

#ifndef ERRORPROPAGATOR_METADATA_H
#define ERRORPROPAGATOR_METADATA_H

#include "InputInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include <memory>
//#include "../PrecisionAnalysis/TaffoPRA/ErrorInfo.hpp"

#define INPUT_INFO_METADATA "taffo.info"
#define FUNCTION_ARGS_METADATA "taffo.funinfo"
#define STRUCT_INFO_METADATA "taffo.structinfo"
#define CONST_INFO_METADATA "taffo.constinfo"
#define COMP_ERROR_METADATA "taffo.abserror"
#define WRONG_CMP_METADATA "taffo.wrongcmptol"
#define MAX_REC_METADATA "taffo.maxrec"
#define UNROLL_COUNT_METADATA "taffo.unroll"
#define START_FUN_METADATA "taffo.start"
#define TARGET_METADATA "taffo.target"
#define ORIGINAL_FUN_METADATA "taffo.originalCall"
#define CLONED_FUN_METADATA "taffo.equivalentChild"
#define SOURCE_FUN_METADATA "taffo.sourceFunction"
#define INDIRECT_METADATA "taffo.indirectFunction"
#define OMP_DISABLED_METADATA "taffo.ompDisabled"

#define INIT_OCL_TRAMPOLINE_METADATA "taffo.ocl.trampoline"

/* Integer which specifies the distance of the metadata from the
 * original annotation as data flow node counts.
 * Used by VRA to determine the metadata to use as a starting point. */
#define INIT_WEIGHT_METADATA "taffo.initweight"

#define PRA_ERROR_METADATA "taffo.praerr"
#define PRA_STRUCT_METADATA "taffo.prastruct"

namespace mdutils
{

/// Class that converts LLVM Metadata into the in.memory representation.
/// It caches internally the converted data structures
/// to reduce memory consumption and conversion overhead.
/// The returned pointers have the same lifetime as the MetadataManager instance.
class MetadataManager
{
public:
  static MetadataManager &getMetadataManager();

  ///\section Input Info & Struct Info

  /// Retrieve the MDInfo associated to the given value.
  MDInfo *retrieveMDInfo(const llvm::Value *v);

  /// Get the Input Info (Type, Range, Initial Error) attached to I.
  InputInfo *retrieveInputInfo(const llvm::Instruction &I);

  /// Get the Input Info (Type, Range, Initial Error) attached to Global Variable V.
  InputInfo *retrieveInputInfo(const llvm::GlobalObject &V);

  /// Get the StructInfo attached to an Instruction or GlobalVariable.
  StructInfo *retrieveStructInfo(const llvm::Instruction &I);
  StructInfo *retrieveStructInfo(const llvm::GlobalObject &V);

  /// Fill vector ResII with the InputInfo for F's parameters retrieved from F's metadata.
  void retrieveArgumentInputInfo(const llvm::Function &F,
                                 llvm::SmallVectorImpl<MDInfo *> &ResII);

  /// Fill vector ResII with the InputInfo for Constant operands of I
  void retrieveConstInfo(const llvm::Instruction &I,
                         llvm::SmallVectorImpl<InputInfo *> &ResII);

  /// Attach to value u the specified MDInfo node.
  static void setMDInfoMetadata(llvm::Value *u, const MDInfo *mdinfo);

  /// Attach to Instruction I an input info metadata node
  /// containing Type info T, Range, and initial Error.
  static void setInputInfoMetadata(llvm::Instruction &I, const InputInfo &IInfo);

  /// Attach Input Info metadata to global object V.
  /// IInfo contains pointers to type, range and initial error
  /// to be attached to global object V as metadata.
  static void setInputInfoMetadata(llvm::GlobalObject &V, const InputInfo &IInfo);

  /// Attach metadata containing field-wise Input Info for an Instruction
  /// or GlobalVariable of a struct type.
  static void setStructInfoMetadata(llvm::Instruction &I, const StructInfo &SInfo);
  static void setStructInfoMetadata(llvm::GlobalObject &V, const StructInfo &SInfo);

  /// Attach metadata containing types, ranges and initial absolute errors
  /// for each argument of the given function.
  /// AInfo is an array/vector of InputInfo object containing type,
  /// range and initial error of each formal parameter of F.
  /// Each InputInfo object refers to the function parameter with the same index.
  static void setArgumentInputInfoMetadata(llvm::Function &F,
                                           const llvm::ArrayRef<MDInfo *> AInfo);


  static void setConstInfoMetadata(llvm::Instruction &I,
                                   const llvm::ArrayRef<InputInfo *> CInfo);

  ///\section Init Metadata

  static void setInputInfoInitWeightMetadata(llvm::Value *v, int weight);
  int retrieveInputInfoInitWeightMetadata(const llvm::Value *v);

  static void setInputInfoInitWeightMetadata(llvm::Function *f, const llvm::ArrayRef<int> weights);
  static void retrieveInputInfoInitWeightMetadata(llvm::Function *f,
                                                  llvm::SmallVectorImpl<int> &ResWs);

  static void setOpenCLCloneTrampolineMetadata(llvm::Function *F, llvm::Function *KernF);
  static bool retrieveOpenCLCloneTrampolineMetadata(llvm::Function *f, llvm::Function **KernF);

  ///\section Error Propagation Metadata

  /// Attach MaxRecursionCount to the given function.
  /// Attaches metadata containing the maximum number of recursive calls
  /// that is allowed for function F.
  static void
  setMaxRecursionCountMetadata(llvm::Function &F, unsigned MaxRecursionCount);

  /// Read the MaxRecursionCount from metadata attached to function F.
  /// Returns 0 if no metadata have been found.
  static unsigned retrieveMaxRecursionCount(const llvm::Function &F);

  /// Attach unroll count metadata to loop L.
  /// Attaches UnrollCount as the number of times to unroll loop L
  /// as metadata to the terminator instruction of the loop header.
  static void
  setLoopUnrollCountMetadata(llvm::Loop &L, unsigned UnrollCount);

  /// Attach loop unroll metadata to Function F.
  /// Loop unroll counts must be provided in Loop preorder.
  static void
  setLoopUnrollCountMetadata(llvm::Function &F,
                             const llvm::SmallVectorImpl<llvm::Optional<unsigned>> &LUCs);

  /// Read loop unroll count from metadata attached to the header of L.
  static llvm::Optional<unsigned>
  retrieveLoopUnrollCount(const llvm::Loop &L, llvm::LoopInfo *LI = nullptr);

  /// Attach metadata containing the computed error to the given instruction.
  static void setErrorMetadata(llvm::Instruction &I, double Error);

  /// Get the error propagated for I from metadata.
  static double retrieveErrorMetadata(const llvm::Instruction &I);

  /// Attach maximum error tolerance to Cmp instruction.
  /// The metadata are attached only if the comparison may be wrong.
  static void setCmpErrorMetadata(llvm::Instruction &I, const CmpErrorInfo &CEI);

  /// Get the computed comparison error info from metadata attached to I.
  static std::unique_ptr<CmpErrorInfo> retrieveCmpError(const llvm::Instruction &I);

  /// Set this function as a starting point for error analysis.
  static void setStartingPoint(llvm::Function &F);

  /// Returns true if F has been marked as a starting point for error analysis.
  static bool isStartingPoint(const llvm::Function &F);

  /// Mark instruction/global variable I/V as a target with name `Name'.
  static void setTargetMetadata(llvm::Instruction &I, llvm::StringRef Name);
  static void setTargetMetadata(llvm::GlobalObject &V, llvm::StringRef Name);

  /// Get the name of the target of this instruction/global variable,
  /// if it is a target. Returns an empty Optional if it is not a target.
  static llvm::Optional<llvm::StringRef> retrieveTargetMetadata(const llvm::Instruction &I);
  static llvm::Optional<llvm::StringRef> retrieveTargetMetadata(const llvm::GlobalObject &V);


protected:
  llvm::DenseMap<llvm::MDNode *, std::shared_ptr<TType>> TTypes;
  llvm::DenseMap<llvm::MDNode *, std::shared_ptr<Range>> Ranges;
  llvm::DenseMap<llvm::MDNode *, std::shared_ptr<double>> IErrors;
  llvm::DenseMap<llvm::MDNode *, std::shared_ptr<InputInfo>> IInfos;
  llvm::DenseMap<llvm::MDNode *, std::shared_ptr<StructInfo>> StructInfos;

  std::shared_ptr<TType> retrieveTType(llvm::MDNode *MDN);
  std::shared_ptr<Range> retrieveRange(llvm::MDNode *MDN);
  std::shared_ptr<double> retrieveError(llvm::MDNode *MDN);
  std::shared_ptr<InputInfo> retrieveInputInfo(llvm::MDNode *MDN);
  std::shared_ptr<StructInfo> retrieveStructInfo(llvm::MDNode *MDN);

  std::unique_ptr<InputInfo> createInputInfoFromMetadata(llvm::MDNode *MDN);
  std::unique_ptr<StructInfo> createStructInfoFromMetadata(llvm::MDNode *MDN);

  static llvm::Optional<unsigned> retrieveLUCFromHeaderMD(const llvm::Loop &L);
  static llvm::Optional<unsigned>
  retrieveLUCFromFunctionMD(const llvm::Loop &L, llvm::LoopInfo &LI);
  static unsigned getLoopIndex(const llvm::Loop &L, llvm::LoopInfo &LI);
  static llvm::SmallVector<llvm::Optional<unsigned>, 4U>
  retrieveLUCListFromFunctionMD(llvm::Function &F);

private:
  static MetadataManager Instance;

  MetadataManager() = default;
  MetadataManager(const MetadataManager &) = delete;
};

} // namespace mdutils

#endif
