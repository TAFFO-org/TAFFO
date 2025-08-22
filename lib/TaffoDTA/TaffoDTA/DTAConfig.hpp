#pragma once

#include <llvm/ADT/Statistic.h>
#include <llvm/Support/CommandLine.h>

#include <string>

#define DEBUG_TYPE "taffo-dta"

extern llvm::cl::opt<int> fracThreshold;
extern llvm::cl::opt<int> totalBits;
extern llvm::cl::opt<int> maxTotalBits;
extern llvm::cl::opt<int> similarBits;
extern llvm::cl::opt<bool> disableTypeMerging;
extern llvm::cl::opt<bool> iterativeMerging;
extern llvm::cl::opt<std::string> UseFloat;
extern llvm::cl::opt<std::string> BufferIDExport;
extern llvm::cl::opt<std::string> BufferIDImport;

/* when adding a new strategy, add an entry here */
enum DtaStrategyType {
  fixedPointOnly,
  floatingPointOnly,
  fixedFloating
};
extern llvm::cl::opt<DtaStrategyType> DtaStrategy;

#ifdef TAFFO_BUILD_ILP_DTA

extern bool hasDouble;
extern bool hasHalf;
extern bool hasQuad;
extern bool hasPPC128;
extern bool hasFP80;
extern bool hasBF16;

extern llvm::cl::opt<bool> MixedMode;
extern llvm::cl::opt<double> MixedTuningENOB;
extern llvm::cl::opt<double> MixedTuningTime;
extern llvm::cl::opt<double> MixedTuningCastingTime;
extern llvm::cl::opt<bool> MixedDoubleEnabled;
extern llvm::cl::opt<bool> MixedTripCount;
extern llvm::cl::opt<std::string> CostModelFilename;

extern std::string InstructionSet;
extern llvm::cl::opt<std::string, true> InstructionSetFlag;

#ifndef NDEBUG
extern llvm::cl::opt<std::string> DumpModelFile;
#endif // NDEBUG

#endif // TAFFO_BUILD_ILP_DTA

#undef DEBUG_TYPE
