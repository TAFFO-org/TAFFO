#ifndef DTACONFIG
#define DTACONFIG

#include "llvm/ADT/Statistic.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <string>

#define DEBUG_TYPE "taffo-dta"

extern llvm::cl::opt<int> FracThreshold;
extern llvm::cl::opt<int> TotalBits;
extern llvm::cl::opt<int> SimilarBits;
extern llvm::cl::opt<bool> DisableTypeMerging;
extern llvm::cl::opt<bool> IterativeMerging;
extern llvm::cl::opt<std::string> UseFloat;

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

#endif
