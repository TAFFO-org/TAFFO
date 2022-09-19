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

llvm::cl::opt<bool> MixedDoubleEnabled("mixeddoubleenabled", llvm::cl::value_desc("Double enabled"),
                                       llvm::cl::desc("Set if the double dataype can be used in the resulting mix"),
                                       llvm::cl::init(true));

llvm::cl::opt<int> SinglePrecisionBits("single_precision_frac_bits", llvm::cl::value_desc("Single precision mantissa bits"),
                                 llvm::cl::desc("Number of mantissa bits in single precision float. -1 means the default number of bits"),
                                 llvm::cl::init(-1));

llvm::cl::opt<bool> MixedTripCount(
    "mixedtripcount", 
    llvm::cl::value_desc("Trip-count weighting flag"), 
    llvm::cl::desc("Enables or disables weighting instructions also based on "
                    "the trip count of the enclosing loop if known."),
    llvm::cl::init(true));

llvm::cl::opt<std::string> CostModelFilename("costmodelfilename", llvm::cl::value_desc("Cost model filename"),
                                             llvm::cl::desc("Set the filename to load optimization constant parameter i.e. operation costs"),
                                             llvm::cl::init("DOES-NOT-EXIST"));


extern std::string InstructionSet;
extern llvm::cl::opt<std::string, true> InstructionSetFlag;
                                                           llvm::cl::desc("Set the filename to load wich instruction set are allowed"),
                                                           llvm::cl::location(InstructionSet), llvm::cl::init("DOES-NOT-EXIST"));

#ifndef NDEBUG
extern llvm::cl::opt<std::string> DumpModelFile;
#endif // NDEBUG

#endif // TAFFO_BUILD_ILP_DTA

#undef DEBUG_TYPE

#endif
