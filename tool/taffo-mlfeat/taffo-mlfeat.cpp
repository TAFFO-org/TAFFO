#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <deque>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/InitializePasses.h"
#include "TaffoMLFeaturesAnalysis.h"

using namespace llvm;


cl::OptionCategory TAFFOMLFeatOptions("taffo-mlfeat options");
cl::opt<bool> Verbose("verbose", cl::value_desc("verbose"),
  cl::desc("Enable Verbose Output"),
  cl::init(false));
cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
  cl::desc("<input file>"));


int main(int argc, char *argv[])
{
  /* The initialization section is mostly copied from the
   * source code of opt */
  InitLLVM X(argc, argv);

  LLVMContext c;

  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeScalarOpts(Registry);
  initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeAggressiveInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);
  
  cl::ParseCommandLineOptions(argc, argv, "TAFFO Machine Learning Feature Extractor");

  SMDiagnostic Err;
  std::unique_ptr<Module> m = parseIRFile(InputFilename, Err, c);
  if (!m) {
    std::cerr << "Error reading module " << InputFilename << std::endl;
    return 1;
  }
  
  if (Verbose) {
    std::cerr << "Successfully read Module:" << std::endl;
    std::cerr << " Name: " << m.get()->getName().str() << std::endl;
    std::cerr << " Target triple: " << m->getTargetTriple() << std::endl;
  }
  
  Function *mainfunc = m->getFunction("main");
  if (!mainfunc) {
    std::cout << "No main function found!\n";
    return 1;
  }
  
  /* WARNING: always remember that the various PassManagers do NOT take
   * ownership of modules, but they DO take ownership of PASSES.
   * This means we can't schedule an analysis and then use its results
   * after *PassManager has been run, and we also can't allocate
   * analyses/passes on the stack.
   * TBH I find this behavior kinda dumb. */
  
  /* remove all functions (when possible) */
  for (Function& fun: m->functions()) {
    if (&fun != mainfunc && isFunctionInlinable(&fun)) {
      fun.addFnAttr(Attribute::AlwaysInline);
      fun.setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
  }
  legacy::PassManager passManager;
  passManager.add(createAlwaysInlinerLegacyPass());
  passManager.add(createGlobalDCEPass());
  passManager.add(createLoopSimplifyPass());
  passManager.run(*m);
  
  /* do the actual work; jump to TaffoMLFeatureAnalysisPass.cpp pls */
  legacy::FunctionPassManager funPassManager(m.get());
  funPassManager.add(new TaffoMLFeatureAnalysisPass());
  funPassManager.run(*mainfunc);
  
  return 0;
}
