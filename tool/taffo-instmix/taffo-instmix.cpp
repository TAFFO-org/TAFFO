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
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "InstructionMix.h"

using namespace llvm;


cl::OptionCategory TAFFOInstMixOptions("taffo-instmix options");
cl::opt<bool> Verbose("verbose", cl::value_desc("verbose"),
  cl::desc("Enable Verbose Output"),
  cl::init(false));
cl::opt<bool> CountCallSite("callsites", cl::value_desc("callsites"),
  cl::desc("Count call instructions to profiled functions"),
  cl::init(false));
cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
  cl::desc("<input file>"));


struct block_eval_status {
  BasicBlock *block;
  int eval;
  
  block_eval_status(BasicBlock *bb, int e): block(bb), eval(e) {};
};


bool analyze_function(InstructionMix&, Function *, std::unordered_set<BasicBlock *>&, int &);
void analyze_basic_block(InstructionMix&, BasicBlock *, std::unordered_set<BasicBlock *>&, int &);


void analyze_basic_block(InstructionMix& imix, BasicBlock *bb, std::unordered_set<BasicBlock *>& countedbbs, int &eval)
{
  if (Verbose) {
    raw_os_ostream stm(std::cerr);
    stm << "  BasicBlock: ";
    bb->printAsOperand(stm, true);
    stm << '\n';
  }
  countedbbs.insert(bb);
  
  for (auto iter3 = bb->begin(); iter3 != bb->end(); iter3++) {
    Instruction &inst = *iter3;
    
    int delim = isDelimiterInstruction(&inst);
    eval += delim;
    if (delim != 0)
      continue;
    
    if (isSkippableInstruction(&inst))
      continue;

    Function *opnd = nullptr;
    CallBase *call = dyn_cast<CallBase>(&inst);
    if (call)
      opnd = call->getCalledFunction();
    
    if (opnd) {
      bool success = analyze_function(imix, opnd, countedbbs, eval);
      if (success && !CountCallSite)
        continue;
    }

    if (!eval)
      continue;
    
    imix.updateWithInstruction(&inst);
  }
  
  return;
}


bool analyze_function(InstructionMix& imix, Function *f, std::unordered_set<BasicBlock *>& countedbbs, int &eval)
{
  if (Verbose)
    std::cerr << " Function: " << f->getName().str() << std::endl;
  
  auto &bblist = f->getBasicBlockList();
  if (bblist.empty()) {
    return false;
  }
  
  if (countedbbs.find(&(f->getEntryBlock())) != countedbbs.end()) {
    if (Verbose)
      std::cerr << "Recursion!" << std::endl;
    return true;
  }
  
  std::deque<block_eval_status> queue;
  queue.push_back(block_eval_status(&f->getEntryBlock(), eval));
  while (queue.size() > 0) {
    block_eval_status top = queue.front();
    queue.pop_front();
    
    analyze_basic_block(imix, top.block, countedbbs, top.eval);
    eval = top.eval;
    
    Instruction *term = top.block->getTerminator();
    assert(term && "denormal bb found; abort");
    int numbb = term->getNumSuccessors();
    for (int bbi = 0; bbi < numbb; bbi++) {
      BasicBlock *nextbb = term->getSuccessor(bbi);
      if (countedbbs.find(nextbb) != countedbbs.end()) {
        if (Verbose)
          std::cerr << "Loop!" << std::endl;
        continue;
      }
      
      queue.push_back(block_eval_status(nextbb, top.eval));
    }
  }
  
  return true;
}


int main(int argc, char *argv[])
{
  cl::ParseCommandLineOptions(argc, argv);

  LLVMContext c;
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

  int eval = 0;
  std::unordered_set<BasicBlock *> bbs;
  InstructionMix imix;
  
  Function *mainfunc = m->getFunction("main");
  if (!mainfunc) {
    std::cout << "No main function found!\n";
  } else {
    analyze_function(imix, mainfunc, bbs, eval);
  }

  std::cout << "* " << imix.ninstr << std::endl;
  for (auto it = imix.stat.begin(); it != imix.stat.end(); it++) {
    std::cout << it->first << " " << it->second;
    std::cout << std::endl;
  }

  return 0;
}
