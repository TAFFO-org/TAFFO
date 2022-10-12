#include "FloatSizeAnalysis.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <fstream>
#include <string>

#include "Metadata.h"

using namespace llvm;

#define DEBUG_TYPE "float-size-analysis"

cl::opt<std::string> output_file("stats_output_file", cl::desc("Specify filename for float size analysis output"), cl::Optional);

bool FloatSizeAnalysis::runOnModule(llvm::Module &M)
{
  stats = std::list<FloatOpStats>();
  bool Changed = false;
  auto &CTX = M.getContext();
  IRBuilder<> Builder(CTX);

  for (auto &F : M) {
    if (!F.hasName() || F.isDeclaration())
      continue;

    for (auto &BB: F.getBasicBlockList()) {
      auto &InstList = BB.getInstList();
      auto current = InstList.getNextNode(InstList.front());
      while (current != nullptr) {
        auto &Inst = *current;
        auto next = InstList.getNextNode(*current);
        if (!Inst.isDebugOrPseudoInst()) {
          auto* binOp = dyn_cast<BinaryOperator>(&Inst);
          if (binOp) {
            auto opCode = binOp->getOpcode();
            switch (opCode) {
            case llvm::Instruction::FMul:
            case llvm::Instruction::FAdd:
            case llvm::Instruction::FDiv:
            case llvm::Instruction::FSub:
              getOpRanges(binOp);
              break ;
            default:
              break ;
            }
          }
        }
        current = next;
      }
    }
  }

  for (auto &s: stats) {
    errs() << "-----\n";
    errs() << *s.instruction << "\n";
    if (s.op0_range_set) {
      errs() << "op0: (" << s.op0.Min << ", " << s.op0.Max << ")\n";
    } else {
      errs() << "op0: (None, None)\n";
    }
    if (s.op1_range_set) {
      errs() << "op1: (" << s.op1.Min << ", " << s.op1.Max << ")\n";
    } else {
      errs() << "op1: (None, None)\n";
    }
  }

  printStatsCSV();

  return Changed;
}

void FloatSizeAnalysis::printStatsCSV() {
  std::stringstream result;
  result << "op_type,"
            "op1_range_set,op2_range_set,"
            "op0_range_min,op0_range_max,"
            "op1_range_min,op1_range_max,"
            "op0_range_normal,op1_range_normal,"
            "op0_exponent_min,op0_exponent_max,"
            "op1_exponent_min,op1_exponent_max,"
            "max_exponent_diff,"
            "\n";
  for (auto &s: stats) {
    std::string opType;
    switch (s.instruction->getOpcode()) {
    case Instruction::FAdd:
      opType = "fadd";
      break;
    case Instruction::FSub:
      opType = "fsub";
      break;
    case Instruction::FMul:
      opType = "fmul";
      break;
    case Instruction::FDiv:
      opType = "fdiv";
      break;
    default:
      opType = "unknown";
      break;
    }
    result << opType << ",";
    result << s.op0_range_set << ",";
    result << s.op1_range_set << ",";
    result << s.op0.Min << ",";
    result << s.op0.Max << ",";
    result << s.op1.Min << ",";
    result << s.op1.Max << ",";
    result << (finite(s.op0.Min) && finite(s.op0.Max)) << ",";
    result << (finite(s.op1.Min) && finite(s.op1.Max)) << ",";
    result << minExponent(s.op0) << ",";
    result << maxExponent(s.op0) << ",";
    result << minExponent(s.op1) << ",";
    result << maxExponent(s.op1) << ",";
    result << maxExponentDiff(s.op0, s.op1) << ",";
    result << "\n";
  }
  errs() << result.str();
  if (output_file.hasArgStr()) {
    std::ofstream outputFile(output_file.getValue());
    outputFile << result.str();
    outputFile.close();
  }
}

int FloatSizeAnalysis::maxExponentDiff(mdutils::Range& range1, mdutils::Range& range2) {
  int r1_min = minExponent(range1);
  int r1_max = maxExponent(range1);
  int r2_min = minExponent(range2);
  int r2_max = maxExponent(range2);
  return std::max(r1_max - r2_min, r2_max - r1_min);
}

int FloatSizeAnalysis::minExponent(mdutils::Range& range) {
  if (!(finite(range.Min) && finite(range.Max))) {
    return 0;
  }
  int minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::IEEEsingle());
  double smallestRepresentableNumber;
  if (range.Min <= 0 && range.Max >= 0) {
    // range overlapping 0
    smallestRepresentableNumber = 0;
  } else if (range.Min >= 0) {
    // both are greater than 0
    smallestRepresentableNumber = range.Min;
  } else {
    // Both are less than 0
    smallestRepresentableNumber = std::abs(range.Max);
  }

  double exponentOfExponent = std::log2(smallestRepresentableNumber);
  int exponentInt = std::floor(exponentOfExponent);
  if (exponentInt < minExponentPower) {
    exponentInt = minExponentPower;
  }
  return exponentInt;
}

int FloatSizeAnalysis::maxExponent(mdutils::Range& range) {
  if (!(finite(range.Min) && finite(range.Max))) {
    return 0;
  }
  int minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::IEEEsingle());
  double largestRepresentableNumber = std::max(std::abs(range.Max), std::abs(range.Min));
  double exponentOfExponent = std::log2(largestRepresentableNumber);
  int exponentInt = std::ceil(exponentOfExponent);
  if (exponentInt < minExponentPower) {
    exponentInt = minExponentPower;
  }
  return exponentInt;
}

std::unique_ptr<mdutils::Range> FloatSizeAnalysis::rangeFromValue(Value* op) {
  if (auto * op_const = dyn_cast<ConstantFP>(op)) {
    auto val_obj = op_const->getValue();
    double val;
    if (&val_obj.getSemantics() == &APFloatBase::IEEEdouble()) {
      val = val_obj.convertToDouble();
    } else {
      val = val_obj.convertToFloat();
    }
    return std::make_unique<mdutils::Range>(val, val);
  }
  mdutils::MDInfo *op_mdi = mdutils::MetadataManager::getMetadataManager().retrieveMDInfo(op);
  if (op_mdi) {
    if (auto* op_ii = dyn_cast<mdutils::InputInfo>(op_mdi)) {
      if (op_ii->IRange) {
        return std::make_unique<mdutils::Range>(op_ii->IRange->Min, op_ii->IRange->Max);
      }
    }
  }
  return nullptr;
}

void FloatSizeAnalysis::getOpRanges(BinaryOperator *binOp) {
  auto op0 = binOp->getOperand(0);
  auto op1 = binOp->getOperand(1);
  std::unique_ptr<mdutils::Range> op0_range = rangeFromValue(op0), op1_range = rangeFromValue(op1);
  FloatOpStats opStats;
  opStats.instruction = binOp;
  if (op0_range) {
    opStats.op0 = *op0_range;
    opStats.op0_range_set = true;
  } else {
    opStats.op0 = {FP_NAN, FP_NAN};
  }
  if (op1_range) {
    opStats.op1 = *op1_range;
    opStats.op1_range_set = true;
  } else {
    opStats.op1 = {FP_NAN, FP_NAN};
  }
  stats.push_back(opStats);
}

PreservedAnalyses FloatSizeAnalysis::run(llvm::Module &M,
                                        llvm::ModuleAnalysisManager &) {
  bool Changed =  runOnModule(M);

  return (Changed ? llvm::PreservedAnalyses::none()
                  : llvm::PreservedAnalyses::all());
}
