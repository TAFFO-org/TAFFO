#pragma once

#include "CodeInterpreter.hpp"
#include "PtrCasts.hpp"
#include "TaffoInfo/ValueInfo.hpp"

#include <llvm/Support/Debug.h>

#include <string>

#define DEBUG_TYPE "taffo-vra"
#define DEBUG_HEAD "[TAFFO][VRA]"

namespace taffo {

class VRALogger : public CILogger {
public:
  VRALogger()
  : CILogger(CILK_VRALogger), IndentLevel(0U) {}

  const char* getDebugType() const override { return DEBUG_TYPE; }

  void logBasicBlock(const llvm::BasicBlock* BB) const override {
    assert(BB);
    lineHead();
    llvm::dbgs() << BB->getName() << "\n";
  }

  void logStartFunction(const llvm::Function* F) override {
    assert(F);
    ++IndentLevel;
    llvm::dbgs() << "\n";
    lineHead();
    llvm::dbgs() << "Interpreting function " << F->getName() << "\n";
  }

  void logEndFunction(const llvm::Function* F) override {
    assert(F);
    lineHead();
    llvm::dbgs() << "Finished interpreting function " << F->getName() << "\n\n";
    if (IndentLevel > 0)
      --IndentLevel;
  }

  void logInstruction(const llvm::Value* V) {
    assert(V);
    lineHead();
    llvm::dbgs() << *V << ": ";
  }

  void logRange(const std::shared_ptr<ValueInfo> Range) { llvm::dbgs() << toString(Range); }

  void logRangeln(const std::shared_ptr<ValueInfo> Range) { llvm::dbgs() << toString(Range) << "\n"; }

  void logRange(const std::shared_ptr<Range> Range) { llvm::dbgs() << toString(Range); }

  void logRangeln(const std::shared_ptr<Range> Range) { llvm::dbgs() << toString(Range) << "\n"; }

  void logInfo(const llvm::StringRef Info) { llvm::dbgs() << "(" << Info << ") "; }

  void logInfoln(const llvm::StringRef Info) { llvm::dbgs() << Info << "\n"; }

  void logErrorln(const llvm::StringRef Error) {
    lineHead();
    llvm::dbgs() << Error << "\n";
  }

  void lineHead() const { llvm::dbgs() << DEBUG_HEAD << std::string(IndentLevel * 2U, ' '); }

  static std::string toString(const std::shared_ptr<ValueInfo> Range) {
    if (Range) {
      switch (Range->getKind()) {
      case ValueInfo::K_Scalar: {
        const std::shared_ptr<ScalarInfo> ScalarNode = std::static_ptr_cast<ScalarInfo>(Range);
        return toString(ScalarNode->range);
      }
      case ValueInfo::K_Struct: {
        std::shared_ptr<StructInfo> StructNode = std::static_ptr_cast<StructInfo>(Range);
        std::string Result("{ ");
        for (const std::shared_ptr<ValueInfo>& Field : *StructNode) {
          Result.append(toString(Field));
          Result.append(", ");
        }
        Result.append("}");
        return Result;
      }
      case ValueInfo::K_GetElementPointer: {
        return "GEPNode";
      }
      case ValueInfo::K_Pointer: {
        return "Pointer Node";
      }
      default:
        llvm_unreachable("Unhandled node type.");
      }
    }
    return "null range";
  }

  static std::string toString(const std::shared_ptr<Range> R) {
    if (R) {
      char minstr[32];
      char maxstr[32];
      std::snprintf(minstr, 32, "%.20e", R->min);
      std::snprintf(maxstr, 32, "%.20e", R->max);
      return "[" + std::string(minstr) + ", " + std::string(maxstr) + "]";
    }
    return "null range";
  }

  static bool classof(const CILogger* L) { return L->getKind() == CILK_VRALogger; }

private:
  unsigned IndentLevel;
};

} // namespace taffo

#undef DEBUG_TYPE
