#ifndef TAFFO_VRALOGGER_HPP
#define TAFFO_VRALOGGER_HPP

#include "CodeInterpreter.hpp"
#include "RangeNode.hpp"
#include "llvm/Support/Debug.h"
#include <string>

namespace taffo
{

#define DEBUG_TYPE "taffo-vra"
#define DEBUG_HEAD "[TAFFO][VRA]"

class VRALogger : public CILogger
{
public:
  VRALogger() : CILogger(CILK_VRALogger), IndentLevel(0U) {}

  const char *getDebugType() const override { return DEBUG_TYPE; }

  void logBasicBlock(const llvm::BasicBlock *BB) const override
  {
    assert(BB);
    lineHead();
    llvm::dbgs() << BB->getName() << "\n";
  }

  void logStartFunction(const llvm::Function *F) override
  {
    assert(F);
    ++IndentLevel;
    llvm::dbgs() << "\n";
    lineHead();
    llvm::dbgs() << "Interpreting function " << F->getName() << "\n";
  }

  void logEndFunction(const llvm::Function *F) override
  {
    assert(F);
    lineHead();
    llvm::dbgs() << "Finished interpreting function " << F->getName() << "\n\n";
    if (IndentLevel > 0)
      --IndentLevel;
  }

  void logInstruction(const llvm::Value *V)
  {
    assert(V);
    lineHead();
    llvm::dbgs() << *V << ": ";
  }

  void logRange(const NodePtrT Range)
  {
    llvm::dbgs() << toString(Range);
  }

  void logRangeln(const NodePtrT Range)
  {
    llvm::dbgs() << toString(Range) << "\n";
  }

  void logRange(const range_ptr_t Range)
  {
    llvm::dbgs() << toString(Range);
  }

  void logRangeln(const range_ptr_t Range)
  {
    llvm::dbgs() << toString(Range) << "\n";
  }

  void logInfo(const llvm::StringRef Info)
  {
    llvm::dbgs() << "(" << Info << ") ";
  }

  void logInfoln(const llvm::StringRef Info)
  {
    llvm::dbgs() << Info << "\n";
  }

  void logErrorln(const llvm::StringRef Error)
  {
    lineHead();
    llvm::dbgs() << Error << "\n";
  }

  void lineHead() const
  {
    llvm::dbgs() << DEBUG_HEAD << std::string(IndentLevel * 2U, ' ');
  }

  static std::string toString(const NodePtrT Range)
  {
    if (Range) {
      switch (Range->getKind()) {
      case VRANode::VRAScalarNodeK: {
        const std::shared_ptr<VRAScalarNode> ScalarNode =
            std::static_ptr_cast<VRAScalarNode>(Range);
        return toString(ScalarNode->getRange());
      }
      case VRANode::VRAStructNodeK: {
        std::shared_ptr<VRAStructNode> StructNode =
            std::static_ptr_cast<VRAStructNode>(Range);
        std::string Result("{ ");
        for (const NodePtrT Field : StructNode->fields()) {
          Result.append(toString(Field));
          Result.append(", ");
        }
        Result.append("}");
        return Result;
      }
      case VRANode::VRAGEPNodeK: {
        return "GEPNode";
      }
      case VRANode::VRAPtrNodeK: {
        return "Pointer Node";
      }
      default:
        llvm_unreachable("Unhandled node type.");
      }
    }
    return "null range";
  }

  static std::string toString(const range_ptr_t R)
  {
    if (R) {
      char minstr[32];
      char maxstr[32];
      std::snprintf(minstr, 32, "%.20e", R->min());
      std::snprintf(maxstr, 32, "%.20e", R->max());
      return "[" + std::string(minstr) + ", " + std::string(maxstr) + "]";
    }
    return "null range";
  }

  static bool classof(const CILogger *L)
  {
    return L->getKind() == CILK_VRALogger;
  }

private:
  unsigned IndentLevel;
};

} // namespace taffo

#endif
