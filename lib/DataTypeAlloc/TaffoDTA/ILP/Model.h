#ifndef TAFFO_DTA_MODEL_H
#define TAFFO_DTA_MODEL_H

#include "ortools/linear_solver/linear_solver.h"
#include "llvm/Support/CommandLine.h"
#include <map>
#include <string>
#include <vector>

#define MODEL_OBJ_CASTCOST "castCostObj"
#define MODEL_OBJ_ENOB "enobCostObj"
#define MODEL_OBJ_MATHCOST "mathCostObj"

extern llvm::cl::opt<double> MixedTuningTime;
extern llvm::cl::opt<double> MixedTuningENOB;
extern llvm::cl::opt<double> MixedTuningCastingTime;
extern llvm::cl::opt<bool> MixedDoubleEnabled;
#ifndef NDEBUG
extern llvm::cl::opt<std::string> DumpModelFile;
#endif

namespace tuner
{
class Model
{
public:
  enum ProblemType {
    MIN,
    MAX
  };

private:
  std::map<const std::string, operations_research::MPVariable *> variablesPool;
  std::map<const std::string, double> variableValues;

  std::map<const std::string, std::vector<std::pair<operations_research::MPVariable *, double>>> objDeclarationOccoured;

  std::map<const std::string, double> objMaxCosts;

  std::vector<std::pair<const std::string, double>> objectiveFunction;
  ProblemType problemType;
  Model() = delete;

  std::unique_ptr<operations_research::MPSolver> solver;

public:
  Model(ProblemType type);

  enum ConstraintType {
    EQ, // Equal
    LE, // Less or equal
    GE  // Greater or equal
  }; // Usually, strict inequalities are not handled by the tools.

  // void createVariable(const std::string &varName);
  void insertLinearConstraint(const std::vector<std::pair<std::string, double>> &variables, ConstraintType constraintType, double rightSide /*, std::string& comment*/);
  bool isVariableDeclared(const std::string &variable);
  bool finalizeAndSolve();
  void createVariable(const std::string &varName, double min, double max);
  void insertObjectiveElement(const std::pair<std::string, double> &variables, std::string costName, double maxValue);
  void writeOutObjectiveFunction();
  bool VARIABLE_NOT_DECLARED(std::string var);
  bool loadResultsFromFile(std::string modelFile);
  double getVariableValue(std::string variable);
  double getMultiplier(std::string var);

#ifndef NDEBUG
  void dumpModel();
  void dumpSolution();
#endif

  // void insertComment(std::string comment, int spaceBefore=0, int spaceAfter=0);
};
} // namespace tuner

#endif
