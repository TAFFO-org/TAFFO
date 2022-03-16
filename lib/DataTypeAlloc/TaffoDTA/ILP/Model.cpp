#include "Model.h"
#include "DebugUtils.h"
#include "InputInfo.h"
#include "Metadata.h"
#include "OptimizerInfo.h"
#include "TypeUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cmath>


#define M_BIG 1000000

#define DEBUG_TYPE "taffo-dta"

using namespace tuner;
using namespace llvm;
void Model::insertLinearConstraint(const vector<pair<string, double>> &variables, ConstraintType constraintType, double rightSide /*, string&  comment*/)
{
  // modelFile << "inserting constraint: ";
  // solver.Add(x + 7 * y <= 17.5)
  // Example of

  auto constraint = solver->MakeRowConstraint();
  switch (constraintType) {
  case EQ:
    constraint->SetBounds(rightSide, rightSide);
    break;
  case LE:
    constraint->SetUB(rightSide);
    break;
  case GE:
    constraint->SetLB(rightSide);
    break;
  }

  for (auto p : variables) {
    assert(isVariableDeclared(p.first) || VARIABLE_NOT_DECLARED(p.first));
    if (p.second == HUGE_VAL || p.second == -HUGE_VAL) {
      constraint->SetCoefficient(variablesPool.at(p.first), p.second > 0 ? M_BIG : -M_BIG);
      continue;
    }
    constraint->SetCoefficient(variablesPool.at(p.first), p.second);
  }

  // TODO what to do about comment

  IF_TAFFO_DEBUG
  {
    dbgs() << "constraint: ";
    bool first = true;
    for (auto v : variables) {
      if (first)
        first = false;
      else
        dbgs() << " + ";
      dbgs() << v.first << "*" << constraint->GetCoefficient(variablesPool.at(v.first));
    }
    dbgs() << " in [" << constraint->lb() << ", " << constraint->ub() << "]\n";
  }
}

// void Model::createVariable(const string& varName) {
//     assert(false && "Not working");
//     assert(!isVariableDeclared(varName) && "Variable already declared!");
//     variablesPool.insert(varName);


//     modelFile<<varName<<" = solver.IntVar('"<<varName<<"')\n";
// }

void Model::createVariable(const string &varName, double min, double max)
{
  llvm::dbgs() << "Creating variable: " << varName << "\n";
  assert(!isVariableDeclared(varName) && "Variable already declared!");
  variablesPool.insert({varName, solver->MakeIntVar(min, max, varName)});
}

Model::Model(ProblemType type) : solver(operations_research::MPSolver::CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING"))
{
  this->problemType = type;
  if (!solver) {
    llvm_unreachable("CBC solver unavailable.");
  }
}

bool Model::finalizeAndSolve()
{
  writeOutObjectiveFunction();

  LLVM_DEBUG(llvm::dbgs() << "****************************************************************************************\n");
#ifndef NDEBUG
  dumpModel();
#endif

  LLVM_DEBUG(llvm::dbgs() << "Solving model...\n");
  LLVM_DEBUG(solver->EnableOutput());
  const operations_research::MPSolver::ResultStatus result_status =
      solver->Solve();

  // Check that the problem has an optimal solution.
  if (result_status != operations_research::MPSolver::OPTIMAL && result_status != operations_research::MPSolver::FEASIBLE) {
    LLVM_DEBUG(
        llvm::dbgs() << "[ERROR] There was an error while solving the model!\n");
    switch (result_status) {
      case operations_research::MPSolver::INFEASIBLE:
        LLVM_DEBUG(llvm::dbgs() << "status = INFEASIBLE\n");
        break;
      case operations_research::MPSolver::UNBOUNDED:
        LLVM_DEBUG(llvm::dbgs() << "status = UNBOUNDED\n");
        break;
      case operations_research::MPSolver::ABNORMAL:
        LLVM_DEBUG(llvm::dbgs() << "status = ABNORMAL\n");
        break;
      case operations_research::MPSolver::MODEL_INVALID:
        LLVM_DEBUG(llvm::dbgs() << "status = MODEL_INVALID\n");
        break;
      case operations_research::MPSolver::NOT_SOLVED:
        LLVM_DEBUG(llvm::dbgs() << "status = NOT_SOLVED????\n");
        break;
      default:
        LLVM_DEBUG(llvm::dbgs() << "status = " << result_status << "\n");
    }
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (result_status == operations_research::MPSolver::FEASIBLE) {
    LLVM_DEBUG(llvm::dbgs() << "[WARNING] Model is feasible but solver was stopped by limit, solution is not optimal\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "****************************************************************************************\n");
  LLVM_DEBUG(llvm::dbgs() << "                                HOUSTON WE HAVE A SOLUTION\n");
  LLVM_DEBUG(llvm::dbgs() << "****************************************************************************************\n");

  for (auto &v : variablesPool) {
    variableValues.insert(make_pair(v.first, v.second->solution_value()));
  }

#ifndef NDEBUG
  dumpSolution();
#endif

  if (variableValues.size() != variablesPool.size()) {
    LLVM_DEBUG(dbgs() << "[ERROR] The numbers of variables in the program and in the model do not match!\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "****************************************************************************************\n");
  return true;
}

// bool Model::loadResultsFromFile(string modelFile) {
//     fstream fin;

//     fin.open(modelFile, ios::in);

//     assert(fin.is_open() && "Cannot open results file!");

//     string line, field, temp;
//     vector<string> row;

//     //for each line in the file
//     int nline=0;
//     while (getline(fin, line)) {

//         //read the file until a newline is found (discarded from final string)
//         row.clear();
//         double value = 0;
//         nline++;


//         //Generate a stream in order to be used by getLine
//         stringstream lineStream(line);
//         //llvm::dbgs() << "Line: " << line << "\n";

//         while (getline(lineStream, field, ',')) {
//             row.push_back(field);
//         }

//         if (row.size() != 2) {
//             LLVM_DEBUG(llvm::dbgs() << "Malformed line found: [" << line << "] on line"<< nline << ", skipping...\n";);
//             continue;
//         }

//         string varName = row[0];
//         value = stod(row[1]);

//         if(varName == "__ERROR__"){
//             if(value==0){
//                 LLVM_DEBUG(dbgs() << "The model was solved correctly!\n";);
//             }else{
//                 LLVM_DEBUG(dbgs() << "[ERROR] The Python solver signalled an
// error!\n\n";);
//                 return false;
//             }
//             //Skips any other computation as this is a state message
//             continue;
//         }

//         if(varName == "__COST_ENOB__"){
//             costEnob = value;
//             continue;
//         }

//         if(varName == "__COST_TIME__"){
//             costTime = value;
//             continue;
//         }

//         if(varName == "__COST_CONV__"){
//             costCast=value;
//             continue;
//         }

//         if(!isVariableDeclared(varName)){
//             LLVM_DEBUG(dbgs() << "Trying to load results for an unknown variable!\nThis may be signal of a more problematic error!\n\n";);
//             VARIABLE_NOT_DECLARED(varName);
//         }

//         if(variableValues.find(varName) != variableValues.end()){
//             LLVM_DEBUG(dbgs() << "Found duplicated result: [" << line << "], skipping...\n";);
//             continue;
//         }

//         variableValues.insert(make_pair(varName, value));


//     }

//     if(variableValues.size() != variablesPool.size()){
//         LLVM_DEBUG(dbgs() << "[ERROR] The number of variables in the file and in the model does not match!\n";);
//         return false;
//     }
//     return true;
// }

bool Model::isVariableDeclared(const string &variable)
{
  return variablesPool.count(variable) != 0;
}


void Model::insertObjectiveElement(const pair<string, double> &p, string costName, double maxVal)
{
  assert(isVariableDeclared(p.first) && "Variable not declared!");


  if (objDeclarationOccoured.find(costName) == objDeclarationOccoured.end() /*!objDeclarationOccoured[costName].second*/) {
    objDeclarationOccoured.insert(std::make_pair(costName, std::vector<std::pair<operations_research::MPVariable *, double>>{}));
  }

  // We use this to normalize the objective value against program complexity changes
  objMaxCosts[costName] = objMaxCosts[costName] + maxVal;

  if (p.second == HUGE_VAL || p.second == -HUGE_VAL) {

    objDeclarationOccoured[costName].push_back({variablesPool.at(p.first), p.second > 0 ? M_BIG : -M_BIG});

  } else {
    objDeclarationOccoured[costName].push_back({variablesPool.at(p.first), p.second});
  }
}

void Model::writeOutObjectiveFunction()
{
  auto obj = solver->MutableObjective();

  switch (problemType) {
  case MIN:
    obj->SetMinimization();
    break;
  case MAX:
    obj->SetMaximization();
    break;
  }

  for (auto &objectives : objDeclarationOccoured) {
    for (auto &a : objectives.second) {
      obj->SetCoefficient(a.first, a.second * getMultiplier(objectives.first) / objMaxCosts[objectives.first]);
    }
  }
}

double Model::getMultiplier(string var)
{
  if (var == MODEL_OBJ_CASTCOST) {
    return MixedTuningCastingTime;
  }

  if (var == MODEL_OBJ_ENOB) {
    return MixedTuningENOB;
  }

  if (var == MODEL_OBJ_MATHCOST) {
    return MixedTuningTime;
  }

  llvm_unreachable("Cost variable not declared.");
}

bool Model::VARIABLE_NOT_DECLARED(string var)
{
  LLVM_DEBUG(dbgs() << "THIS VARIABLE WAS NOT DECLARED >>" << var << "<<\n";);
  LLVM_DEBUG(dbgs() << "Here is a list of declared vars:\n";);

  for (auto &a : variablesPool) {
    LLVM_DEBUG(dbgs() << ">>" << a.first << "<<\n";);
  }

  assert(false);
}

double Model::getVariableValue(string variable)
{
  if (!isVariableDeclared(variable)) {
    VARIABLE_NOT_DECLARED(variable);
  }

  auto res = variableValues.find(variable);
  assert(res != variableValues.end() && "The value of this variable was not found in the model!");

  return res->second;
}

#ifndef NDEBUG
void Model::dumpModel()
{
  std::string tmp;
  solver->ExportModelAsLpFormat(false, &tmp);
  if (DumpModelFile != "") {
    LLVM_DEBUG(llvm::dbgs() << "Dumping model to " << DumpModelFile << "...");
    std::error_code EC;
    llvm::raw_fd_ostream model_file(DumpModelFile, EC);
    if (!EC) {
      model_file << tmp;
      model_file.close();
      LLVM_DEBUG(llvm::dbgs() << " done.\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << " failed. An error occurred while trying to dump the model: "
                 << EC.message() << '\n');
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Dumping model to log:");
    LLVM_DEBUG(llvm::dbgs() << tmp << "\n");
  }
}

void Model::dumpSolution()
{
  std::string tmp;
  llvm::raw_string_ostream acc(tmp);

  acc << "Variable Name = Solution Value\n";
  for (auto &v : variablesPool) {
    acc << v.first << " = " << v.second->solution_value() << "\n";
  }

  acc << "________________________________________________________________________________\n\n";
  acc << "Objective Function Members:\n";
  double castcost = 0, mathcost = 0, enob = 0;
  for (auto &coefficient_id_and_variables : objDeclarationOccoured) {
    for (auto &variable : coefficient_id_and_variables.second) {
      std::string coefficient_id = coefficient_id_and_variables.first;
      if (coefficient_id == MODEL_OBJ_CASTCOST) {
        castcost += variable.first->solution_value() * variable.second;
      } else if (coefficient_id == MODEL_OBJ_MATHCOST) {
        mathcost += variable.first->solution_value() * variable.second;
      } else if (coefficient_id == MODEL_OBJ_ENOB) {
        enob += variable.first->solution_value() * variable.second;
      } else {
        llvm_unreachable("and why is it now that we discover that we have a coefficient id that does not exist?");
      }
    }
  }
  acc << " Cast Cost = " << castcost << "\n";
  acc << " Math Cost = " << mathcost << "\n";
  acc << " ENOB      = " << enob << "\n";
  acc.flush();

  if (DumpModelFile != "") {
    std::string solution_file_name = DumpModelFile + ".sol";
    LLVM_DEBUG(llvm::dbgs() << "Dumping solution to " << solution_file_name << "...");
    std::error_code EC;
    llvm::raw_fd_ostream solution_file(solution_file_name, EC);
    if (!EC) {
      solution_file << tmp;
      solution_file.close();
      LLVM_DEBUG(llvm::dbgs() << " done.\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << " failed. An error occurred while trying to dump the solution: "
                 << EC.message() << '\n');
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Dumping solution to log:\n");
    LLVM_DEBUG(llvm::dbgs() << tmp << "\n");
  }
}
#endif

// void Model::insertComment(string comment, int spaceBefore, int spaceAfter) {
//     int i;

//     for(i=0; i<spaceBefore; i++){
//         modelFile << "\n";
//     }


//     //delete newline
//     std::replace(comment.begin(), comment.end(), '\n', '_');
//     std::replace(comment.begin(), comment.end(), '\r', '_');
//     modelFile << "#" << comment << "\n";

//     for(i=0; i<spaceAfter; i++){
//         modelFile << "\n";
//     }

//
// }
