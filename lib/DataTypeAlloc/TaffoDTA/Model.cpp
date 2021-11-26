
#include <cassert>
#include <cmath>
#include "Model.h"
#include "llvm/Support/Debug.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "InputInfo.h"
#include "Metadata.h"
#include "TypeUtils.h"
#include "Infos.h"
#include "OptimizerInfo.h"
#include "DebugUtils.h"


#define M_BIG 1000000 


#define DEBUG_TYPE "taffo-dta"

using namespace tuner;
using namespace llvm;
void Model::insertLinearConstraint(const vector<pair<string, double>> &variables, ConstraintType constraintType, double rightSide/*, string&  comment*/) {
    //modelFile << "inserting constraint: ";
    //solver.Add(x + 7 * y <= 17.5)
    //Example of

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
        if(p.second==HUGE_VAL || p.second == -HUGE_VAL){
            constraint->SetCoefficient(variablesPool.at(p.first), p.second>0 ? M_BIG: -M_BIG );
            continue;
        }
        constraint->SetCoefficient(variablesPool.at(p.first), p.second );
    }

    //TODO what to do about comment

  IF_TAFFO_DEBUG {
    dbgs() << "constraint: ";
    bool first = true;
    for (auto v: variables) {
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

void Model::createVariable(const string& varName, double min, double max) {
    llvm::dbgs() <<"Creating variable: " << varName << "\n";
    assert(!isVariableDeclared(varName) && "Variable already declared!");
    variablesPool.insert({varName,solver->MakeIntVar(min, max, varName)});
}

Model::Model(ProblemType type): solver(operations_research::MPSolver::CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")){
        this->problemType=type;
        if (!solver){    
        llvm_unreachable("CBC solver unavailable.");
        }
}

bool Model::finalizeAndSolve() {

  writeOutObjectiveFunction();

  LLVM_DEBUG(

    std::string tmp;
    solver->ExportModelAsLpFormat(false, &tmp);
    llvm::dbgs() << "####LP Format####\n" <<tmp;
    llvm::dbgs() << "\n\n";



  );

  const operations_research::MPSolver::ResultStatus result_status =
      solver->Solve();
  // Check that the problem has an optimal solution.
  if (result_status != operations_research::MPSolver::OPTIMAL) {
    LLVM_DEBUG(
        dbgs() << "[ERROR] There was an error while solving the model!\n\n";);
    return false;
  }



  LLVM_DEBUG(dbgs() << "****************************************************************************************\n");
  LLVM_DEBUG(dbgs() << "****************************************************************************************\n");
  LLVM_DEBUG(dbgs() << "****************************************************************************************\n");
  LLVM_DEBUG(dbgs() << "                                HOUSTON WE HAVE A SOLUTION\n");
  LLVM_DEBUG(dbgs() << "                                HOUSTON WE HAVE A SOLUTION\n");
  LLVM_DEBUG(dbgs() << "                                HOUSTON WE HAVE A SOLUTION\n");
  LLVM_DEBUG(dbgs() << "****************************************************************************************\n");
  LLVM_DEBUG(dbgs() << "****************************************************************************************\n");
  LLVM_DEBUG(dbgs() << "****************************************************************************************\n");
    for(auto& v : variablesPool){
        variableValues.insert(make_pair(v.first, v.second->solution_value()));
                LLVM_DEBUG(dbgs() << v.first << " = " << v.second->solution_value() << "\n";);

    }

  IF_TAFFO_DEBUG {
    dbgs() << "\n************* < TRUMPETS HERE > *************\n";
    dbgs() << "**** THE HOLY OBJECTIVE FUNCTION MEMBERS ****\n";
    dbgs() << "*********************************************\n";
    double castcost = 0, mathcost = 0, enob = 0;
    for(auto& coefficient_id_and_variables: objDeclarationOccoured) {
      for(auto& variable: coefficient_id_and_variables.second){
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
    dbgs() << " Cast Cost = " << castcost << "\n";
    dbgs() << " Math Cost = " << mathcost << "\n";
    dbgs() << " ENOB      = " << enob << "\n";
    dbgs() << "*********************************************\n\n";



  }

    if(variableValues.size() != variablesPool.size()){
        LLVM_DEBUG(dbgs() << "[ERROR] The number of variables in the file and in the model does not match!\n";);
        return false;
    }


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

bool Model::isVariableDeclared(const string& variable) {
    return variablesPool.count(variable)!=0;
}


void Model::insertObjectiveElement(const pair<string, double> &p, string costName, double maxVal) {
    assert(isVariableDeclared(p.first) && "Variable not declared!");



    if(objDeclarationOccoured.find(costName) == objDeclarationOccoured.end() /*!objDeclarationOccoured[costName].second*/){         
        objDeclarationOccoured.insert(std::make_pair(costName, std::vector<std::pair<operations_research::MPVariable*,double>>{}));
    }

    //We use this to normalize the objective value against program complexity changes
    objMaxCosts[costName] = objMaxCosts[costName] + maxVal;

    if(p.second==HUGE_VAL || p.second == -HUGE_VAL){
        
        objDeclarationOccoured[costName].push_back({variablesPool.at(p.first),  p.second>0 ? M_BIG : -M_BIG});

    }else {
        objDeclarationOccoured[costName].push_back({variablesPool.at(p.first),  p.second});
    }


}

void Model::writeOutObjectiveFunction() {
    //solver.Minimize(x + 10 * y)    
    auto obj = solver->MutableObjective();

    switch (problemType) {
        case MIN:
            obj->SetMinimization();
            break;
        case MAX:
            obj->SetMaximization();
            break;
    }


    for(auto& objectives : objDeclarationOccoured){
        for(auto& a : objectives.second){
            obj->SetCoefficient(a.first, a.second * getMultiplier(objectives.first) / objMaxCosts[objectives.first]);
        }
    }


}

double Model::getMultiplier(string var){
    if(var == MODEL_OBJ_CASTCOST){
        return MixedTuningCastingTime;
    }

    if(var == MODEL_OBJ_ENOB){
        return MixedTuningENOB;
    }

    if(var == MODEL_OBJ_MATHCOST){
        return MixedTuningTime;
    }

    llvm_unreachable("Cost variable not declared.");
}

bool Model::VARIABLE_NOT_DECLARED(string var){
    LLVM_DEBUG(dbgs() << "THIS VARIABLE WAS NOT DECLARED >>" << var <<"<<\n";);
    LLVM_DEBUG(dbgs() << "Here is a list of declared vars:\n";);

    for(auto& a : variablesPool){
        LLVM_DEBUG(dbgs() << ">>"<<a.first<<"<<\n";);
    }

    assert(false);
}

double Model::getVariableValue(string variable){
    if(!isVariableDeclared(variable)){
        VARIABLE_NOT_DECLARED(variable);
    }

    auto res = variableValues.find(variable);
    assert(res!=variableValues.end() && "The value of this variable was not found in the model!");

    return res->second;
}

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
