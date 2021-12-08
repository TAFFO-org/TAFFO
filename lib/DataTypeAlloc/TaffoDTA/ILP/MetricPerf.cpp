#include "MetricBase.h"
#include "Optimizer.h"
#include "LoopAnalyzerUtil.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/ADT/APFloat.h"
#include "Optimizer.h"

using namespace std;
using namespace mdutils;


shared_ptr<tuner::OptimizerScalarInfo> MetricPerf::allocateNewVariableForValue(Value *value, shared_ptr<mdutils::FPType> fpInfo, shared_ptr<mdutils::Range> rangeInfo, shared_ptr<double> suggestedMinError,
                                       string functionName, bool insertInList, string nameAppendix, bool insertENOBinMin, bool respectFloatingPointConstraint) {
    assert(!valueHasInfo(value) && "The value considered already have an info!");

    assert(fpInfo && "fpInfo should not be nullptr here!");
    assert(rangeInfo && "rangeInfo should not be nullptr here!");

    if (!functionName.empty()) {
        functionName = functionName.append("_");
    }


    auto& model = getModel();


    string varNameBase(string(functionName).append((std::string)value->getName()).append(nameAppendix));
    std::replace(varNameBase.begin(), varNameBase.end(), '.', '_');
    string varName(varNameBase);

    int counter = 0;
    while (model.isVariableDeclared(varName + "_fixp")) {
        varName = string(varNameBase).append("_").append(to_string(counter));
        counter++;
    }


    LLVM_DEBUG(llvm::dbgs() << "Allocating new variable, will have the following name: " << varName << "\n";);

    auto optimizerInfo = make_shared<tuner::OptimizerScalarInfo>(varName, 0, fpInfo->getPointPos(), fpInfo->getWidth(),
                                                          fpInfo->isSigned(), *rangeInfo, "");


    LLVM_DEBUG(llvm::dbgs() << "Allocating variable " << varName << " with limits [" << optimizerInfo->minBits << ", " << optimizerInfo->maxBits << "];\n";);

    string out;
    raw_string_ostream stream(out);
    value->print(stream);

    //model.insertComment("Stuff for " + stream.str(), 3);

    model.createVariable(optimizerInfo->getFractBitsVariable(), optimizerInfo->minBits, optimizerInfo->maxBits);

    //binary variables for mixed precision
    model.createVariable(optimizerInfo->getFixedSelectedVariable(), 0, 1);
    model.createVariable(optimizerInfo->getFloatSelectedVariable(), 0, 1);
    model.createVariable(optimizerInfo->getDoubleSelectedVariable(), 0, 1);

    if(hasHalf)
    model.createVariable(optimizerInfo->getHalfSelectedVariable(), 0, 1);
    if(hasQuad)
    model.createVariable(optimizerInfo->getQuadSelectedVariable(), 0, 1);
    if(hasFP80)
    model.createVariable(optimizerInfo->getFP80SelectedVariable(), 0, 1);
    if(hasPPC128)
    model.createVariable(optimizerInfo->getPPC128SelectedVariable(), 0, 1);
    if(hasBF16)
    model.createVariable(optimizerInfo->getBF16SelectedVariable(), 0, 1);

    //ENOB propagation, free variable
    model.createVariable(optimizerInfo->getRealEnobVariable(), -BIG_NUMBER, BIG_NUMBER);

    auto constraint = vector<pair<string, double>>();
    int ENOBfloat = getENOBFromRange(rangeInfo, FloatType::Float_float);
    int ENOBdouble = getENOBFromRange(rangeInfo, FloatType::Float_double);
    int ENOBhalf = 0;
    int ENOBquad = 0;
    int ENOBppc128 = 0;
    int ENOBfp80 = 0;
    int ENOBbf16 = 0;       


    //Enob constraints fix
    constraint.clear();
    constraint.push_back(make_pair(optimizerInfo->getRealEnobVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getFractBitsVariable(), -1.0));
    constraint.push_back(make_pair(optimizerInfo->getFixedSelectedVariable(), BIG_NUMBER));
    model.insertLinearConstraint(constraint, tuner::Model::LE, BIG_NUMBER/*, "Enob constraint for fix"*/);

    auto enobconstraint = [&]( int ENOB, const std::string (tuner::OptimizerScalarInfo::* getVariable)(), const char * desc) mutable {
    constraint.clear();
    constraint.push_back(make_pair(optimizerInfo->getRealEnobVariable(), 1.0));
    constraint.push_back(make_pair( ((*optimizerInfo).*getVariable)(), BIG_NUMBER));
    model.insertLinearConstraint(constraint, tuner::Model::LE, BIG_NUMBER + ENOB/*, desc*/);
    };
    //Enob constraints float     
    enobconstraint(ENOBfloat, &tuner::OptimizerScalarInfo::getFloatSelectedVariable, "Enob constraint for float");

    //Enob constraints Double     
    enobconstraint(ENOBdouble, &tuner::OptimizerScalarInfo::getDoubleSelectedVariable, "Enob constraint for double");

    //Enob constraints Half
    if(hasHalf){
    ENOBhalf = getENOBFromRange(rangeInfo, FloatType::Float_half);
    enobconstraint(ENOBhalf, &tuner::OptimizerScalarInfo::getHalfSelectedVariable, "Enob constraint for half");    
    }

    // Enob constraints Quad
    if (hasQuad) {
      ENOBquad = getENOBFromRange(rangeInfo, FloatType::Float_fp128);
      enobconstraint(ENOBquad,
                     &tuner::OptimizerScalarInfo::getQuadSelectedVariable,
                     "Enob constraint for quad");
    }
    // Enob constraints FP80
    
      if (hasFP80){
        ENOBfp80 = getENOBFromRange(rangeInfo, FloatType::Float_x86_fp80);
      enobconstraint(ENOBfp80,
                     &tuner::OptimizerScalarInfo::getFP80SelectedVariable,
                     "Enob constraint for fp80");
    }
    // Enob constraints PPC128
    
      if (hasPPC128){
        ENOBppc128 = getENOBFromRange(rangeInfo, FloatType::Float_ppc_fp128);
      enobconstraint(ENOBppc128,
                     &tuner::OptimizerScalarInfo::getPPC128SelectedVariable,
                     "Enob constraint for ppc128");
    }
    // Enob constraints FP80
    
      if (hasBF16){
        ENOBbf16 = getENOBFromRange(rangeInfo, FloatType::Float_bfloat);
      enobconstraint(ENOBbf16,
                     &tuner::OptimizerScalarInfo::getBF16SelectedVariable,
                     "Enob constraint for bf16");
    }

    constraint.clear();
    constraint.push_back(make_pair(optimizerInfo->getFractBitsVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getFixedSelectedVariable(), -BIG_NUMBER));
    //DO NOT REMOVE THE CAST OR SOMEONE WILL DEBUG THIS FOR AN WHOLE DAY AGAIN
    model.insertLinearConstraint(constraint, tuner::Model::GE, (-BIG_NUMBER-FIX_DELTA_MAX)+((int)fpInfo->getPointPos())/*, "Limit the lower number of frac bits"+to_string(fpInfo->getPointPos())*/);

    int enobMaxCost = max({ENOBfloat, ENOBdouble, (int)fpInfo->getPointPos()});
    
    enobMaxCost = hasHalf ? max(enobMaxCost, ENOBhalf) : enobMaxCost;
    enobMaxCost = hasFP80 ? max(enobMaxCost, ENOBfp80) : enobMaxCost;
    enobMaxCost = hasQuad ? max(enobMaxCost, ENOBquad) : enobMaxCost;
    enobMaxCost = hasPPC128 ? max(enobMaxCost, ENOBppc128) : enobMaxCost;
    enobMaxCost = hasBF16 ? max(enobMaxCost, ENOBbf16) : enobMaxCost;




    if(suggestedMinError){
        /*If we have a suggested min initial error, that is used for error propagation, we should cap the enob to that erro.
         * In facts, it is not really necessary to "unbound" the minimum error while the input variables are not error free
         * Think about a reading from a sensor (ADC) or something similar, the error there will be even if we use a double to
         * store its result. Therefore we limit the enob to a useful value even for floating points.*/


        double errorEnob = getENOBFromError(*suggestedMinError);

        LLVM_DEBUG(llvm::dbgs() << "We have a suggested min error, limiting the enob in the model to " << errorEnob << "\n";);

        constraint.clear();
        constraint.push_back(make_pair(optimizerInfo->getRealEnobVariable(), 1.0));
        model.insertLinearConstraint(constraint, tuner::Model::LE, errorEnob/*, "Enob constraint for error maximal"*/);

        //Capped at max
        enobMaxCost = min(enobMaxCost, (int) errorEnob);
    }

    if(!MixedDoubleEnabled && respectFloatingPointConstraint){
        constraint.clear();
        constraint.push_back(make_pair(optimizerInfo->getDoubleSelectedVariable(), 1.0));
        model.insertLinearConstraint(constraint, tuner::Model::LE, 0/*, "Disable double data type"*/);
    }


    /*//introducing precision cost: the more a variable is precise, the better it is
    model.insertObjectiveElement(make_pair(optimizerInfo->getFractBitsVariable(), (-1) * TUNING_ENOB));

    //La variabile indica solo se il costo è attivo o meno, senza indicare nulla riguardo ENOB
    //Enob is computed from Range

    model.insertObjectiveElement(make_pair(optimizerInfo->getFloatSelectedVariable(), (-1) * TUNING_ENOB * ENOBfloat));
    model.insertObjectiveElement(
            make_pair(optimizerInfo->getDoubleSelectedVariable(), (-1) * TUNING_ENOB * ENOBdouble));*/
    if(insertENOBinMin) {
        model.insertObjectiveElement(
                make_pair(optimizerInfo->getRealEnobVariable(), (-1)), MODEL_OBJ_ENOB, enobMaxCost);
    }

    //Constraint for mixed precision: only one constraint active at one time:
    //_float + _double + _fixed = 1
    constraint.clear();
    constraint.push_back(make_pair(optimizerInfo->getFixedSelectedVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getFloatSelectedVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getDoubleSelectedVariable(), 1.0));
    if(hasHalf)
    constraint.push_back(make_pair(optimizerInfo->getHalfSelectedVariable(), 1.0));
    if(hasQuad)
    constraint.push_back(make_pair(optimizerInfo->getQuadSelectedVariable(), 1.0));    
    if(hasPPC128)
    constraint.push_back(make_pair(optimizerInfo->getPPC128SelectedVariable(), 1.0));  
    if(hasFP80)
    constraint.push_back(make_pair(optimizerInfo->getFP80SelectedVariable(), 1.0)); 
    if(hasBF16)
    constraint.push_back(make_pair(optimizerInfo->getBF16SelectedVariable(), 1.0)); 

    model.insertLinearConstraint(constraint, tuner::Model::EQ, 1/*, "Exactly one selected type"*/);

    //Constraint for mixed precision: if fixed is not the selected data type, force bits to 0
    //x_bits - M * x_fixp <= 0
    constraint.clear();
    constraint.push_back(make_pair(optimizerInfo->getFractBitsVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getFixedSelectedVariable(), -BIG_NUMBER));
    model.insertLinearConstraint(constraint, tuner::Model::LE, 0/*, "If not fix, frac part to zero"*/);


    if (insertInList) {
        saveInfoForValue(value, optimizerInfo);
    }

    return optimizerInfo;

}



shared_ptr<tuner::OptimizerScalarInfo> MetricPerf::allocateNewVariableWithCastCost(Value *toUse, Value *whereToUse) {
    auto info_t = getInfoOfValue(toUse);
    auto& model = getModel();
    if (!info_t) {
        llvm_unreachable("Every value should have an info here!");
    }

    auto info = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerScalarInfo>(info_t);
    if (!info) {
        llvm_unreachable("Here we should only have floating variable, not aggregate.");
    }

    auto originalVar = info->getBaseName();

    string endName = whereToUse->getNameOrAsOperand();
    if(endName.empty()){
        if(auto istr = dyn_cast_or_null<Instruction>(whereToUse)){
            endName = string(istr->getOpcodeName());
        }

    }
    std::replace(endName.begin(), endName.end(), '.', '_');



    string varNameBase((originalVar + ("_CAST_") + endName));

    string varName(varNameBase);

    int counter = 0;
    while (model.isVariableDeclared(varName + "_fixp")) {
        varName = string(varNameBase).append("_").append(to_string(counter));
        counter++;
    }

    LLVM_DEBUG(dbgs() << "Allocating new variable, will have the following name: " << varName << "\n";);


    unsigned minBits = info->minBits;
    unsigned maxBits = info->maxBits;

    auto optimizerInfo = make_shared<tuner::OptimizerScalarInfo>(varName, minBits, maxBits, info->totalBits, info->isSigned, *info->getRange(), info->getOverridedEnob());




    LLVM_DEBUG(dbgs() << "Allocating variable " << varName << " with limits [" << minBits << ", " << maxBits
           << "] with casting cost from " << info->getBaseName() << "\n";);

    string out;
    raw_string_ostream stream(out);
    whereToUse->print(stream);

    //model.insertComment("Constraint for cast for " + stream.str(), 3);

    model.createVariable(optimizerInfo->getFractBitsVariable(), minBits, maxBits);

    //binary variables for mixed precision
    model.createVariable(optimizerInfo->getFixedSelectedVariable(), 0, 1);
    model.createVariable(optimizerInfo->getFloatSelectedVariable(), 0, 1);
    model.createVariable(optimizerInfo->getDoubleSelectedVariable(), 0, 1);
    if(hasHalf)
    model.createVariable(optimizerInfo->getHalfSelectedVariable(), 0, 1);
    if(hasQuad)
    model.createVariable(optimizerInfo->getQuadSelectedVariable(), 0, 1);
    if(hasPPC128)
    model.createVariable(optimizerInfo->getPPC128SelectedVariable(), 0, 1);
    if(hasFP80)
    model.createVariable(optimizerInfo->getFP80SelectedVariable(), 0, 1);
    if(hasBF16)
    model.createVariable(optimizerInfo->getBF16SelectedVariable(), 0, 1);
    //model.createVariable(optimizerInfo->getRealEnobVariable(), -BIG_NUMBER, BIG_NUMBER);

    auto constraint = vector<pair<string, double>>();
    //Constraint for mixed precision: only one constraint active at one time:
    //_float + _double + _fixed = 1
    constraint.clear();
    constraint.push_back(make_pair(optimizerInfo->getFixedSelectedVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getFloatSelectedVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getDoubleSelectedVariable(), 1.0));
    if(hasHalf)
    constraint.push_back(make_pair(optimizerInfo->getHalfSelectedVariable(), 1.0));
     if(hasQuad)
    constraint.push_back(make_pair(optimizerInfo->getQuadSelectedVariable(), 1.0));
    if(hasPPC128)
    constraint.push_back(make_pair(optimizerInfo->getPPC128SelectedVariable(), 1.0));
    if(hasFP80)
    constraint.push_back(make_pair(optimizerInfo->getFP80SelectedVariable(), 1.0));
    if(hasBF16)
    constraint.push_back(make_pair(optimizerInfo->getBF16SelectedVariable(), 1.0));


    model.insertLinearConstraint(constraint, tuner::Model::EQ, 1/*, "exactly 1 type"*/);


    //Real enob is still the same!
    //constraint.clear();
    //constraint.push_back(make_pair(info->getRealEnobVariable(), -1.0));
    //constraint.push_back(make_pair(optimizerInfo->getRealEnobVariable(), 1.0));
    //model.insertLinearConstraint(constraint, Model::LE, 0/*, "The ENOB is less or equal!"*/);

    //Constraint for mixed precision: if fixed is not the selected data type, force bits to 0
    //x_bits - M * x_fixp <= 0
    constraint.clear();
    constraint.push_back(make_pair(optimizerInfo->getFractBitsVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getFixedSelectedVariable(), -BIG_NUMBER));
    model.insertLinearConstraint(constraint, tuner::Model::LE, 0/*, "If no fix, fix frac part = 0"*/);

    auto& cpuCosts = getCpuCosts();
    double maxCastCost = cpuCosts.MaxMinCosts("CAST").first;


    //Variables for costs:

    //Shift cost
    auto C1 = "C1_" + varName;
    auto C2 = "C2_" + varName;
    model.createVariable(C1, 0, 1);
    model.createVariable(C2, 0, 1);

    //Constraint for binary value to activate

    

    constraint.clear();
    constraint.push_back(make_pair(info->getFractBitsVariable(), 1.0));
    constraint.push_back(make_pair(optimizerInfo->getFractBitsVariable(), -1.0));
    constraint.push_back(make_pair(C1, -BIG_NUMBER));
    model.insertLinearConstraint(constraint, tuner::Model::LE, 0/*, "Shift cost 1"*/);

    constraint.clear();
    //Constraint for binary value to activate
    constraint.push_back(make_pair(info->getFractBitsVariable(), -1.0));
    constraint.push_back(make_pair(optimizerInfo->getFractBitsVariable(), 1.0));
    constraint.push_back(make_pair(C2, -BIG_NUMBER));
    model.insertLinearConstraint(constraint, tuner::Model::LE, 0/*, "Shift cost 2"*/);
    /*

    */

    //Casting costs
    //Is correct to only place here the maxCastCost, as only one cast will be active at a time
    model.insertObjectiveElement(make_pair(C1, I_COST * cpuCosts.getCost(tuner::CPUCosts::CAST_FIX_FIX)), MODEL_OBJ_CASTCOST, maxCastCost);
    model.insertObjectiveElement(make_pair(C2, I_COST * cpuCosts.getCost(tuner::CPUCosts::CAST_FIX_FIX)), MODEL_OBJ_CASTCOST, 0);

    //TYPE CAST
    auto costcrosslambda = [&](std::string& variable , tuner::CPUCosts::CostsId cost, const string (tuner::OptimizerScalarInfo::*getFirstVariable)(), const string (tuner::OptimizerScalarInfo::*getSecondVariable)(), const std::string& desc) mutable {
    
    constraint.clear();
    constraint.push_back(make_pair(((*info).*getFirstVariable)(), 1.0));
    constraint.push_back(make_pair(((*optimizerInfo).*getSecondVariable)(), 1.0));
    constraint.push_back(make_pair(variable, -1));
    model.insertLinearConstraint(constraint, tuner::Model::LE, 1/*, desc*/);
    model.insertObjectiveElement(make_pair(variable, I_COST * cpuCosts.getCost(cost)), MODEL_OBJ_CASTCOST, 0);
    };

    int counter2 = 3;
    for (auto& CostsString : cpuCosts.CostsIdValues){


        if(CostsString.find("CAST") == 0 && CostsString.find("CAST_FIX_FIX") == std::string::npos){
            const string (tuner::OptimizerScalarInfo::*first_f )() = nullptr;
            std::size_t first_i = 0;
            std::size_t second_i = 0;
            const char * first_c;
            const char * second_c;            
            const string (tuner::OptimizerScalarInfo::*second_f)() = nullptr;
            std::size_t fixed_i  = CostsString.find("FIX");
            std::size_t float_i  = CostsString.find("FLOAT");
            std::size_t double_i  = CostsString.find("DOUBLE");
            std::size_t quad_i  = CostsString.find("QUAD");
            std::size_t fp80_i  = CostsString.find("FP80");
            std::size_t ppc128_i  = CostsString.find("PPC128");
            std::size_t half_i  = CostsString.find("HALF");
            std::size_t bf16_i  = CostsString.find("BF16");
            if(!hasHalf && half_i != std::string::npos) continue;
            if(!hasQuad && quad_i != std::string::npos) continue;
            if(!hasFP80 && fp80_i != std::string::npos) continue;
            if(!hasPPC128 && ppc128_i != std::string::npos) continue;
            if(!hasBF16 && bf16_i != std::string::npos) continue;

            if (fixed_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getFixedSelectedVariable; first_i = fixed_i; first_c = "Fixed";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getFixedSelectedVariable; second_i = fixed_i; second_c = "Fixed";}
            }
            if (float_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getFloatSelectedVariable; first_i = float_i; first_c = "Float";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getFloatSelectedVariable; second_i = float_i; second_c = "Float";}
            }
            if (double_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getDoubleSelectedVariable; first_i = double_i; first_c = "Double";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getDoubleSelectedVariable; second_i = double_i; second_c = "Double";}
            }
            if (quad_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getQuadSelectedVariable; first_i = quad_i; first_c = "Quad";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getQuadSelectedVariable; second_i = quad_i; second_c = "Quad";}
            }
            if (fp80_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getFP80SelectedVariable; first_i = fp80_i; first_c = "FP80";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getFP80SelectedVariable; second_i = fp80_i; second_c = "FP80";}
            }
            if (ppc128_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getPPC128SelectedVariable; first_i = ppc128_i; first_c = "PPC128";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getPPC128SelectedVariable; second_i = ppc128_i; second_c = "PPC128";}
            }
            if (half_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getHalfSelectedVariable; first_i = half_i; first_c = "Half";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getHalfSelectedVariable; second_i = half_i; second_c = "Half";}
            }
            if (bf16_i != std::string::npos){
                if (first_f == nullptr) {first_f = &tuner::OptimizerScalarInfo::getBF16SelectedVariable; first_i = half_i; first_c = "bf16";}
                else                    {second_f = &tuner::OptimizerScalarInfo::getBF16SelectedVariable; second_i = half_i; second_c = "bf16";}
            }            


            if(first_i > second_i){
                std::swap(first_f,second_f);
                std::swap(first_c,second_c);
            }

            auto CX = std::string("C") + std::to_string(counter2) + "_" + varName;
            counter2++;

            model.createVariable(CX, 0, 1);

            LLVM_DEBUG(llvm::dbgs() << "Inserting constraint " << CX << " " << CostsString << " first " << first_c << " second " << second_c << " cost \n" << cpuCosts.getCost(cpuCosts.decodeId(CostsString)) << " wtih desc " << std::string(first_c) + " to " + std::string(second_c) << "\n" );

            costcrosslambda(CX, cpuCosts.decodeId(CostsString), first_f, 
                        second_f, std::string(first_c) + " to " + std::string(second_c));

                
            
        }
    }
    auto CX = std::string("C") + std::to_string(counter2) + "_" + varName;
    model.createVariable(CX, 0, 1);
    costcrosslambda(CX, cpuCosts.CAST_FIX_FIX, &tuner::OptimizerScalarInfo::getFixedSelectedVariable, &tuner::OptimizerScalarInfo::getFixedSelectedVariable
            ,"Fix to Fix");
    
    return optimizerInfo;
}


void MetricPerf::saveInfoForValue(Value *value, shared_ptr<tuner::OptimizerInfo> optInfo) {
    assert(value && "Value must not be nullptr!");
    assert(optInfo && "optInfo must be a valid info!");
    assert(!valueHasInfo(value) && "Double insertion of value info!");

    LLVM_DEBUG(dbgs() << "Saved info " << optInfo->toString() << " for ";);
    LLVM_DEBUG(value->print(dbgs()););
    LLVM_DEBUG(dbgs()<<"\n";);

    auto&  valueToVariableName = getValueToVariableName();
    auto&  phiWatcher = getPhiWatcher();


    valueToVariableName.insert(make_pair(value, optInfo));

    int closed_phi = 0;
    while (PHINode *phiNode = phiWatcher.getPhiNodeToClose(value)) {
        closePhiLoop(phiNode, value);
        closed_phi++;
    }
    if (closed_phi) {
        LLVM_DEBUG(dbgs() << "Closed " << closed_phi << " PHI loops\n";);
    }



    int closed_mem=0;
    while (auto *phiNode = getMemWatcher().getPhiNodeToClose(value)) {
        closeMemLoop(phiNode, value);
        closed_mem++;
    }
    if (closed_mem) {
        LLVM_DEBUG(dbgs() << "Closed " << closed_mem << " MEM loops\n";);
    }

}



void MetricPerf::closePhiLoop(PHINode *phiNode, Value *requestedValue) {
    LLVM_DEBUG(dbgs() << "Closing PhiNode reference!\n";);
    auto phiInfo = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerScalarInfo>(getInfoOfValue(phiNode));
    auto destInfo = allocateNewVariableWithCastCost(requestedValue, phiNode);

    assert(phiInfo && "phiInfo not available!");
    assert(destInfo && "destInfo not available!");

    string enob_var;

    for (unsigned int index = 0; index < phiNode->getNumIncomingValues(); index++) {
        if (phiNode->getIncomingValue(index) == requestedValue) {
            enob_var = getEnobActivationVariable(phiNode, index);
            break;
        }
    }

    assert(!enob_var.empty() && "Enob var not found!");

    opt->insertTypeEqualityConstraint(phiInfo, destInfo, true);

    auto info1 = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerScalarInfo>(getInfoOfValue(requestedValue));
    auto constraint = vector<pair<string, double>>();
    constraint.clear();
    constraint.push_back(make_pair(phiInfo->getRealEnobVariable(), 1.0));
    constraint.push_back(make_pair(info1->getRealEnobVariable(), -1.0));
    constraint.push_back(make_pair(enob_var, BIG_NUMBER));
    getModel().insertLinearConstraint(constraint, tuner::Model::LE, BIG_NUMBER/*, "Enob: forcing phi enob"*/);
    getPhiWatcher().closePhiLoop(phiNode, requestedValue);
}

void MetricPerf::closeMemLoop(LoadInst *load, Value *requestedValue) {
    LLVM_DEBUG(dbgs() << "Closing MemPhi reference!\n";);
    auto phiInfo = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerScalarInfo>(getInfoOfValue(load));
    //auto destInfo = allocateNewVariableWithCastCost(requestedValue, load);

    assert(phiInfo && "phiInfo not available!");
    //assert(destInfo && "destInfo not available!");

    string enob_var;

    MemorySSA &memssa = getTuner()->getAnalysis<llvm::MemorySSAWrapperPass>(*load->getFunction()).getMSSA();
    taffo::MemSSAUtils memssa_utils(memssa);
    SmallVectorImpl<Value *> &def_vals = memssa_utils.getDefiningValues(load);
    def_vals.push_back(load->getPointerOperand());

    for (unsigned int index = 0; index < def_vals.size(); index++) {
        if (def_vals[index] == requestedValue) {
            enob_var = getEnobActivationVariable(load, index);
            break;
        }
    }

    assert(!enob_var.empty() && "Enob var not found!");

    //as this is a load, it is implicit that the type is equal!
    //insertTypeEqualityConstraint(phiInfo, destInfo, true);

    auto info1 = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerScalarInfo>(getInfoOfValue(requestedValue));
    assert(info1 && "No info for the just saved value!");

    //getModel().insertComment("Closing MEM phi loop...", 3);
    auto constraint = vector<pair<string, double>>();
    constraint.clear();
    constraint.push_back(make_pair(phiInfo->getRealEnobVariable(), 1.0));
    constraint.push_back(make_pair(info1->getRealEnobVariable(), -1.0));
    constraint.push_back(make_pair(enob_var, BIG_NUMBER));
    getModel().insertLinearConstraint(constraint, tuner::Model::LE, BIG_NUMBER/*, "Enob: forcing MEM phi enob"*/);
    getMemWatcher().closePhiLoop(load, requestedValue);
}

void MetricPerf::openPhiLoop(PHINode *phiNode, Value *value) {
    getPhiWatcher().openPhiLoop(phiNode, value);
}

void MetricPerf::openMemLoop(LoadInst *load, Value *value) {
    getMemWatcher().openPhiLoop(load, value);
}



int MetricPerf::getENOBFromError(double error) {
    int enob=floor(log2(error));




    //Fix enob to be at least 0.
    return max(-enob, 0);
}


 int MetricPerf::getENOBFromRange(const shared_ptr<mdutils::Range>& range, mdutils::FloatType::FloatStandard standard) {
    assert(range && "We must have a valid range here!");

    int fractionalDigits;
    int minExponentPower; //eheheh look at this
    switch (standard) {
        case mdutils::FloatType::Float_half:
            fractionalDigits = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEhalf()) - 1;
            minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::IEEEhalf());
            break;        
        case mdutils::FloatType::Float_float:
            fractionalDigits = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEsingle()) - 1;
            minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::IEEEsingle());
            break;
        case mdutils::FloatType::Float_double:
            fractionalDigits = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEdouble()) - 1;
            minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::IEEEdouble());
            break;
        case mdutils::FloatType::Float_bfloat:
            fractionalDigits = llvm::APFloat::semanticsPrecision(llvm::APFloat::BFloat()) - 1;
            minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::BFloat());
            break;
        case mdutils::FloatType::Float_fp128:
            fractionalDigits = llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEquad()) - 1;
            minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::IEEEquad());
            break;
        case mdutils::FloatType::Float_ppc_fp128:
            fractionalDigits = llvm::APFloat::semanticsPrecision(llvm::APFloat::PPCDoubleDouble()) - 1;
            minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::PPCDoubleDouble());
            break;
        case mdutils::FloatType::Float_x86_fp80:
            fractionalDigits = llvm::APFloat::semanticsPrecision(llvm::APFloat::x87DoubleExtended()) - 1;
            minExponentPower = llvm::APFloat::semanticsMinExponent(llvm::APFloat::x87DoubleExtended());
            break;                                            
        default:
            llvm_unreachable("Unsupported type here!");
    }

    //We explore the range in order to understand where to compute the number of bits
    //TODO: implement other less pessimistics algorithm, like medium value, or wathever
    double smallestRepresentableNumber;
    if (range->Min <= 0 && range->Max >= 0) {
        //range overlapping 0
        smallestRepresentableNumber = 0;
    } else if (range->Min >= 0) {
        //both are greater than 0
        smallestRepresentableNumber = range->Min;
    } else {
        //Both are less than 0
        smallestRepresentableNumber = abs(range->Max);
    }

    double exponentOfExponent = log2(smallestRepresentableNumber);
    int exponentInt = floor(exponentOfExponent);

    /*dbgs() << "smallestNumber: " << smallestRepresentableNumber << "\n";
    dbgs() << "exponentInt: " << exponentInt << "\n";*/

    if (exponentInt < minExponentPower) exponentInt = minExponentPower;


    return (-exponentInt) + fractionalDigits;
}


std::string MetricPerf::getEnobActivationVariable(Value *value, int cardinal) {
    assert(value && "Value must not be null!");
    assert(cardinal >= 0 && "Cardinal should be a positive number!");
    string valueName;

    if (auto instr = dyn_cast_or_null<Instruction>(value)) {
        valueName.append(instr->getFunction()->getName().str());
        valueName.append("_");
    }

    if (!value->getName().empty()) {
        valueName.append(value->getNameOrAsOperand());
    } else {
        valueName.append(to_string(value->getValueID()));
        valueName.append("_");
    }

    std::replace(valueName.begin(), valueName.end(), '.', '_');

    assert(!valueName.empty() && "The value should have a name!!!");

    string fname;
    if (auto instr = dyn_cast_or_null<Instruction>(value)) {
        fname = instr->getFunction()->getName().str();
        std::replace(fname.begin(), fname.end(), '.', '_');
    }

    if (!fname.empty()) {
        valueName = fname + "_" + valueName;
    }

    string toreturn = valueName + "_enob_" + to_string(cardinal);

    return toreturn;
}

int MetricPerf::getMinIntBitOfValue(Value *pValue) {
    int bits = -1024;
    double smallestRepresentableNumber;
    auto* tuner = getTuner();

    auto *fp_i = dyn_cast<llvm::ConstantFP>(pValue);
    if (fp_i) {
        APFloat tmp = fp_i->getValueAPF();
        bool losesInfo;
        tmp.convert(APFloatBase::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
        auto a = tmp.convertToDouble();
        LLVM_DEBUG(dbgs() << "Getting max bits of constant " << a << "!\n";);
        smallestRepresentableNumber = abs(a);
    } else {
        if (!tuner->hasInfo(pValue)) {
            LLVM_DEBUG(dbgs() << "No info available for IntBit computation. Using default value\n";);
            return bits;
        }

        auto metadata = tuner->valueInfo(pValue)->metadata;

        if (!metadata) {
            LLVM_DEBUG(dbgs() << "No metadata available for IntBit computation. Using default value\n";);
            return bits;
        }


        auto metadata_InputInfo = tuner::dynamic_ptr_cast_or_null<mdutils::InputInfo>(metadata);
        assert(metadata_InputInfo && "Not an InputInfo!");

        auto range = metadata_InputInfo->IRange;


        if (range->Min <= 0 && range->Max >= 0) {
            LLVM_DEBUG(dbgs() << "The lowest possible number is a 0, infinite ENOB wooooo.\n";);
            return bits;
        } else if (range->Min >= 0) {
            //both are greater than 0
            smallestRepresentableNumber = range->Min;
        } else {
            //Both are less than 0
            smallestRepresentableNumber = abs(range->Max);
        }
    }

    double exponentOfExponent = log2(smallestRepresentableNumber);
    bits = round(exponentOfExponent);

    return bits;

}

int MetricPerf::getMaxIntBitOfValue(Value *pValue) {
    int bits = 1024;
    auto * tuner = getTuner();

    double biggestRepresentableNumber;
    auto *fp_i = dyn_cast<llvm::ConstantFP>(pValue);
    if (fp_i) {
        APFloat tmp = fp_i->getValueAPF();
        bool losesInfo;
        tmp.convert(APFloatBase::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
        LLVM_DEBUG(dbgs() << "Getting max bits of constant!\n";);
        biggestRepresentableNumber = abs(tmp.convertToDouble());
    } else {
        if (!tuner->hasInfo(pValue)) {
            LLVM_DEBUG(dbgs() << "No info available for IntBit computation. Using default value\n";);
            return bits;
        }

        auto metadata = tuner->valueInfo(pValue)->metadata;

        if (!metadata) {
            LLVM_DEBUG(dbgs() << "No metadata available for IntBit computation. Using default value\n";);
            return bits;
        }


        auto metadata_InputInfo = tuner::dynamic_ptr_cast_or_null<mdutils::InputInfo>(metadata);
        assert(metadata_InputInfo && "Not an InputInfo!");

        auto range = metadata_InputInfo->IRange;

        biggestRepresentableNumber = max(abs(range->Min), abs(range->Max));
    }

    double exponentOfExponent = log2(biggestRepresentableNumber);
    bits = round(exponentOfExponent);

    return bits;

}

void MetricPerf::handleSelect(Instruction *instruction, shared_ptr<tuner::ValueInfo> valueInfo) {
    auto *select = dyn_cast<SelectInst>(instruction);

    if (!select->getType()->isFloatingPointTy()) {
        LLVM_DEBUG(dbgs() << "select node with non float value, skipping...\n";);
        return;
    }

    //The select is different from phi because we have all the value in the current basic block, therefore we will have
    // them while computing top down



    if (!select) {
        llvm_unreachable("Could not convert Select instruction to Selectinstruction");
    }


    auto fieldInfo = tuner::dynamic_ptr_cast_or_null<mdutils::InputInfo>(valueInfo->metadata);
    if (!fieldInfo) {
        LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n";);
        return;
    }

    auto fptype = tuner::dynamic_ptr_cast_or_null<mdutils::FPType>(fieldInfo->IType);
    if (!fptype) {
        LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n";);
        return;
    }

    //Allocating variable for result
    shared_ptr<tuner::OptimizerScalarInfo> variable = allocateNewVariableForValue(instruction, fptype, fieldInfo->IRange,
                                                                           fieldInfo->IError,
                                                                           instruction->getFunction()->getName().str());
    auto constraint = vector<pair<string, double>>();
    auto& model = getModel();
    constraint.clear();

    vector<Value *> incomingValues;
    incomingValues.push_back(select->getFalseValue());
    incomingValues.push_back(select->getTrueValue());

    //Yes yes there is not the need to do a loop, but it has the same structure of the phi instruction!
    for (unsigned index = 0; index < incomingValues.size(); index++) {
        Value *op = incomingValues[index];
        if (auto info = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerScalarInfo>(getInfoOfValue(op))) {
            if (info->doesReferToConstant()) {
                //We skip the variable if it is a constant
                LLVM_DEBUG(dbgs() << "[INFO] Skipping ";);
                LLVM_DEBUG(op->print(dbgs()););
                LLVM_DEBUG(dbgs() << " as it is a constant!\n";);
                continue;
            }
        }
        if (!valueHasInfo(op)) {
            LLVM_DEBUG(dbgs() << "[INFO] Skipping ";);
            LLVM_DEBUG(op->print(dbgs()););
            LLVM_DEBUG(dbgs() << " as it is does not have an info!\n";);
            continue;
        }


        string enob_selection = getEnobActivationVariable(instruction, index);
        model.createVariable(enob_selection, 0, 1);
        constraint.push_back(make_pair(enob_selection, 1.0));
    }

    if (constraint.size() > 0) {
        model.insertLinearConstraint(constraint, tuner::Model::EQ, 1/*, "Enob: one selected constraint"*/);
    } else {
        LLVM_DEBUG(dbgs() << "[INFO] All constants or unknown nodes, nothing to do!!!\n";);
        return;
    }

    for (unsigned index = 0; index < incomingValues.size(); index++) {
        LLVM_DEBUG(dbgs() << "[Select] Handlign operator " << index << "...\n";);
        Value *op = incomingValues[index];

        if (auto info = getInfoOfValue(op)) {
            if (auto info2 = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerScalarInfo>(info)) {
                if (info2->doesReferToConstant()) {
                    //We skip the variable if it is a constant
                    LLVM_DEBUG(dbgs() << "[INFO] Skipping ";);
                    LLVM_DEBUG(op->print(dbgs()););
                    LLVM_DEBUG(dbgs() << " as it is a constant!\n";);
                    continue;
                }
            } else {
                LLVM_DEBUG(dbgs() << "Strange select value as it is not a number...\n";);
            }


            LLVM_DEBUG(dbgs() << "[Select] We have infos, treating as usual.\n";);

            auto destInfo = allocateNewVariableWithCastCost(op, select);

            string enob_var = getEnobActivationVariable(select, index);


            assert(!enob_var.empty() && "Enob var not found!");

            opt->insertTypeEqualityConstraint(variable, destInfo, true);


            auto constraint = vector<pair<string, double>>();
            constraint.clear();
            constraint.push_back(make_pair(variable->getRealEnobVariable(), 1.0));
            constraint.push_back(make_pair(destInfo->getRealEnobVariable(), -1.0));
            constraint.push_back(make_pair(enob_var, -BIG_NUMBER));
            model.insertLinearConstraint(constraint, tuner::Model::LE, 0/*, "Enob: forcing select enob"*/);
        }
        //if no info is to skip
    }


}



