#include <llvm/Analysis/ScalarEvolution.h>
#include "Optimizer.h"
#include "LoopAnalyzerUtil.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/InstIterator.h"
#include "MetricBase.h"


using namespace tuner;
using namespace mdutils;


    Optimizer::Optimizer(Module &mm, TaffoTuner *tuner,  MetricBase*  met, string modelFile, CPUCosts::CostType cType) : model(Model::MIN), module(mm), tuner(tuner), DisabledSkipped(0), metric(met) {
        auto& TTI = tuner->getAnalysis<llvm::TargetTransformInfoWrapperPass>().getTTI(*(mm.begin()));
        if (cType == CPUCosts::CostType::Performance){  
            cpuCosts = CPUCosts(modelFile); 
        } else if (cType == CPUCosts::CostType::Size){
            cpuCosts = CPUCosts(mm,TTI); 
        } 
        
        LLVM_DEBUG(dbgs() << "\n\n\n[WARNING] Mixed precision mode enabled. This is an experimental feature. Use it at your own risk!\n\n\n";);
        cpuCosts.dump();
        LLVM_DEBUG(dbgs() << "ENOB tuning knob: " << to_string(TUNING_ENOB) << "\n";);
        LLVM_DEBUG(dbgs() << "Time tuning knob: " << to_string(TUNING_MATH) << "\n";);
        LLVM_DEBUG(dbgs() << "Time tuning CAST knob: " << to_string(TUNING_CASTING) << "\n";);
        metric->setOpt(this);

        LLVM_DEBUG(dbgs() << "has half: " << to_string(hasHalf) << "\n";);
        LLVM_DEBUG(dbgs() << "has Quad: " << to_string(hasQuad) << "\n";);
        LLVM_DEBUG(dbgs() << "has PPC128: " << to_string(hasPPC128) << "\n";);
        LLVM_DEBUG(dbgs() << "has FP80: " << to_string(hasFP80) << "\n";);
        LLVM_DEBUG(dbgs() << "has BF16: " << to_string(hasBF16) << "\n";);
        }

        Optimizer::~Optimizer() = default;



void Optimizer::initialize() {

     for (llvm::Function &f : module.functions()) {
        LLVM_DEBUG(dbgs() << "\nGetting info of " << f.getName() << ":\n");
        if (f.empty()) {
            continue;
        }
        const std::string name = f.getName().str();
        known_functions[name] = &f;
        functions_still_to_visit[name] = &f;
    }

}

void Optimizer::handleGlobal(GlobalObject *glob, shared_ptr<ValueInfo> valueInfo) {
    LLVM_DEBUG(dbgs() << "handleGlobal called.\n");

    auto * globalVar = dyn_cast_or_null<GlobalVariable>(glob);
    assert(globalVar && "glob is not a global variable!");

    if (!glob->getValueType()->isPointerTy()) {
        if(!valueInfo->metadata->getEnableConversion()){
            LLVM_DEBUG(dbgs() << "Skipping as conversion is disabled!");
            return;
        }
        if (valueInfo->metadata->getKind() == MDInfo::K_Field) {
            LLVM_DEBUG(dbgs() << " ^ This is a real field\n");
            auto fieldInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata);
            if (!fieldInfo) {
                LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n");
                return;
            }

            auto fptype = dynamic_ptr_cast_or_null<FPType>(fieldInfo->IType);
            if (!fptype) {
                LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n");
                return;
            }
            auto optInfo = metric->allocateNewVariableForValue(glob, fptype, fieldInfo->IRange, fieldInfo->IError, "", false);
            metric->saveInfoForValue(glob, make_shared<OptimizerPointerInfo>(optInfo));
        } else if (valueInfo->metadata->getKind() == MDInfo::K_Struct) {
            LLVM_DEBUG(dbgs() << " ^ This is a real structure\n");

            auto fieldInfo = dynamic_ptr_cast_or_null<StructInfo>(valueInfo->metadata);
            if (!fieldInfo) {
                LLVM_DEBUG(dbgs() << "No struct info. Bailing out.\n");
                return;
            }

            auto optInfo = metric->loadStructInfo(glob, fieldInfo, "");
            metric->saveInfoForValue(glob, optInfo);

        } else {
            llvm_unreachable("Unknown metadata!");
        }


    } else {
        if(!valueInfo->metadata->getEnableConversion()){
            LLVM_DEBUG(dbgs() << "Skipping as conversion is disabled!");
            return;
        }
        LLVM_DEBUG(dbgs() << " ^ this is a pointer.\n");

        if (valueInfo->metadata->getKind() == MDInfo::K_Field) {
            LLVM_DEBUG(dbgs() << " ^ This is a real field ptr\n");
            auto fieldInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata);
            if (!fieldInfo) {
                LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n");
                return;
            }

            auto fptype = dynamic_ptr_cast_or_null<FPType>(fieldInfo->IType);
            if (!fptype) {
                LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n");
                return;
            }
            //FIXME: hack, this is done to respect the fact that a pointer (yes, even a simple pointer) may be used by hugly people
            //as array, that are allocated through a malloc. In this way we must use this as a form of bypass. We allocate a new
            //value even if it may be overwritten at some time...

            if(globalVar->hasInitializer() && !globalVar->getInitializer()->isNullValue()){
                LLVM_DEBUG(dbgs() << "Has initializer and it is not a null value! Need more processing!\n");
            }else{
                LLVM_DEBUG(dbgs() << "No initializer, or null value!\n");
                auto optInfo = metric->allocateNewVariableForValue(glob, fptype, fieldInfo->IRange, fieldInfo->IError, "", false);
                //This is a pointer, so the reference to it is a pointer to a pointer yay
                metric->saveInfoForValue(glob, make_shared<OptimizerPointerInfo>(make_shared<OptimizerPointerInfo>(optInfo)));
            }

        } else if (valueInfo->metadata->getKind() == MDInfo::K_Struct) {
            LLVM_DEBUG(dbgs() << " ^ This is a real structure ptr\n");

            auto fieldInfo = dynamic_ptr_cast_or_null<StructInfo>(valueInfo->metadata);
            if (!fieldInfo) {
                LLVM_DEBUG(dbgs() << "No struct info. Bailing out.\n");
                return;
            }

            auto optInfo = metric->loadStructInfo(glob, fieldInfo, "");
            metric->saveInfoForValue(glob, make_shared<OptimizerPointerInfo>(optInfo));

        } else {
            llvm_unreachable("Unknown metadata!");
        }
        return;
    }
}


void Optimizer::handleCallFromRoot(Function *f) {
    //Therefore this should be added as a cost, not simply ignored

    LLVM_DEBUG(dbgs() << "\n============ FUNCTION FROM ROOT: " << f->getName() << " ============\n";);
    const std::string calledFunctionName = f->getName().str();
    LLVM_DEBUG(dbgs() << ("We are calling " + calledFunctionName + " from root\n"););


    auto function = known_functions.find(calledFunctionName);
    if (function == known_functions.end()) {
        LLVM_DEBUG(dbgs() << "Calling an external function, UNSUPPORTED at the moment.\n";);
        return;
    }


    //In teoria non dobbiamo mai pushare variabili per quanto riguarda una chiamata da root
    //Infatti, la chiamata da root implica la compatibilità con codice esterno che si aspetta che non vengano modificate
    //le call ad altri tipi. Per lo stesso motivo non serve nulla per il valore di ritorno.
    /*
    // fetch ranges of arguments
    std::list<shared_ptr<OptimizerInfo>> arg_errors;
    std::list<shared_ptr<OptimizerScalarInfo>> arg_scalar_errors;
    LLVM_DEBUG(dbgs() << ("Arguments:\n"););
    for (auto arg = f->arg_begin(); arg != f->arg_end(); arg++) {
        LLVM_DEBUG(dbgs() << "info for ";);
        (arg)->print(LLVM_DEBUG(dbgs()););
        LLVM_DEBUG(dbgs() << " --> ";);

        //if a variable was declared for type
        auto info = getInfoOfValue(arg);
        if (!info) {
            //This is needed to resolve eventual constants in function call (I'm looking at you, LLVM)
            LLVM_DEBUG(dbgs() << "No error for the argument!\n";);
        } else {
            LLVM_DEBUG(dbgs() << "Got this error: " << info->toString() << "\n";);
        }

        //Even if is a null value, we push it!
        arg_errors.push_back(info);

        //If the error is a scalar, collect it also as a scalar
        auto arg_info_scalar = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info);
        if (arg_info_scalar) {
            arg_scalar_errors.push_back(arg_info_scalar);
        }
        //}
        LLVM_DEBUG(dbgs() << "\n\n";);
    }
    LLVM_DEBUG(dbgs() << ("Arguments end.");*/


    auto it = functions_still_to_visit.find(calledFunctionName);
    if (it != functions_still_to_visit.end()) {
        //We mark the called function as visited from the global queue, so we will not visit it starting from root.
        functions_still_to_visit.erase(calledFunctionName);
        LLVM_DEBUG(dbgs() << "Function " << calledFunctionName << " marked as visited in global queue.\n";);
    } else {
        LLVM_DEBUG(dbgs()<< "[WARNING] We already visited this function, for example when called from another function. Ignoring.\n";);

        return;
    }

    //Allocating variable for result: all returns will have the same type, and therefore a cast, if needed
    //SEE COMMENT BEFORE!
    /*shared_ptr<OptimizerInfo> retInfo;
    if (auto inputInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata)) {
        auto fptype = dynamic_ptr_cast_or_null<FPType>(inputInfo->IType);
        if (fptype) {
            LLVM_DEBUG(dbgs() << fptype->toString(););
            shared_ptr<OptimizerScalarInfo> result = allocateNewVariableForValue(instruction, fptype, inputInfo->IRange,
                                                                                 instruction->getFunction()->getName());
            retInfo = result;
        } else {
            LLVM_DEBUG(dbgs() << "There was an input info but no fix point associated.\n";);
        }
    } else if (auto pInfo = dynamic_ptr_cast_or_null<StructInfo>(valueInfo->metadata)) {
        auto info = loadStructInfo(instruction, pInfo, "");
        saveInfoForValue(instruction, info);
        retInfo = info;
    } else {
        LLVM_DEBUG(dbgs() << "No info available on return value, maybe it is not a floating point call.\n";);
    }*/

    //in retInfo we now have a variable for the return value of the function. Every return should be casted against it!

    //Obviously the type should be sufficient to contain the result

    //In this case we have no known math function.
    //We will have, when enabled, math functions. In this case these will be handled here!

    LLVM_DEBUG(dbgs() << ("The function belongs to the current module.\n"););
    // got the llvm::Function


    // check for recursion
    //no stack check for recursion from root, I hope
    /*size_t call_count = 0;
    for (size_t i = 0; i < call_stack.size(); i++) {
        if (call_stack[i] == f) {
            call_count++;
        }
    }*/

    std::list<shared_ptr<OptimizerInfo>> arg_errors;
    LLVM_DEBUG(dbgs() << ("Arguments:\n"););
    for (auto arg_i = f->arg_begin(); arg_i != f->arg_end(); arg_i++) {
        //Even if is a null value, we push it!
        arg_errors.push_back(nullptr);
    }


    LLVM_DEBUG(dbgs() << ("Processing function...\n"););

    //See comment before to understand why these variable are set to nulls here
    processFunction(*f, arg_errors, nullptr);
    return;
}



void Optimizer::processFunction(Function &f, list<shared_ptr<OptimizerInfo>> argInfo,
                                shared_ptr<OptimizerInfo> retInfo) {
    LLVM_DEBUG(dbgs() << "\n============ FUNCTION " << f.getName() << " ============\n";);

    if (f.arg_size() != argInfo.size()) {
        llvm_unreachable("Sizes should be equal!");
    }

    auto argInfoIt = argInfo.begin();
    for (auto arg = f.arg_begin(); arg != f.arg_end(); arg++, argInfoIt++) {
        if (*argInfoIt) {
            LLVM_DEBUG(dbgs() << "Copying info of this value.\n";);
            metric->saveInfoForValue(&(*arg), *argInfoIt);
        } else {
            LLVM_DEBUG(dbgs() << "No info for this value.\n";);
        }
    }

    //Even if null, we push this on the stack. The return will handle it hopefully
    retStack.push(retInfo);


    //As we have copy of the same function for
    for (inst_iterator iIt = inst_begin(&f), iItEnd = inst_end(&f); iIt != iItEnd; iIt++) {
        //C++ is horrible
        LLVM_DEBUG((*iIt).print(dbgs()););
        LLVM_DEBUG(dbgs() << "     -having-     ";);
        if (!tuner->hasInfo(&(*iIt)) || !tuner->valueInfo(&(*iIt))->metadata) {
            LLVM_DEBUG(dbgs() << "No info available.\n";);
        } else {
            LLVM_DEBUG(dbgs() << tuner->valueInfo(&(*iIt))->metadata->toString() << "\n";);

            if (!tuner->valueInfo(&(*iIt))->metadata->getEnableConversion()) {
                LLVM_DEBUG(dbgs() << "Skipping as conversion is disabled!\n";);
                DisabledSkipped++;
                continue;
            }
        }


        handleInstruction(&(*iIt), tuner->valueInfo(&(*iIt)));
        LLVM_DEBUG(dbgs() << "\n\n";);
    }

    //When the analysis is completed, we remove the info from the stack, as it is no more necessary.
    retStack.pop();

}




shared_ptr<OptimizerInfo> Optimizer::getInfoOfValue(Value *value) {
    assert(value && "Value must not be nullptr!");

    //Global object are constant too but we have already seen them :)
    auto findIt = valueToVariableName.find(value);
    if (findIt != valueToVariableName.end()) {
        return findIt->second;
    }

    if (auto constant = dyn_cast_or_null<Constant>(value)) {
        return metric->processConstant(constant);
    }

    LLVM_DEBUG(dbgs() << "Could not find any info for ");
    LLVM_DEBUG(value->print(dbgs()););
    LLVM_DEBUG(dbgs() << "     :( \n");

    return nullptr;
}


//FIXME: replace with a dynamic version!
#define I_COST 1

void
Optimizer::handleBinaryInstruction(Instruction *instr, const unsigned OpCode, const shared_ptr<ValueInfo> &valueInfos) {
    //We are only handling operations between floating point, as we do not care about other values when building the model
    //This is ok as floating point instruction can only be used inside floating point operations in LLVM! :D
    auto binop = dyn_cast_or_null<BinaryOperator>(instr);


    switch (OpCode) {
        case llvm::Instruction::FAdd:
            metric->handleFAdd(binop, OpCode, valueInfos);
            break;
        case llvm::Instruction::FSub:
            metric->handleFSub(binop, OpCode, valueInfos);
            break;
        case llvm::Instruction::FMul:
            metric->handleFMul(binop, OpCode, valueInfos);
            break;
        case llvm::Instruction::FDiv:;
           metric->handleFDiv(binop, OpCode, valueInfos);
            break;
        case llvm::Instruction::FRem:
            metric->handleFRem(binop, OpCode, valueInfos);
            break;

        case llvm::Instruction::Add:
        case llvm::Instruction::Sub:
        case llvm::Instruction::Mul:
        case llvm::Instruction::UDiv:
        case llvm::Instruction::SDiv:
        case llvm::Instruction::URem:
        case llvm::Instruction::SRem:
        case llvm::Instruction::Shl:
        case llvm::Instruction::LShr:
        case llvm::Instruction::AShr:
        case llvm::Instruction::And:
        case llvm::Instruction::Or:
        case llvm::Instruction::Xor:
            LLVM_DEBUG(dbgs() << "Skipping operation between integers...\n";);
            break;
        default:
            emitError("Unhandled binary operator " + to_string(OpCode)); // unsupported operation
            break;
    }
}



void Optimizer::handleInstruction(Instruction *instruction, shared_ptr<ValueInfo> valueInfo) {
    //This will be a mess. God bless you.
    LLVM_DEBUG(llvm::dbgs() << "Handling instruction " << (instruction->dump() , "\n"));
    auto info = LoopAnalyzerUtil::computeFullTripCount(tuner, instruction);
    LLVM_DEBUG(dbgs() << "Optimizer: got trip count " << info << "\n");

    const unsigned opCode = instruction->getOpcode();
    if (opCode == Instruction::Call) {
        metric->handleCall(instruction, valueInfo);
    } else if (Instruction::isTerminator(opCode)) {
         handleTerminators(instruction, valueInfo);
    } else if (Instruction::isCast(opCode)) {
         metric->handleCastInstruction(instruction, valueInfo);

    } else if (Instruction::isBinaryOp(opCode)) {
        handleBinaryInstruction(instruction, opCode, valueInfo);

    } else if (Instruction::isUnaryOp(opCode)) {
        llvm_unreachable("Not handled.");

    } else {
        switch (opCode) {
            // memory operations
            case llvm::Instruction::Alloca:
                 metric->handleAlloca(instruction, valueInfo);
                break;
            case llvm::Instruction::Load:
                 metric->handleLoad(instruction, valueInfo);
                break;
            case llvm::Instruction::Store:
                 metric->handleStore(instruction, valueInfo);
                break;
            case llvm::Instruction::GetElementPtr:
                 metric->handleGEPInstr(instruction, valueInfo);
                break;
            case llvm::Instruction::Fence:
                emitError("Handling of Fence not supported yet");
                break; // TODO implement
            case llvm::Instruction::AtomicCmpXchg:
                emitError("Handling of AtomicCmpXchg not supported yet");
                break; // TODO implement
            case llvm::Instruction::AtomicRMW:
                emitError("Handling of AtomicRMW not supported yet");
                break; // TODO implement

                // other operations
            case llvm::Instruction::ICmp: {
                LLVM_DEBUG(dbgs() << "Comparing two integers, skipping...\n");
                break;
            }
            case llvm::Instruction::FCmp: {
                 metric->handleFCmp(instruction, valueInfo);
            }
                break;
            case llvm::Instruction::PHI: {
                 metric->handlePhi(instruction, valueInfo);
            }
                break;
            case llvm::Instruction::Select:
                 metric->handleSelect(instruction, valueInfo);;
                break;
            case llvm::Instruction::UserOp1: // TODO implement
            case llvm::Instruction::UserOp2: // TODO implement
                emitError("Handling of UserOp not supported yet");
                break;
            case llvm::Instruction::VAArg: // TODO implement
                emitError("Handling of VAArg not supported yet");
                break;
            case llvm::Instruction::ExtractElement: // TODO implement
                emitError("Handling of ExtractElement not supported yet");
                break;
            case llvm::Instruction::InsertElement: // TODO implement
                emitError("Handling of InsertElement not supported yet");
                break;
            case llvm::Instruction::ShuffleVector: // TODO implement
                emitError("Handling of ShuffleVector not supported yet");
                break;
            case llvm::Instruction::ExtractValue: // TODO implement
                emitError("Handling of ExtractValue not supported yet");
                break;
            case llvm::Instruction::InsertValue: // TODO implement
                emitError("Handling of InsertValue not supported yet");
                break;
            case llvm::Instruction::LandingPad: // TODO implement
                emitError("Handling of LandingPad not supported yet");
                break;
            default:
                emitError("unknown instruction " + std::to_string(opCode));
                break;
        }
        // TODO here be dragons
    } // end else

}

void Optimizer::handleTerminators(llvm::Instruction *term, shared_ptr<ValueInfo> valueInfo) {
    const unsigned opCode = term->getOpcode();
    switch (opCode) {
        case llvm::Instruction::Ret:
             metric->handleReturn(term, valueInfo);
            break;
        case llvm::Instruction::Br:
            // TODO improve by checking condition and relatevely update BB weigths
            // do nothing
            break;
        case llvm::Instruction::Switch:
            emitError("Handling of Switch not implemented yet");
            break; // TODO implement
        case llvm::Instruction::IndirectBr:
            emitError("Handling of IndirectBr not implemented yet");
            break; // TODO implement
        case llvm::Instruction::Invoke:
             metric->handleCall(term, valueInfo);
            break;
        case llvm::Instruction::Resume:
            emitError("Handling of Resume not implemented yet");
            break; // TODO implement
        case llvm::Instruction::Unreachable:
            emitError("Handling of Unreachable not implemented yet");
            break; // TODO implement
        case llvm::Instruction::CleanupRet:
            emitError("Handling of CleanupRet not implemented yet");
            break; // TODO implement
        case llvm::Instruction::CatchRet:
            emitError("Handling of CatchRet not implemented yet");
            break; // TODO implement
        case llvm::Instruction::CatchSwitch:
            emitError("Handling of CatchSwitch not implemented yet");
            break; // TODO implement
        default:
            break;
    }

    return;
}

void Optimizer::emitError(const string& stringhina) {
    LLVM_DEBUG(dbgs() << "[ERROR] " << stringhina << "\n");

}


bool Optimizer::finish() {
    LLVM_DEBUG(dbgs() << "[Phi] Phi node state:\n");
    phiWatcher.dumpState();

    LLVM_DEBUG(dbgs() << "[Mem] MemPhi node state:\n");
    memWatcher.dumpState();

    bool result = model.finalizeAndSolve();

    LLVM_DEBUG(dbgs() << "Skipped conversions due to disabled flag: " << DisabledSkipped << "\n");

    return result;
}

void Optimizer::insertTypeEqualityConstraint(shared_ptr<OptimizerScalarInfo> op1, shared_ptr<OptimizerScalarInfo> op2,
                                             bool forceFixBitsConstraint) {
    assert(op1 && op2 && "One of the info is nullptr!");


    auto constraint = vector<pair<string, double>>();
    //Inserting constraint about of the very same type


    auto eqconstraintlambda = [&](const string (tuner::OptimizerScalarInfo::*getFirstVariable)(), const std::string desc) mutable {
    constraint.clear();
    constraint.push_back(make_pair(((*op1).*getFirstVariable)(), 1.0));
    constraint.push_back(make_pair(((*op2).*getFirstVariable)(), -1.0));
    model.insertLinearConstraint(constraint, Model::EQ, 0/*, desc*/);
    };

    eqconstraintlambda(&tuner::OptimizerScalarInfo::getFixedSelectedVariable,"fix equality");

    eqconstraintlambda(&tuner::OptimizerScalarInfo::getFloatSelectedVariable,"float equality");

    eqconstraintlambda(&tuner::OptimizerScalarInfo::getDoubleSelectedVariable,"double equality");
   
    if(hasHalf)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getHalfSelectedVariable,"Half equality");

    if(hasQuad)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getQuadSelectedVariable,"Quad equality");

    if(hasPPC128)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getPPC128SelectedVariable,"PPC128 equality");

    if(hasFP80)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getFP80SelectedVariable,"FP80 equality");

    if(hasBF16)
    eqconstraintlambda(&tuner::OptimizerScalarInfo::getBF16SelectedVariable,"FP80 equality");

    if (forceFixBitsConstraint) {
        constraint.clear();
        constraint.push_back(make_pair(op1->getFractBitsVariable(), 1.0));
        constraint.push_back(make_pair(op2->getFractBitsVariable(), -1.0));
        model.insertLinearConstraint(constraint, Model::EQ, 0);
    }

}


bool Optimizer::valueHasInfo(Value *value) {
    return valueToVariableName.find(value) != valueToVariableName.end();
}


/*This is ugly as hell, but we use this data type to prevent creating other custom classes for nothing*/
shared_ptr<mdutils::MDInfo> Optimizer::getAssociatedMetadata(Value *pValue) {
    auto res = getInfoOfValue(pValue);
    if (!res) {
        return nullptr;
    }

    if (res->getKind() == OptimizerInfo::K_Pointer) {
        //FIXME: do we support double pointers?
        auto res1 = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(res);
        //Unwrap pointer
        res = res1->getOptInfo();
    }

    return buildDataHierarchy(res);
}

shared_ptr<mdutils::MDInfo> Optimizer::buildDataHierarchy(shared_ptr<OptimizerInfo> info) {
    if (info->getKind() == OptimizerInfo::K_Field) {
        auto i = modelvarToTType(dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info));
        auto result = make_shared<InputInfo>();
        result->IType = i;
        return result;
    } else if (info->getKind() == OptimizerInfo::K_Struct) {
        auto sti = dynamic_ptr_cast_or_null<OptimizerStructInfo>(info);
        auto result = make_shared<StructInfo>(sti->size());
        for (unsigned int i = 0; i < sti->size(); i++) {
            result->setField(i, buildDataHierarchy(sti->getField(i)));
        }

        return result;
    }else if(info->getKind() == OptimizerInfo::K_Pointer){
        auto apr = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(info);
        LLVM_DEBUG(dbgs() << "Unwrapping pointer...\n");
        return buildDataHierarchy(apr->getOptInfo());
    }

    if(!info){
        LLVM_DEBUG(dbgs() << "OptimizerInfo null!\n");
    }else{
        LLVM_DEBUG(dbgs() << "Unknown OptimizerInfo: " << info->toString() << "\n");
    }
    llvm_unreachable("Unnknown data type");
}

shared_ptr<mdutils::TType> Optimizer::modelvarToTType(shared_ptr<OptimizerScalarInfo> scalarInfo) {
    if (!scalarInfo) {
        LLVM_DEBUG(dbgs() << "Nullptr scalar info!");
        return nullptr;
    }
    LLVM_DEBUG(dbgs() << "\nmodel var values\n" );
    double selectedFixed = model.getVariableValue(scalarInfo->getFixedSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getFixedSelectedVariable() << " " << selectedFixed << "\n" );
    double selectedFloat = model.getVariableValue(scalarInfo->getFloatSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getFloatSelectedVariable() << " " << selectedFloat << "\n" );
    double selectedDouble = model.getVariableValue(scalarInfo->getDoubleSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getDoubleSelectedVariable() << " " << selectedDouble << "\n" );
    double selectedHalf = 0;
    double selectedFP80 = 0;
    double selectedPPC128 = 0;
    double selectedQuad = 0;
    double selectedBF16 = 0;


    if(hasHalf){
     selectedHalf = model.getVariableValue(scalarInfo->getHalfSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getHalfSelectedVariable() << " " << selectedHalf << "\n" );
    }
    if(hasQuad){
     selectedQuad = model.getVariableValue(scalarInfo->getQuadSelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getQuadSelectedVariable() << " " << selectedQuad << "\n" );
    }
    if(hasPPC128){
     selectedPPC128 = model.getVariableValue(scalarInfo->getPPC128SelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getPPC128SelectedVariable() << " " << selectedPPC128 << "\n" );
    }
    if(hasFP80){
     selectedFP80 = model.getVariableValue(scalarInfo->getFP80SelectedVariable());
    LLVM_DEBUG(dbgs() << scalarInfo->getFP80SelectedVariable() << " " << selectedFP80 << "\n" );
    }   
    if(hasBF16){
    selectedBF16 = model.getVariableValue(scalarInfo->getBF16SelectedVariable());   
    LLVM_DEBUG(dbgs() << scalarInfo->getBF16SelectedVariable() << " " << selectedBF16 << "\n" );
    }


    double fracbits = model.getVariableValue(scalarInfo->getFractBitsVariable());
    
    assert(selectedDouble + selectedFixed + selectedFloat + selectedHalf + selectedFP80 + selectedPPC128 + selectedQuad + selectedBF16 == 1 &&
           "OMG! Catastrophic failure! Exactly one variable should be selected here!!!");

    if (selectedFixed == 1) {
        StatSelectedFixed++;
        return make_shared<mdutils::FPType>(scalarInfo->getTotalBits(), (int) fracbits, scalarInfo->isSigned);
    }


    if (selectedFloat == 1) {
        StatSelectedFloat++;
        return make_shared<mdutils::FloatType>(FloatType::Float_float, 0);
    }

    if (selectedDouble == 1) {
        StatSelectedDouble++;
        return make_shared<mdutils::FloatType>(FloatType::Float_double, 0);
    }


    if (selectedHalf == 1) {
        StatSelectedHalf++;
        return make_shared<mdutils::FloatType>(FloatType::Float_half, 0);
    }

    if (selectedQuad == 1) {
        StatSelectedQuad++;
        return make_shared<mdutils::FloatType>(FloatType::Float_fp128, 0);
    }

    if (selectedPPC128 == 1) {
        StatSelectedPPC128++;
        return make_shared<mdutils::FloatType>(FloatType::Float_ppc_fp128, 0);
    }

    if (selectedFP80 == 1) {
        StatSelectedFP80++;
        return make_shared<mdutils::FloatType>(FloatType::Float_x86_fp80, 0);
    }

    if (selectedBF16 == 1) {
        StatSelectedBF16++;
        return make_shared<mdutils::FloatType>(FloatType::Float_bfloat, 0);
    }


    llvm_unreachable("Trying to implement a new datatype? look here :D");
}



void Optimizer::printStatInfos() {
    LLVM_DEBUG(dbgs() << "Converted to fix: " << StatSelectedFixed << "\n");
    LLVM_DEBUG(dbgs() << "Converted to float: " << StatSelectedFloat << "\n");
    LLVM_DEBUG(dbgs() << "Converted to double: " << StatSelectedDouble << "\n");
    LLVM_DEBUG(dbgs() << "Converted to half: " << StatSelectedHalf << "\n");

    int total = StatSelectedFixed + StatSelectedFloat + StatSelectedDouble + StatSelectedHalf;

    LLVM_DEBUG(dbgs() << "Conversion entropy as equally distributed variables: " << -(
            ((double)StatSelectedDouble / total) * log2(((double)StatSelectedDouble) / total) +
                    ((double)StatSelectedFloat / total) * log2(((double)StatSelectedFloat) / total) +
                    ((double)StatSelectedDouble / total) * log2(((double)StatSelectedDouble) / total)
            ) << "\n";);

/*
    ofstream statFile;
    statFile.open("./stats.txt", ios::out|ios::trunc);
    assert(statFile.is_open() && "File open failed!");
    statFile << "TOFIX, " << StatSelectedFixed << "\n";
    statFile << "TOFLOAT, " << StatSelectedFloat << "\n";
    statFile << "TODOUBLE, " << StatSelectedDouble << "\n";
    statFile << "TOHALF, " << StatSelectedHalf << "\n";
    statFile.flush();
    statFile.close();
*/
}







