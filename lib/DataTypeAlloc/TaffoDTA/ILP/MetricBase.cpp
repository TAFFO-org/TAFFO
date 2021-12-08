#include "MetricBase.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Debug.h"
#include "Optimizer.h"


extern llvm::cl::opt<int> TotalBits;
extern llvm::cl::opt<int> FracThreshold;

using namespace tuner;
using namespace mdutils;
using namespace std;
using namespace taffo;




  bool MetricBase::valueHasInfo(llvm::Value *value)  { return opt->valueHasInfo(value); }

  tuner::Model &MetricBase::getModel()  { return opt->model; }

  std::unordered_map<std::string, llvm::Function *> &
  MetricBase::getFunctions_still_to_visit()  {
    return opt->functions_still_to_visit;
  }

  std::vector<llvm::Function *> &MetricBase::getCall_stack()  { return opt->call_stack; }

  DenseMap<llvm::Value *, std::shared_ptr<tuner::OptimizerInfo>> &
  MetricBase::getValueToVariableName(){
    return opt->valueToVariableName;
  }

  std::stack<shared_ptr<tuner::OptimizerInfo>> &MetricBase::getRetStack()  {
    return opt->retStack;
  }

  void MetricBase::addDisabledSkipped()  { opt->DisabledSkipped++; }

  shared_ptr<tuner::OptimizerInfo> MetricBase::getInfoOfValue(llvm::Value *value)  {
    return opt->getInfoOfValue(value);
  }

  tuner::TaffoTuner *MetricBase::getTuner()  { return opt->tuner; }
  tuner::PhiWatcher &MetricBase::getPhiWatcher()  { return opt->phiWatcher; }
  std::unordered_map<std::string, llvm::Function *> &MetricBase::getKnown_functions()  {
    return opt->known_functions;
  }
  tuner::MemWatcher &MetricBase::getMemWatcher()  { return opt->memWatcher; }
  tuner::CPUCosts &MetricBase::getCpuCosts()  { return opt->cpuCosts; }



shared_ptr<tuner::OptimizerInfo> MetricBase::processConstant(Constant *constant) {
    //Constant variable should, in general, not be saved anywhere.
    //In fact, the same constant may be used in different ways, but by the fact that
    //it is a constant, it may be modified in the final code
    //For example, a double 1.00 can become a float 1.00 in one place and a fixp 1 in another!
    LLVM_DEBUG(dbgs() << "Processing constant...\n";);

    if(dyn_cast_or_null<GlobalObject>(constant)){
        if(getTuner()->hasInfo(constant)){
            llvm_unreachable("This should already have been handled!");
        }else{
            LLVM_DEBUG(dbgs() << "Trying to process a non float global...\n";);
            return nullptr;
        }

    }

    if (dyn_cast_or_null<ConstantData>(constant)) {
        //ATM: only handling FP types, should be enough
        if (auto constantFP = dyn_cast_or_null<ConstantFP>(constant)) {
            LLVM_DEBUG(dbgs() << "Processing FPconstant...\n";);

            APFloat tmp = constantFP->getValueAPF();
            bool losesInfo;
            tmp.convert(APFloatBase::IEEEdouble(), APFloat::rmNearestTiesToEven , &losesInfo);
            double a = tmp.convertToDouble();

            double min, max;
            min = a;
            max = a;

            Range rangeInfo(min, max);


            FixedPointTypeGenError fpgerr;
            FPType fpInfo = fixedPointTypeFromRange(rangeInfo, &fpgerr, TotalBits, FracThreshold, 64, TotalBits);
            if (fpgerr != FixedPointTypeGenError::NoError) {
                LLVM_DEBUG(dbgs() << "Error generating infos for constant propagation!" << "\n";);
                return nullptr;
            }

            string fname = "ConstantValue";
            //ENOB should not be considered for constant.... It is a constant and will be converted as best as possible
            //WE DO NOT SAVE CONSTANTS INFO!
            auto info = allocateNewVariableForValue(constantFP, make_shared<FPType>(fpInfo), make_shared<Range>(rangeInfo), nullptr, fname, false, "", false);
            info->setReferToConstant(true);
            return info;
        }

        LLVM_DEBUG(dbgs() << "[ERROR] handling unknown ConstantData, I don't know what to do: ";);
        LLVM_DEBUG(constant->print(dbgs()););
        LLVM_DEBUG(dbgs() << "\n";);
        return nullptr;
    }

    if (auto constantExpr = dyn_cast_or_null<ConstantExpr>(constant)) {
        if (constantExpr->isGEPWithNoNotionalOverIndexing()) {
            return handleGEPConstant(constantExpr);
        }
        LLVM_DEBUG(dbgs() << "Unknown constant expr!\n";);
        return nullptr;
    }


    LLVM_DEBUG(dbgs() << "Cannot handle ";);
    LLVM_DEBUG(constant->print(dbgs()););
    LLVM_DEBUG(dbgs() << "!\n\n";);
    LLVM_DEBUG(constant->getType()->print(dbgs()););
    llvm_unreachable("Constant not handled!");
}


shared_ptr<OptimizerInfo> MetricBase::handleGEPConstant(const ConstantExpr *cexp_i) {
    //The first operand is the beautiful object
    Value *operand = cexp_i->getOperand(0U);


    Type *source_element_type =
            cast<PointerType>(operand->getType()->getScalarType())->getElementType();
    std::vector<unsigned> offset;

    //We compute all the offsets that will be used in our "data structure" to navigate it, to reach the correct range
    if (extractGEPOffset(source_element_type,
                         iterator_range<User::const_op_iterator>(cexp_i->op_begin() + 1,
                                                                 cexp_i->op_end()),
                         offset)) {


        LLVM_DEBUG(dbgs() << "Exctracted offset: [";);
        for (unsigned int i = 0; i < offset.size(); i++) {
            LLVM_DEBUG(dbgs() << offset[i] << ", ";);
        }
        LLVM_DEBUG(dbgs() << "]\n";);
        //When we load an address from a "thing" we need to store a reference to it in order to successfully update the error
        auto optInfo_t = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(getInfoOfValue(operand));
        if (!optInfo_t) {
            LLVM_DEBUG(dbgs() << "Probably trying to access a non float element, bailing out.\n";);
            return nullptr;
        }


        auto optInfo = optInfo_t->getOptInfo();
        if (!optInfo) {
            LLVM_DEBUG(dbgs() << "Probably trying to access a non float element, bailing out.\n";);
            return nullptr;
        }
        //This will only contain displacements for struct fields...
        for (unsigned int i = 0; i < offset.size(); i++) {
            auto structInfo = dynamic_ptr_cast_or_null<OptimizerStructInfo>(optInfo);
            if (!structInfo) {
                LLVM_DEBUG(dbgs() << "Probably trying to access a non float element, bailing out.\n";);
                return nullptr;
            }

            optInfo = structInfo->getField(offset[i]);
        }


        return make_shared<OptimizerPointerInfo>(optInfo);
    }
    return nullptr;
}







void emitError(string stringhina) {
    LLVM_DEBUG(dbgs() << "[ERROR] " << stringhina << "\n");

}




shared_ptr<OptimizerStructInfo> MetricBase::loadStructInfo(Value *glob, shared_ptr<StructInfo> pInfo, string name) {
    shared_ptr<OptimizerStructInfo> optInfo = make_shared<OptimizerStructInfo>(pInfo->size());

    string function = "";
    if (auto instr = dyn_cast_or_null<Instruction>(glob)) {
        function = instr->getFunction()->getName().str();
    }

    int i = 0;
    for (auto it = pInfo->begin(); it != pInfo->end(); it++) {
        if (auto structInfo = dynamic_ptr_cast_or_null<StructInfo>(*it)) {
            optInfo->setField(i, loadStructInfo(glob, structInfo, (name + "_" + to_string(i))));
        } else if (auto ii = dyn_cast<InputInfo>(it->get())) {
            auto fptype = dynamic_ptr_cast_or_null<FPType>(ii->IType);
            if (!fptype) {
                LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n");

            } else {
                auto info = allocateNewVariableForValue(glob, fptype, ii->IRange, ii->IError, function, false,
                                                        name + "_" + to_string(i));
                optInfo->setField(i, info);
            }
        } else {

        }
        i++;
    }

    return optInfo;
}



void MetricBase::handleGEPInstr(llvm::Instruction *gep, shared_ptr<ValueInfo> valueInfo) {
    const llvm::GetElementPtrInst *gep_i = dyn_cast<llvm::GetElementPtrInst>(gep);
    LLVM_DEBUG(dbgs() << "Handling GEP. \n";);

    Value *operand = gep_i->getOperand(0);
    LLVM_DEBUG(dbgs() << "Operand: ";);
    LLVM_DEBUG(operand->print(dbgs()););
    LLVM_DEBUG(dbgs() << "\n";);

    std::vector<unsigned> offset;

    if (extractGEPOffset(gep_i->getSourceElementType(),
                         iterator_range<User::const_op_iterator>(gep_i->idx_begin(),
                                                                 gep_i->idx_end()),
                         offset)) {
        LLVM_DEBUG(dbgs() << "Exctracted offset: [";);
        for (unsigned int i = 0; i < offset.size(); i++) {
            LLVM_DEBUG(dbgs() << offset[i] << ", ";);
        }
        LLVM_DEBUG(dbgs() << "]\n";);
        //When we load an address from a "thing" we need to store a reference to it in order to successfully update the error
        auto optInfo_t = tuner::dynamic_ptr_cast_or_null<tuner::OptimizerPointerInfo>(getInfoOfValue(operand));
        if (!optInfo_t) {
            LLVM_DEBUG(dbgs() << "Probably trying to access a non float element, bailing out.\n";);
            return;
        }


        auto optInfo = optInfo_t->getOptInfo();
        if (!optInfo) {
            LLVM_DEBUG(dbgs() << "Probably trying to access a non float element, bailing out.\n";);
            return;
        }


        //This will only contain displacements for struct fields...
        for (unsigned int i = 0; i < offset.size(); i++) {
            auto structInfo = dynamic_ptr_cast_or_null<OptimizerStructInfo>(optInfo);
            if (!structInfo) {
                LLVM_DEBUG(dbgs() << "Probably trying to access a non float element, bailing out.\n";);
                return;
            }

            optInfo = structInfo->getField(offset[i]);
        }


        LLVM_DEBUG(dbgs() << "Infos associated: " << optInfo->toString() << "\n";);
        saveInfoForValue(gep, make_shared<OptimizerPointerInfo>(optInfo));
        return;
    }
    emitError("Cannot extract GEPOffset!");
}

bool MetricBase::extractGEPOffset(const llvm::Type *source_element_type,
                                 const llvm::iterator_range<llvm::User::const_op_iterator> indices,
                                 std::vector<unsigned> &offset) {
    assert(source_element_type != nullptr);
    LLVM_DEBUG((dbgs() << "indices: "););
    for (auto idx_it = indices.begin() + 1; // skip first index
         idx_it != indices.end(); ++idx_it) {

        const llvm::ConstantInt *int_i = dyn_cast<llvm::ConstantInt>(*idx_it);
        if (int_i) {
            int n = static_cast<int>(int_i->getSExtValue());
            if (isa<llvm::ArrayType>(source_element_type) || isa<llvm::VectorType>(source_element_type)) {
                //This is needed to skip the array element in array of structures
                //In facts, we treats arrays as "scalar" things, so we just do not want to deal with them
                source_element_type = source_element_type->getContainedType(n);
                LLVM_DEBUG(dbgs() << "continuing...   ";);
                continue;
            }


            offset.push_back(n);
            /*source_element_type =
                    cast<StructType>(source_element_type)->getTypeAtIndex(n);*/
            LLVM_DEBUG((dbgs() << n << " "););
        } else {
            //We can skip only if is a sequential i.e. we are accessing an index of an array
            if (!isa<llvm::ArrayType>(source_element_type) || isa<llvm::VectorType>(source_element_type)) {
                emitError("Index of GEP not constant");
                return false;
            }
        }
    }
    LLVM_DEBUG((dbgs() << "--end indices\n"););
    return true;
}

void MetricBase::handleFCmp(Instruction *instr, shared_ptr<ValueInfo> valueInfo) {
    assert(instr->getOpcode() == llvm::Instruction::FCmp && "Operand mismatch!");

    auto op1 = instr->getOperand(0);
    auto op2 = instr->getOperand(1);


    auto info1 = getInfoOfValue(op1);
    auto info2 = getInfoOfValue(op2);

    if (!info1 || !info2) {
        LLVM_DEBUG(dbgs() << "One of the two values does not have info, ignoring...\n";);
        return;
    }

    if(auto scalar = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info1)){
        if(scalar->doesReferToConstant()){
            LLVM_DEBUG(dbgs() << "Info1 is a constant, skipping as no further cast cost will be introduced.\n";);
            return;
        }
    }

    if(auto scalar = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info2)){
        if(scalar->doesReferToConstant()){
            LLVM_DEBUG(dbgs() << "Info2 is a constant, skipping as no further cast cost will be introduced.\n";);
            return;;
        }
    }


    shared_ptr<OptimizerScalarInfo> varCast1 = allocateNewVariableWithCastCost(op1, instr);
    shared_ptr<OptimizerScalarInfo> varCast2 = allocateNewVariableWithCastCost(op2, instr);



    //The two variables must only contain the same data type, no floating point value returned.
    opt->insertTypeEqualityConstraint(varCast1, varCast2, true);


}

void MetricBase::openPhiLoop(PHINode *phiNode, Value *value) {
    getPhiWatcher().openPhiLoop(phiNode, value);
}

void MetricBase::openMemLoop(LoadInst *load, Value *value) {
    getMemWatcher().openPhiLoop(load, value);
}





shared_ptr<OptimizerScalarInfo>
MetricBase::handleBinOpCommon(Instruction *instr, Value *op1, Value *op2, bool forceFixEquality,
                             shared_ptr<ValueInfo> valueInfos) {
    auto info1 = getInfoOfValue(op1);
    auto info2 = getInfoOfValue(op2);

    if (!info1 || !info2) {
        LLVM_DEBUG(dbgs() << "One of the two values does not have info, ignoring...\n";);
        return nullptr;
    }

    auto inputInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfos->metadata);
    if (!inputInfo) {
        LLVM_DEBUG(dbgs() << "No info on destination, bailing out, bug in VRA?\n";);
        return nullptr;
    }

    auto fptype = dynamic_ptr_cast_or_null<FPType>(inputInfo->IType);
    if (!fptype) {
        LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n";);
        return nullptr;
    }


    shared_ptr<OptimizerScalarInfo> varCast1 = allocateNewVariableWithCastCost(op1, instr);
    shared_ptr<OptimizerScalarInfo> varCast2 = allocateNewVariableWithCastCost(op2, instr);



    //Obviously the type should be sufficient to contain the result
    shared_ptr<OptimizerScalarInfo> result = allocateNewVariableForValue(instr, fptype, inputInfo->IRange,
                                                                         inputInfo->IError,
                                                                         instr->getFunction()->getName().str());

    opt->insertTypeEqualityConstraint(varCast1, varCast2, forceFixEquality);
    opt->insertTypeEqualityConstraint(varCast1, result, forceFixEquality);

    return result;
}




void MetricBase::handleCall(Instruction *instruction, shared_ptr<ValueInfo> valueInfo) {
    const auto *call_i = dyn_cast<CallBase>(instruction);
    if (!call_i) {
        llvm_unreachable("Cannot cast a call instruction to llvm::CallBase");
        return;
    }

    //FIXME: each time a call is not handled, a forced casting to the older type is needed.
    //Therefore this should be added as a cost, not simply ignored

    // fetch function name
    llvm::Function *callee = call_i->getCalledFunction();
    if (callee == nullptr) {
        emitError("Indirect calls not supported!");
        return;
    }


    const std::string calledFunctionName = callee->getName().str();
    LLVM_DEBUG(dbgs() << ("We are calling " + calledFunctionName + "\n"););


    auto function = getKnown_functions().find(calledFunctionName);
    if (function == getKnown_functions().end()) {
        const auto intrinsicsID = callee->getIntrinsicID();
        if (intrinsicsID != llvm::Intrinsic::not_intrinsic) {
            switch (intrinsicsID) {
                //TODO: implement the rest of the libc...
                /*case llvm::Intrinsic::log2:
                case llvm::Intrinsic::sqrt:
                case llvm::Intrinsic::powi:
                case llvm::Intrinsic::sin:
                case llvm::Intrinsic::cos:
                case llvm::Intrinsic::pow:
                case llvm::Intrinsic::exp:
                case llvm::Intrinsic::exp2:
                case llvm::Intrinsic::log:
                case llvm::Intrinsic::log10:
                case llvm::Intrinsic::fma:
                case llvm::Intrinsic::fabs:
                case llvm::Intrinsic::floor:
                case llvm::Intrinsic::ceil:
                case llvm::Intrinsic::trunc:
                case llvm::Intrinsic::rint:
                case llvm::Intrinsic::nearbyint:
                case llvm::Intrinsic::round:*/
                //FIXME: we, for now, emulate the support for these intrinsic; in the real case, these calls have
                // their counterpart in all the versions, so they can be treated in some other ways

                break;
                default:
                    emitError("skipping intrinsic " + calledFunctionName);
                    return;
            }
        }
        LLVM_DEBUG(dbgs() << "Handling external function call, we will convert all to original parameters.\n";);
        handleUnknownFunction(instruction, valueInfo);
        return;
    }

    // fetch ranges of arguments
    std::list<shared_ptr<OptimizerInfo>> arg_errors;
    std::list<shared_ptr<OptimizerScalarInfo>> arg_scalar_errors;
    LLVM_DEBUG(dbgs() << ("Arguments:\n"););
    for (auto arg_it = call_i->arg_begin(); arg_it != call_i->arg_end(); ++arg_it) {
        LLVM_DEBUG(dbgs() << "info for ";);
        LLVM_DEBUG((*arg_it)->print(dbgs()););
        LLVM_DEBUG(dbgs() << " --> ";);

        //if a variable was declared for type
        auto info = getInfoOfValue(*arg_it);
        if (!info) {
            //This is needed to resolve eventual constants in function call (I'm looking at you, LLVM)
            LLVM_DEBUG(dbgs() << "No error for the argument!\n";);
        } else {
            LLVM_DEBUG(dbgs() << "Got this error: " << info->toString() << "\n";);
        }

        //Even if is a null value, we push it!
        arg_errors.push_back(info);

        /*if (const generic_range_ptr_t arg_info = fetchInfo(*arg_it)) {*/
        //If the error is a scalar, collect it also as a scalar
        auto arg_info_scalar = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info);
        if (arg_info_scalar) {
            arg_scalar_errors.push_back(arg_info_scalar);
        }
        //}
        LLVM_DEBUG(dbgs() << "\n\n";);
    }
    LLVM_DEBUG(dbgs() << ("Arguments end.\n"););




    //Allocating variable for result: all returns will have the same type, and therefore a cast, if needed
    shared_ptr<OptimizerInfo> retInfo;
    if (auto inputInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata)) {
        if(instruction->getType()->isFloatingPointTy()) {
            auto fptype = dynamic_ptr_cast_or_null<FPType>(inputInfo->IType);
            if (fptype) {
                LLVM_DEBUG(dbgs() << fptype->toString();
                dbgs() << "\n";
                dbgs() << "Info: " << inputInfo->toString() << "\n";);
                shared_ptr<OptimizerScalarInfo> result = allocateNewVariableForValue(instruction, fptype,
                                                                                     inputInfo->IRange,
                                                                                     inputInfo->IError,
                                                                                     instruction->getFunction()->getNameOrAsOperand());
                retInfo = result;
                LLVM_DEBUG(dbgs() << "Allocated variable for returns.\n";);
            } else {
                LLVM_DEBUG(dbgs() << "There was an input info but no fix point associated.\n";);
            }
        }else{
            LLVM_DEBUG(dbgs() << "Has metadata but is not a floating point!!!\n";);
        }
    } else if (auto pInfo = dynamic_ptr_cast_or_null<StructInfo>(valueInfo->metadata)) {
        auto info = loadStructInfo(instruction, pInfo, "");
        saveInfoForValue(instruction, info);
        retInfo = info;
        LLVM_DEBUG(dbgs() << "Allocated variable for struct returns (?).\n";);
    } else {
        LLVM_DEBUG(dbgs() << "No info available on return value, maybe it is not a floating point returning function.\n";);
    }



    //in retInfo we now have a variable for the return value of the function. Every return should be casted against it!


    auto it = getFunctions_still_to_visit().find(calledFunctionName);
    if (it != getFunctions_still_to_visit().end()) {
        //We mark the called function as visited from the global queue, so we will not visit it starting from root.
        getFunctions_still_to_visit().erase(calledFunctionName);
        LLVM_DEBUG(dbgs() << "Function " << calledFunctionName << " marked as visited in global queue.\n";);
    } else {
        LLVM_DEBUG(dbgs() << "\n\n==================================================\n";
        dbgs() << "FUNCTION ALREADY VISITED!\n";
        dbgs() << "As we have already visited the function we can not visit it again, as it will cause errors.\n";
        dbgs() << "Probably, some precedent component of TAFFO did not clone this function, therefore this error.\n";
        dbgs() << "==================================================\n\n";);
        //Ok it may happen to visit the same function two times. In this case, just reuse the variable. If the function was cloneable, TAFFO would have already done it!
        return;
    }

    //Obviously the type should be sufficient to contain the result
    /* THIS IS A WHITELIST! USE FOR DEBUG
     * auto nm = callee->getName();
    if (!nm.equals("main") &&
        !nm.equals("_Z9bs_threadPv") &&
        !nm.equals("_Z19BlkSchlsEqEuroNoDivdddddifPdS_.3") &&
            !nm.equals("_Z19BlkSchlsEqEuroNoDivfffffifPfS_.5")&&
            !nm.equals("_Z4CNDFf.2.13")&&
            !nm.equals("CNDF.1")) {
        dbgs() << "HALTING CALLING DUE TO DEBUG REQUEST!";
        return;
    }*/

    //In this case we have no known math function.
    //We will have, when enabled, math functions. In this case these will be handled here!

    // if not a whitelisted then try to fetch it from Module
    // fetch llvm::Function
    if (function != getKnown_functions().end()) {
        LLVM_DEBUG(dbgs() << ("The function belongs to the current module.\n"););
        // got the llvm::Function
        llvm::Function *f = function->second;

        // check for recursion
        size_t call_count = 0;
        for (size_t i = 0; i < getCall_stack().size(); i++) {
            if (getCall_stack()[i] == f) {
                call_count++;
            }
        }

        //WE DO NOT SUPPORT RECURSION!
        if (call_count <= 1) {
            // Can process
            // update parameter metadata
            LLVM_DEBUG(dbgs() << ("Processing function...\n"););
            opt->processFunction(*f, arg_errors, retInfo);
            LLVM_DEBUG(dbgs() << "Finished processing call " << calledFunctionName << " : ";);
        } else {
            emitError("Recursion NOT supported!");
            return;
        }

    } else {
        //FIXME: this branch is totally avoided as it is handled before, make the correct corrections
        //the function was not found in current module: it is undeclared
        const auto intrinsicsID = callee->getIntrinsicID();
        if (intrinsicsID == llvm::Intrinsic::not_intrinsic) {
            // TODO: handle case of external function call: element must be in original type form and result is forced to be a double
        } else {
            switch (intrinsicsID) {
                case llvm::Intrinsic::memcpy:
                    //handleMemCpyIntrinsics(call_i);
                    llvm_unreachable("Memcpy not handled atm!");
                    break;
                default:
                    emitError("skipping intrinsic " + calledFunctionName);
            }
            // TODO handle case of llvm intrinsics function call
        }
    }


    return;

}


void MetricBase::handleReturn(Instruction *instr, shared_ptr<ValueInfo> valueInfo) {
    const auto *ret_i = dyn_cast<llvm::ReturnInst>(instr);
    if (!ret_i) {
        llvm_unreachable("Could not convert Return Instruction to ReturnInstr");
        return;
    }

    llvm::Value *ret_val = ret_i->getReturnValue();

    if (!ret_val) {
        LLVM_DEBUG(dbgs() << "Handling return void, doing nothing.\n";);
        return;
    }


    //When returning, we must return the same data type used.
    //Therefore we should eventually take into account the conversion cost.
    auto regInfo = getInfoOfValue(ret_val);
    if (!regInfo) {
        LLVM_DEBUG(dbgs() << "No info on returned value, maybe a non float return, forgetting about it.\n";);
        return;
    }


    auto info = getRetStack().top();
    if (!info) {
        emitError("We wanted to save a result, but on the stack there is not an info available. This maybe an error!");
        return;
    }

    auto infoLinear = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info);
    if (!infoLinear) {
        llvm_unreachable("Structure return still not handled!");
    }

    auto allocated = allocateNewVariableWithCastCost(ret_val, instr);
    opt->insertTypeEqualityConstraint(infoLinear, allocated, true);


}

void MetricBase::saveInfoForPointer(Value *value, shared_ptr<OptimizerPointerInfo> pointerInfo) {
    assert(value && "Value cannot be null!");
    assert(pointerInfo && "Pointer info cannot be nullptr!");

    auto info = getInfoOfValue(value);
    if (!info) {
        LLVM_DEBUG(dbgs() << "Storing new info for the value!\n";);
        saveInfoForValue(value, pointerInfo);
        return;
    }

    LLVM_DEBUG(dbgs() << "Updating info of pointer...\n";);

    //PointerInfo() -> PointerInfo() -> Value[s]
    auto info_old = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(info);
    assert(info_old && info_old->getOptInfo()->getKind() == OptimizerInfo::K_Pointer);
    info_old = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(info_old->getOptInfo());
    assert(info_old);
    //We here should have info about the pointed element
    auto info_old_pointee = info_old->getOptInfo();
    if (info_old_pointee->getKind() == OptimizerInfo::K_Pointer) {
        LLVM_DEBUG(dbgs() << "[WARNING] not handling pointer to pointer update!\n";);
        return;
    }

    //Same code but for new data
    auto info_new = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(pointerInfo);
    assert(info_new && info_new->getOptInfo()->getKind() == OptimizerInfo::K_Pointer);
    info_new = dynamic_ptr_cast_or_null<OptimizerPointerInfo>(info_new->getOptInfo());
    assert(info_new);
    //We here should have info about the pointed element
    auto info_new_pointee = info_new->getOptInfo();
    if (info_new_pointee->getKind() == OptimizerInfo::K_Pointer) {
        LLVM_DEBUG(dbgs() << "[WARNING] not handling pointer to pointer update (and also unpredicted state!)!\n";);
        return;
    }

    if (info_old_pointee->getKind() != info_new_pointee->getKind()) {
        LLVM_DEBUG(dbgs()
                << "[WARNING] This pointer will in a point refer to two different variable that may have different data types.\n"
                   "The results may be unpredictable, you have been warned!\n";);
    }

    if (!info_old_pointee->operator==(*info_new_pointee)) {
        LLVM_DEBUG(dbgs()
                << "[WARNING] This pointer will in a point refer to two different variable that may have different data types.\n"
                   "The results may be unpredictable, you have been warned!\n";);
    }


    getValueToVariableName()[value] = pointerInfo;


}


void MetricBase::handleUnknownFunction(Instruction *instruction, shared_ptr<ValueInfo> valueInfo) {

    assert(instruction && "Instruction is nullptr");
    const auto *call_i = dyn_cast<CallBase>(instruction);
    LLVM_DEBUG(dbgs() << "=====> Unknown function handling: " << call_i->getCalledFunction()->getName() << "\n";);


    assert(call_i && "Cannot cast instruction to call!");
    shared_ptr<OptimizerScalarInfo> retInfo;
    //handling return value. We will force it to be in the original type.
    if (auto inputInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata)) {
        if (inputInfo->IRange && instruction->getType()->isFloatingPointTy()) {
            auto fptype = dynamic_ptr_cast_or_null<FPType>(inputInfo->IType);
            if (fptype) {
                LLVM_DEBUG(dbgs() << fptype->toString(););
                shared_ptr<OptimizerScalarInfo> result = allocateNewVariableForValue(instruction, fptype,
                                                                                     inputInfo->IRange,
                                                                                     inputInfo->IError,
                                                                                     call_i->getFunction()->getName().str(),
                                                                                     true,
                                                                                     "",
                                                                                     true,
                                                                                     false);
                retInfo = result;
                LLVM_DEBUG(dbgs() << "Correctly handled. New info: " << retInfo->toString() << "\n";);
            } else {
                LLVM_DEBUG(dbgs() << "There was an input info but no fix point associated.\n";);
            }
        } else {
            LLVM_DEBUG(dbgs() << "The call does not return a floating point value.\n";);
        }
    } else if (auto pInfo = dynamic_ptr_cast_or_null<StructInfo>(valueInfo->metadata)) {
        emitError("The function considered returns a struct [?]\n");
        return;
    } else {
        LLVM_DEBUG(dbgs() << "No info available on return value, maybe it is not a floating point returning function.\n";);
    }

    //If we have info on return value, forcing the return value in the model to be of the returned type of function
    if (retInfo) {
        if (instruction->getType()->isDoubleTy()) {
            auto constraint = vector<pair<string, double>>();
            constraint.clear();
            constraint.push_back(make_pair(retInfo->getDoubleSelectedVariable(), 1.0));
            getModel().insertLinearConstraint(constraint, Model::EQ, 1/*, "Type constraint for return value"*/);
            LLVM_DEBUG(dbgs() << "Forced return cast to double.\n";);
        } else if (instruction->getType()->isFloatTy()) {
            auto constraint = vector<pair<string, double>>();
            constraint.clear();
            constraint.push_back(make_pair(retInfo->getFloatSelectedVariable(), 1.0));
            getModel().insertLinearConstraint(constraint, Model::EQ, 1/*, "Type constraint for return value"*/);
            LLVM_DEBUG(dbgs() << "Forced return cast to float.\n";);
        } else if (instruction->getType()->isFloatingPointTy()) {
            LLVM_DEBUG(dbgs() << "The function returns a floating point type not implemented in the model. Bailing out.\n";);
        } else {
            LLVM_DEBUG(dbgs() << "Probably the functions returns a pointer but i do not known what to do!\n";);
        }
    }



    //Return value handled, now it's time for parameters
    LLVM_DEBUG(dbgs() << ("Arguments:\n"););
    int arg = 0;
    for (auto arg_it = call_i->arg_begin(); arg_it != call_i->arg_end(); ++arg_it, arg++) {
        LLVM_DEBUG(dbgs() << "[" << arg << "] info for ";);
        LLVM_DEBUG((*arg_it)->print(dbgs()););
        LLVM_DEBUG(dbgs() << " --> ";);

        //if a variable was declared for type
        auto info = getInfoOfValue(*arg_it);
        if (!info) {
            //This is needed to resolve eventual constants in function call (I'm looking at you, LLVM)
            LLVM_DEBUG(dbgs() << "No info for the argument!\n";);
            continue;
        } else {
            LLVM_DEBUG(dbgs() << "Got this info: " << info->toString() << "\n";);
        }

        /*if (const generic_range_ptr_t arg_info = fetchInfo(*arg_it)) {*/
        //If the error is a scalar, collect it also as a scalar
        auto arg_info_scalar = dynamic_ptr_cast_or_null<OptimizerScalarInfo>(info);
        if (arg_info_scalar) {
            //Ok, we have info and it is a scalar, let's hope that it's not a pointer
            if ((*arg_it)->getType()->isFloatTy()) {
                auto info2 = allocateNewVariableWithCastCost(arg_it->get(), instruction);
                auto constraint = vector<pair<string, double>>();
                constraint.clear();
                constraint.push_back(make_pair(info2->getFloatSelectedVariable(), 1.0));
                getModel().insertLinearConstraint(constraint, Model::EQ, 1/*, "Type constraint for argument value"*/);
                LLVM_DEBUG(dbgs() << "Forcing argument to float type.\n";);
            } else if ((*arg_it)->getType()->isDoubleTy()) {
                auto info2 = allocateNewVariableWithCastCost(arg_it->get(), instruction);
                auto constraint = vector<pair<string, double>>();
                constraint.clear();
                constraint.push_back(make_pair(info2->getDoubleSelectedVariable(), 1.0));
                getModel().insertLinearConstraint(constraint, Model::EQ, 1/*, "Type constraint for argument value"*/);
                LLVM_DEBUG(dbgs() << "Forcing argument to double type.\n";);
            } else if ((*arg_it)->getType()->isFloatingPointTy()) {
                LLVM_DEBUG(dbgs() << "The function uses a floating point type not implemented in the model. Bailing out.\n";);
            } else {
                LLVM_DEBUG(dbgs() << "Probably the functions uses a pointer but I do not known what to do!\n";);
            }

        } else {
            LLVM_DEBUG(dbgs() << "This is a struct passed to an external function but has been optimized by TAFFO. Is this even possible???\n";);
        }

        LLVM_DEBUG(dbgs() << "\n\n";);
    }
    LLVM_DEBUG(dbgs() << ("Arguments end.\n"););

    LLVM_DEBUG(dbgs() << "Function should be correctly handled now.\n";);


}


void MetricBase::handleAlloca(Instruction *instruction, shared_ptr<ValueInfo> valueInfo) {
    if (!valueInfo) {
        LLVM_DEBUG(dbgs() << "No value info, skipping...\n";);
        return;
    }

    if (!valueInfo->metadata) {
        LLVM_DEBUG(dbgs() << "No value metadata, skipping...\n";);
        return;
    }

    auto *alloca = dyn_cast<AllocaInst>(instruction);


    if (!alloca->getAllocatedType()->isPointerTy()) {
        if (valueInfo->metadata->getKind() == MDInfo::K_Field) {
            LLVM_DEBUG(dbgs() << " ^ This is a real field\n";);
            auto fieldInfo = dynamic_ptr_cast_or_null<InputInfo>(valueInfo->metadata);
            if (!fieldInfo) {
                LLVM_DEBUG(dbgs() << "Not enough information. Bailing out.\n\n";);
                return;
            }

            auto fptype = dynamic_ptr_cast_or_null<FPType>(fieldInfo->IType);
            if (!fptype) {
                LLVM_DEBUG(dbgs() << "No fixed point info associated. Bailing out.\n";);
                return;
            }
            auto info = allocateNewVariableForValue(alloca, fptype, fieldInfo->IRange, fieldInfo->IError,
                                                    alloca->getFunction()->getName().str(),
                                                    false);
            saveInfoForValue(alloca, make_shared<OptimizerPointerInfo>(info));
        } else if (valueInfo->metadata->getKind() == MDInfo::K_Struct) {
            LLVM_DEBUG(dbgs() << " ^ This is a real structure\n";);

            auto fieldInfo = dynamic_ptr_cast_or_null<StructInfo>(valueInfo->metadata);
            if (!fieldInfo) {
                LLVM_DEBUG(dbgs() << "No struct info. Bailing out.\n";);
                return;
            }

            auto optInfo = loadStructInfo(alloca, fieldInfo, "");
            saveInfoForValue(alloca, optInfo);

        } else {
            llvm_unreachable("Unknown metadata!");
        }


    } else {
        LLVM_DEBUG(dbgs() << " ^ this is a pointer, skipping as it is unsupported at the moment.\n";);
        return;
    }


}

