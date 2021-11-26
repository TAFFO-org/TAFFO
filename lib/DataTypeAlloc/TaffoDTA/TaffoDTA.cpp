#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "Optimizer.h"
#include "llvm/IR/IRBuilder.h"
#include "MetricBase.h"
#include "TaffoDTA.h"
#include "Metadata.h"
#include "DTAConfig.h"


using namespace llvm;
using namespace tuner;
using namespace mdutils;
using namespace taffo;


char TaffoTuner::ID = 0;

static RegisterPass<TaffoTuner> X(
        "taffodta",
        "TAFFO Framework Data Type Allocation",
        false /* does not affect the CFG */,
        true /* Optimization Pass */);

void TaffoTuner::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredTransitive<LoopInfoWrapperPass>();
    AU.addRequiredTransitive<MemorySSAWrapperPass>();
    AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
    AU.addRequiredTransitive<TargetTransformInfoWrapperPass>();
    AU.setPreservesAll();
}


bool TaffoTuner::runOnModule(Module &m) {
    std::vector<llvm::Value *> vals;
    llvm::SmallPtrSet<llvm::Value *, 8U> valset;
    retrieveAllMetadata(m, vals, valset);

    LLVM_DEBUG(llvm::dbgs() << "Model " << CostModelFilename << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Inst " << InstructionSet << "\n");


    if (MixedMode) {
        buildModelAndOptimze(m, vals, valset);
    } else {
        mergeFixFormat(vals, valset);
    }

    std::vector<Function *> toDel;
    toDel = collapseFunction(m);

    attachFPMetaData(vals);
    attachFunctionMetaData(m);

    for (Function *f : toDel)
        f->eraseFromParent();

    return true;
}


void TaffoTuner::retrieveAllMetadata(Module &m, std::vector<llvm::Value *> &vals,
                                     llvm::SmallPtrSetImpl<llvm::Value *> &valset) {
    mdutils::MetadataManager &MDManager = mdutils::MetadataManager::getMetadataManager();

    for (GlobalObject &globObj : m.globals()) {
        MDInfo *MDI = MDManager.retrieveMDInfo(&globObj);
        if (processMetadataOfValue(&globObj, MDI))
            vals.push_back(&globObj);
    }

    for (Function &f : m.functions()) {
        if (f.isIntrinsic())
            continue;

        SmallVector<mdutils::MDInfo *, 5> argsII;
        MDManager.retrieveArgumentInputInfo(f, argsII);
        auto arg = f.arg_begin();
        for (auto itII = argsII.begin(); itII != argsII.end(); itII++) {
            if (processMetadataOfValue(arg, *itII))
                vals.push_back(arg);
            arg++;
        }

        for (inst_iterator iIt = inst_begin(&f), iItEnd = inst_end(&f); iIt != iItEnd; iIt++) {
            MDInfo *MDI = MDManager.retrieveMDInfo(&(*iIt));
            if (processMetadataOfValue(&(*iIt), MDI))
                vals.push_back(&*iIt);
        }
    }

    sortQueue(vals, valset);
}


bool TaffoTuner::processMetadataOfValue(Value *v, MDInfo *MDI) {
    LLVM_DEBUG(dbgs() << __FUNCTION__ << " v=" << *v << " MDI=" << (MDI ? MDI->toString() : std::string("(null)"))
                      << "\n");
    if (!MDI)
        return false;
    std::shared_ptr<MDInfo> newmdi(MDI->clone());

    if (v->getType()->isVoidTy()) {
        LLVM_DEBUG(dbgs() << "[Info] Value " << *v << " has void type, leaving metadata unchanged\n");
        valueInfo(v)->metadata = newmdi;
        return true;
    }

    /* HACK to set the enabled status on phis which compensates for a bug in vra.
     * Affects axbench/sobel. */
    bool forceEnableConv = false;
    if (isa<PHINode>(v) && !conversionDisabled(v) && isa<InputInfo>(newmdi.get())) {
        forceEnableConv = true;
    }

    bool skippedAll = true;
    Type *fuwt = fullyUnwrapPointerOrArrayType(v->getType());
    llvm::SmallVector<std::pair<MDInfo *, Type *>, 8> queue({std::make_pair(newmdi.get(), fuwt)});

    while (queue.size() > 0) {
        std::pair<MDInfo *, Type *> elem = queue.pop_back_val();

        if (InputInfo *II = dyn_cast<InputInfo>(elem.first)) {
            if (forceEnableConv)
                II->IEnableConversion = true;

            //FIXME: hack to propagate itofp metadata
            if(isa<UIToFPInst>(v) ||
               isa<SIToFPInst>(v)){
                LLVM_DEBUG(dbgs() << "FORCING CONVERSION OF A ITOFP!\n";);
                II->IEnableConversion = true;
            }

            if (!isFloatType(elem.second)) {
                LLVM_DEBUG(dbgs() << "[Info] Skipping a member of " << *v << " because not a float\n");
                continue;
            }

            //TODO: insert logic here to associate different types in a clever way
            if (associateFixFormat(*II, elem.second->getTypeID())) {
                skippedAll = false;
            }

        } else if (StructInfo *SI = dyn_cast<StructInfo>(elem.first)) {
            if (!elem.second->isStructTy()) {
                LLVM_DEBUG(dbgs() << "[ERROR] found non conforming structinfo " << SI->toString() << " on value " << *v
                                  << "\n");
                LLVM_DEBUG(dbgs() << "contained type " << *elem.second << " is not a struct type\n");
                LLVM_DEBUG(dbgs() << "The top-level MDInfo was " << MDI->toString() << "\n");
                llvm_unreachable("Non-conforming StructInfo.");
            }
            int i = 0;
            for (std::shared_ptr<MDInfo> se: *SI) {
                if (se.get() != nullptr) {
                    Type *thisT = fullyUnwrapPointerOrArrayType(elem.second->getContainedType(i));
                    queue.push_back(std::make_pair(se.get(), thisT));
                }
                i++;
            }

        } else {
            llvm_unreachable("unknown mdinfo subclass");
        }
    }

    if (!skippedAll) {
        std::shared_ptr<ValueInfo> vi = valueInfo(v);
        vi->metadata = newmdi;
        LLVM_DEBUG(dbgs() << "associated metadata '" << newmdi->toString() << "' to value " << *v);
        if (Instruction *i = dyn_cast<Instruction>(v))
            LLVM_DEBUG(dbgs() << " (parent function = " << i->getFunction()->getName() << ")");
        LLVM_DEBUG(dbgs() << "\n");
        if (InputInfo *ii = dyn_cast<InputInfo>(newmdi.get()))
            vi->initialType = ii->IType;
    }
    return !skippedAll;
}


bool TaffoTuner::associateFixFormat(InputInfo &II, Type::TypeID origType) {
    if (!II.IEnableConversion) {
        LLVM_DEBUG(dbgs() << "[Info] Skipping " << II.toString() << ", conversion disabled\n");
        return false;
    }

    if (II.IType.get() != nullptr) {
        LLVM_DEBUG(dbgs() << "[Info] Type of " << II.toString() << " already assigned\n");
        return true;
    }

    Range *rng = II.IRange.get();
    if (rng == nullptr) {
        LLVM_DEBUG(dbgs() << "[Info] Skipping " << II.toString() << ", no range\n");
        return false;
    }


    //New super performing algorithm to compute a lot of things
    //Preserving this code just in case, shold be not necessary
    /*if (ForceFloat >= 0) {
        auto standard = static_cast<mdutils::FloatType::FloatStandard>(ForceFloat.getValue());

        double greatest = abs(II.IRange->Min);
        double max = abs(II.IRange->Max);
        if (max > greatest) greatest = max;

        FloatType res = FloatType(standard, greatest);

        LLVM_DEBUG(dbgs() << "[Info] Forcing conversion to float " << res.toString() << "\n");

        II.IType.reset(res.clone());
        return true;


    } else {*/
    FixedPointTypeGenError fpgerr;
    FPType res = fixedPointTypeFromRange(*rng, &fpgerr, TotalBits, FracThreshold, 64, TotalBits);
    if (fpgerr == FixedPointTypeGenError::InvalidRange) {
        LLVM_DEBUG(dbgs() << "[Info] Skipping " << II.toString() << ", FixedPointTypeGenError::InvalidRange\n");
        return false;
    }
    II.IType.reset(res.clone());
    return true;
    //}


}


void TaffoTuner::sortQueue(std::vector<llvm::Value *> &vals,
                           llvm::SmallPtrSetImpl<llvm::Value *> &valset) {
    // Topological sort by means of a reversed DFS.
    enum VState {
        Visited, Visiting
    };
    DenseMap<Value *, VState> vstates;
    std::vector<Value *> revQueue;
    std::vector<Value *> stack;
    revQueue.reserve(vals.size());
    stack.reserve(vals.size());

    for (Value *v : vals) {
        if (vstates.count(v))
            continue;

        stack.push_back(v);
        while (!stack.empty()) {
            Value *c = stack.back();
            auto cstate = vstates.find(c);
            if (cstate == vstates.end()) {
                vstates[c] = Visiting;
                for (Value *u : c->users()) {
                    if (!isa<Instruction>(u) && !isa<GlobalObject>(u))
                        continue;

                    if (conversionDisabled(u)) {
                        LLVM_DEBUG(dbgs() << "[WARNING] Skipping " << *u << " without TAFFO info!\n");
                        continue;
                    }

                    stack.push_back(u);
                    if (!hasInfo(u)) {
                        LLVM_DEBUG(dbgs() << "[WARNING] Found Value " << *u << " without range! (uses " << *c << ")\n");
                        Type *utype = fullyUnwrapPointerOrArrayType(u->getType());
                        Type *ctype = fullyUnwrapPointerOrArrayType(c->getType());
                        if (!utype->isStructTy() && !ctype->isStructTy()) {
                            InputInfo *ii = cast<InputInfo>(valueInfo(c)->metadata->clone());
                            ii->IRange.reset();
                            std::shared_ptr<ValueInfo> viu = valueInfo(u);
                            viu->metadata.reset(ii);
                            viu->initialType = ii->IType;
                        } else if (utype->isStructTy() && ctype->isStructTy()
                                   && ctype->canLosslesslyBitCastTo(utype)) {
                            valueInfo(u)->metadata.reset(valueInfo(c)->metadata->clone());
                        } else {
                            if (utype->isStructTy())
                                valueInfo(u)->metadata = StructInfo::constructFromLLVMType(utype);
                            else
                                valueInfo(u)->metadata.reset(new InputInfo());
                            LLVM_DEBUG(dbgs() << "not copying metadata of " << *c << " to " << *u
                                              << " because one value has struct typing and the other has not.\n");
                        }
                    }
                }
            } else if (cstate->second == Visiting) {
                revQueue.push_back(c);
                stack.pop_back();
                vstates[c] = Visited;
            } else {
                assert(cstate->second == Visited);
                stack.pop_back();
            }
        }
    }

    vals.clear();
    valset.clear();
    for (auto i = revQueue.rbegin(); i != revQueue.rend(); ++i) {
        vals.push_back(*i);
        valset.insert(*i);
    }
}

void TaffoTuner::mergeFixFormat(const std::vector<llvm::Value *> &vals,
                                const llvm::SmallPtrSetImpl<llvm::Value *> &valset) {
    if (DisableTypeMerging)
        return;

    assert(vals.size() == valset.size() && "They must contain the same elements.");
    bool merged = false;
    for (Value *v : vals) {
        for (Value *u: v->users()) {
            if (valset.count(u)) {
                if (IterativeMerging ? mergeFixFormatIterative(v, u) : mergeFixFormat(v, u)) {
                    restoreTypesAcrossFunctionCall(v);
                    restoreTypesAcrossFunctionCall(u);

                    merged = true;
                }
            }
        }
    }
    if (IterativeMerging && merged)
        mergeFixFormat(vals, valset);
}

bool TaffoTuner::mergeFixFormat(llvm::Value *v, llvm::Value *u) {
    std::shared_ptr<ValueInfo> viv = valueInfo(v);
    std::shared_ptr<ValueInfo> viu = valueInfo(u);
    InputInfo *iiv = dyn_cast<InputInfo>(viv->metadata.get());
    InputInfo *iiu = dyn_cast<InputInfo>(viu->metadata.get());
    if (!iiv || !iiu) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a struct\n");
        return false;
    }
    if (!iiv->IType || !viv->initialType || !iiu->IType || !viu->initialType) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u
                          << " because at least one does not change to a fixed point type\n");
        return false;
    }
    if (v->getType()->isPointerTy() || u->getType()->isPointerTy()) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a pointer\n");
        return false;
    }
    FPType *fpv = dyn_cast<FPType>(viv->initialType.get());
    FPType *fpu = dyn_cast<FPType>(viu->initialType.get());
    if (!fpv || !fpu) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because one is not a FPType\n");
        return false;
    }
    if (!(*fpv == *fpu)) {
        if (isMergeable(fpv, fpu)) {
            std::shared_ptr<mdutils::FPType> fp = merge(fpv, fpu);
            if (!fp) {
                LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u
                                  << " because resulting type is invalid\n");
                return false;
            }
            LLVM_DEBUG(dbgs() << "Merged fixp : \n"
                              << "\t" << *v << " fix type " << fpv->toString() << "\n"
                              << "\t" << *u << " fix type " << fpu->toString() << "\n"
                              << "Final format " << fp->toString() << "\n";);

            iiv->IType.reset(fp->clone());
            iiu->IType.reset(fp->clone());
            return true;
        } else {
            FixCast++;
        }
    }
    return false;
}

bool TaffoTuner::mergeFixFormatIterative(llvm::Value *v, llvm::Value *u) {
    std::shared_ptr<ValueInfo> viv = valueInfo(v);
    std::shared_ptr<ValueInfo> viu = valueInfo(u);
    InputInfo *iiv = dyn_cast<InputInfo>(viv->metadata.get());
    InputInfo *iiu = dyn_cast<InputInfo>(viu->metadata.get());
    if (!iiv || !iiu) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a struct\n");
        return false;
    }
    if (!iiv->IType.get() || !iiu->IType.get()) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u
                          << " because at least one does not change to a fixed point type\n");
        return false;
    }
    if (v->getType()->isPointerTy() || u->getType()->isPointerTy()) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a pointer\n");
        return false;
    }
    FPType *fpv = cast<FPType>(iiv->IType.get());
    FPType *fpu = cast<FPType>(iiu->IType.get());
    if (!(*fpv == *fpu)) {
        if (isMergeable(fpv, fpu)) {
            std::shared_ptr<mdutils::FPType> fp = merge(fpv, fpu);
            if (!fp) {
                LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because resulting type "
                                  << fp->toString() << " is invalid\n");
                return false;
            }
            LLVM_DEBUG(dbgs() << "Merged fixp : \n"
                              << "\t" << *v << " fix type " << fpv->toString() << "\n"
                              << "\t" << *u << " fix type " << fpu->toString() << "\n"
                              << "Final format " << fp->toString() << "\n";);

            iiv->IType.reset(fp->clone());
            iiu->IType.reset(fp->clone());
            return true;
        } else {
            FixCast++;
        }
    }
    return false;
}

bool TaffoTuner::isMergeable(mdutils::FPType *fpv, mdutils::FPType *fpu) const {
    return fpv->getWidth() == fpu->getWidth()
           && (std::abs((int) fpv->getPointPos() - (int) fpu->getPointPos())
               + (fpv->isSigned() == fpu->isSigned() ? 0 : 1)) <= SimilarBits;
}

std::shared_ptr<mdutils::FPType> TaffoTuner::merge(mdutils::FPType *fpv, mdutils::FPType *fpu) const {
    int sign_v = fpv->isSigned() ? 1 : 0;
    int int_v = fpv->getWidth() - fpv->getPointPos() - sign_v;
    int sign_u = fpu->isSigned() ? 1 : 0;
    int int_u = fpu->getWidth() - fpu->getPointPos() - sign_u;

    int sign_res = std::max(sign_u, sign_v);
    int int_res = std::max(int_u, int_v);
    int size_res = std::max(fpv->getWidth(), fpu->getWidth());
    int frac_res = size_res - int_res - sign_res;
    if (sign_res + int_res + frac_res != size_res || frac_res < 0)
        return nullptr; // Invalid format.
    else
        return std::shared_ptr<FPType>(new FPType(size_res, frac_res, sign_res));
}


void TaffoTuner::restoreTypesAcrossFunctionCall(Value *v) {
    LLVM_DEBUG(dbgs() << "restoreTypesAcrossFunctionCall(" << *v << ")\n");
    if (!hasInfo(v)) {
        LLVM_DEBUG(dbgs() << " --> skipping restoring types because value is not converted\n");
        return;
    }

    std::shared_ptr<MDInfo> finalMd = valueInfo(v)->metadata;

    if (Argument *arg = dyn_cast<Argument>(v)) {
        setTypesOnCallArgumentFromFunctionArgument(arg, finalMd);
        return;
    }

    for (Use &use: v->uses()) {
        User *user = use.getUser();
        AbstractCallSite call(&use);
        if (call.getInstruction() == nullptr)
            continue;

        Function *fun = dyn_cast<Function>(call.getCalledFunction());
        if (fun == nullptr) {
            LLVM_DEBUG(dbgs() << " --> skipping restoring types from call site " << *user
                              << " because function reference cannot be resolved\n");
            continue;
        }
        if (fun->isVarArg()) {
            LLVM_DEBUG(dbgs() << " --> skipping restoring types from call site " << *user
                              << " because function is vararg\n");
            continue;
        }

        assert(fun->arg_size() > use.getOperandNo() && "invalid call to function; operandNo > numOperands");
        Argument *arg = fun->arg_begin() + use.getOperandNo();
        if (hasInfo(arg)) {
            valueInfo(arg)->metadata.reset(finalMd->clone());
            setTypesOnCallArgumentFromFunctionArgument(arg, finalMd);
        }
    }
}


void TaffoTuner::setTypesOnCallArgumentFromFunctionArgument(Argument *arg, std::shared_ptr<MDInfo> finalMd) {
    Function *fun = arg->getParent();
    int n = arg->getArgNo();
    LLVM_DEBUG(dbgs() << " --> setting types to " << finalMd->toString() << " on call arguments from function "
                      << fun->getName() << " argument " << n << "\n");
    for (auto it = fun->user_begin(); it != fun->user_end(); it++) {
        if (isa<CallInst>(*it) || isa<InvokeInst>(*it)) {
            Value *callarg = it->getOperand(n);
            LLVM_DEBUG(dbgs() << " --> target " << *callarg << ", callsite " << **it << "\n");
            if (!hasInfo(callarg)) {
                LLVM_DEBUG(dbgs() << " --> argument doesn't get converted; skipping\n");
            } else {
                valueInfo(callarg)->metadata.reset(finalMd->clone());
            }
        }
    }
}


std::vector<Function *> TaffoTuner::collapseFunction(Module &m) {
    std::vector<Function *> toDel;
    for (Function &f : m.functions()) {
        if (MDNode *mdNode = f.getMetadata(CLONED_FUN_METADATA)) {
            if (std::find(toDel.begin(), toDel.end(), &f) != toDel.end())
                continue;
            DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "Analyzing original function " << f.getName() << "\n";);

            for (auto mdIt = mdNode->op_begin(); mdIt != mdNode->op_end(); mdIt++) {
                DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "\t Clone : " << **mdIt << "\n";);

                ValueAsMetadata *md = dyn_cast<ValueAsMetadata>(*mdIt);
                Function *fClone = dyn_cast<Function>(md->getValue());
                if (fClone->user_begin() == fClone->user_end()) {
                    DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "\t Ignoring " << fClone->getName()
                                                      << " because it's not used anywhere\n");
                } else if (Function *eqFun = findEqFunction(fClone, &f)) {
                    DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "\t Replace function " << fClone->getName()
                                                      << " with " << eqFun->getName() << "\n";);
                    fClone->replaceAllUsesWith(eqFun);
                    toDel.push_back(fClone);
                }

            }
        }
    }
    return toDel;
}


bool compareTypesOfMDInfo(MDInfo &mdi1, MDInfo &mdi2) {
    if (mdi1.getKind() != mdi2.getKind())
        return false;

    if (isa<InputInfo>(&mdi1)) {
        InputInfo &ii1 = cast<InputInfo>(mdi1);
        InputInfo &ii2 = cast<InputInfo>(mdi2);
        if (ii1.IType.get() && ii2.IType.get()) {
            return *ii1.IType == *ii2.IType;
        } else
            return false;

    } else if (isa<StructInfo>(&mdi1)) {
        StructInfo &si1 = cast<StructInfo>(mdi1);
        StructInfo &si2 = cast<StructInfo>(mdi2);
        if (si1.size() == si2.size()) {
            int c = si1.size();
            for (int i = 0; i < c; i++) {
                std::shared_ptr<MDInfo> p1 = si1.getField(i);
                std::shared_ptr<MDInfo> p2 = si1.getField(i);
                if ((p1.get() == nullptr) != (p2.get() == nullptr))
                    return false;
                if (p1.get() != nullptr) {
                    if (!compareTypesOfMDInfo(*p1, *p2))
                        return false;
                }
            }
            return true;

        } else
            return false;

    } else {
        return false;
    }
}


Function *TaffoTuner::findEqFunction(Function *fun, Function *origin) {
    std::vector<std::pair<int, std::shared_ptr<MDInfo>>> fixSign;

    DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "\t\t Search eq function for " << fun->getName()
                                      << " in " << origin->getName() << " pool\n";);

    if (isFloatType(fun->getReturnType()) && hasInfo(*fun->user_begin())) {
        std::shared_ptr<MDInfo> retval = valueInfo(*fun->user_begin())->metadata;
        if (retval) {
            fixSign.push_back(std::pair<int, std::shared_ptr<MDInfo>>(-1, retval)); //ret value in signature
            DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "\t\t Return type : "
                                              << valueInfo(*fun->user_begin())->metadata->toString() << "\n";);
        }
    }

    int i = 0;
    for (Argument &arg: fun->args()) {
        if (hasInfo(&arg) && valueInfo(&arg)->metadata) {
            fixSign.push_back(std::pair<int, std::shared_ptr<MDInfo>>(i, valueInfo(&arg)->metadata));
            DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "\t\t Arg " << i << " type : "
                                              << valueInfo(&arg)->metadata->toString() << "\n";);
        }
        i++;
    }

    for (FunInfo fi : functionPool[origin]) {
        if (fi.fixArgs.size() == fixSign.size()) {
            auto fcheck = fi.fixArgs.begin();
            auto fthis = fixSign.begin();
            for (; fthis != fixSign.end(); fcheck++, fthis++) {
                if (fcheck->first != fthis->first)
                    break;
                if (fcheck->second != fthis->second)
                    if (!compareTypesOfMDInfo(*fcheck->second, *fthis->second))
                        break;
            }
            if (fthis == fixSign.end())
                return fi.newFun;
        }
    }

    FunInfo funInfo;
    funInfo.newFun = fun;
    funInfo.fixArgs = fixSign;
    functionPool[origin].push_back(funInfo);
    DEBUG_WITH_TYPE(DEBUG_FUN, dbgs() << "\t Function " << fun->getName() << " used\n";);
    return nullptr;
}


void TaffoTuner::attachFPMetaData(std::vector<llvm::Value *> &vals) {
    for (Value *v : vals) {
        assert(info[v] && "Every value should have info");
        assert(valueInfo(v)->metadata.get() && "every value should have metadata");

        if (isa<Instruction>(v) || isa<GlobalObject>(v)) {
            mdutils::MetadataManager::setMDInfoMetadata(v, valueInfo(v)->metadata.get());
        } else {
            LLVM_DEBUG(dbgs() << "[WARNING] Cannot attach MetaData to " << *v << " (normal for function args)\n");
        }
    }
}


void TaffoTuner::attachFunctionMetaData(llvm::Module &m) {
    mdutils::MetadataManager &MDManager = mdutils::MetadataManager::getMetadataManager();

    for (Function &f : m.functions()) {
        if (f.isIntrinsic())
            continue;

        SmallVector<mdutils::MDInfo *, 5> argsII;
        MDManager.retrieveArgumentInputInfo(f, argsII);
        auto argsIt = argsII.begin();
        for (Argument &arg : f.args()) {
            if (*argsIt) {
                if (hasInfo(&arg)) {
                    MDInfo *mdi = valueInfo(&arg)->metadata.get();
                    *argsIt = mdi;
                }
            }
            argsIt++;
        }
        MDManager.setArgumentInputInfoMetadata(f, argsII);
    }
}


void TaffoTuner::buildModelAndOptimze(Module &m, const vector<llvm::Value *> &vals,
                                      const SmallPtrSetImpl<llvm::Value *> &valset) {
    assert(vals.size() == valset.size() && "They must contain the same elements.");

    
    Optimizer optimizer(m, this, new MetricPerf(),CostModelFilename, CPUCosts::CostType::Performance);
    //Optimizer optimizer(m, this, new MetricPerf(),"", CPUCosts::CostType::Size);
    optimizer.initialize();

    LLVM_DEBUG(dbgs() << "\n============ GLOBALS ============\n");

    for (GlobalObject &globObj : m.globals()) {
        LLVM_DEBUG(globObj.print(dbgs()););
        LLVM_DEBUG(dbgs() << "     -having-     ");
        if (!hasInfo(&globObj)) {
            LLVM_DEBUG(dbgs() << "No info available, skipping.");
        } else {
            LLVM_DEBUG(dbgs() << valueInfo(&globObj)->metadata->toString() << "\n");

            optimizer.handleGlobal(&globObj, valueInfo(&globObj));
        }
        LLVM_DEBUG(dbgs() << "\n\n";);
    }

    //FIXME: this is an hack to prevent multiple visit of the same function if it will be called somewhere from the program
    for (Function &f : m.functions()) {
        //Skip compiler provided functions
          if (f.isIntrinsic() || f.isDeclaration())
            continue;

        if (!f.isIntrinsic() && !f.empty() && f.getName().equals("main")) {
            LLVM_DEBUG(dbgs() << "========== GLOBAL ENTRY POINT main ==========";);

            optimizer.handleCallFromRoot(&f);
            break;
        }
    }

    //Looking for remaining functions
    for (Function &f : m.functions()) {
        //Skip compiler provided functions
        if (f.isIntrinsic()) {
            LLVM_DEBUG(dbgs() << "Skipping intrinsic function " << f.getName() << "\n";);
            continue;
        }

        //Skip empty functions
        if (f.empty()) {
            LLVM_DEBUG(dbgs() << "Skipping empty function " << f.getName() << "\n";);
            continue;
        }

        optimizer.handleCallFromRoot(&f);


    }

    assert(optimizer.finish() && "Optimizer did not found a solution!");


    for (Value *v : vals) {
        if (!valset.count(v)) {
            LLVM_DEBUG(dbgs() << "Not in the conversion queue! Skipping!\n";);
            continue;
        }
        LLVM_DEBUG(dbgs() << "Assigning to ";);
        LLVM_DEBUG(v->print(dbgs()););


        std::shared_ptr<ValueInfo> viu = valueInfo(v);

        //Read from the model, search for the data type associated with that value and convert it!
        auto fp = optimizer.getAssociatedMetadata(v);
        if (!fp) {
            LLVM_DEBUG(dbgs() << "Invalid datatype returned!\n";);
            continue;
        }

        LLVM_DEBUG(dbgs() << " datatype " << fp->toString(););

        LLVM_DEBUG(dbgs() << "\n";);


        bool result = mergeDataTypes(viu->metadata, fp);
        if (result) {
            //Some datatype has changed, restore in function call
            LLVM_DEBUG(dbgs() << "Restoring call type...\n";);
            restoreTypesAcrossFunctionCall(v);
        }

        /*auto *iiv = dyn_cast<InputInfo>(viu->metadata.get());

        iiv->IType.reset(fp->clone());*/

    }

    optimizer.printStatInfos();


}

bool TaffoTuner::mergeDataTypes(shared_ptr<mdutils::MDInfo> old, shared_ptr<mdutils::MDInfo> model) {
    if (!old || !model) return false;

    if (old->getKind() == mdutils::MDInfo::K_Field) {
        assert(model->getKind() == mdutils::MDInfo::K_Field && "Mismatching metadata infos!!!");

        auto old1 = dynamic_ptr_cast_or_null<InputInfo>(old);
        auto model1 = dynamic_ptr_cast_or_null<InputInfo>(model);




        if(!old1->IType) return false;
        LLVM_DEBUG(dbgs() << "model1: " << model1->IType->toString() << "\n";);
        LLVM_DEBUG(dbgs() << "old1: " << old1->IType->toString() << "\n\n";);
        if (old1->IType->operator==(*model1->IType)) {
            return false;
        }

        old1->IType.reset(model1->IType->clone());
        return true;
    } else if (old->getKind() == mdutils::MDInfo::K_Struct) {
        auto old1 = dynamic_ptr_cast_or_null<StructInfo>(old);
        auto model1 = dynamic_ptr_cast_or_null<StructInfo>(model);

        bool changed = false;
        for (unsigned int i = 0; i < old1->size(); i++) {
            changed |= mergeDataTypes(old1->getField(i), model1->getField(i));
        }
        return changed;
    }

    llvm_unreachable("unknown data type");

}














