#include <set>
#include <fstream>
#include <unordered_map>
#include <stack>
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "InputInfo.h"
#include "Metadata.h"
#include "TypeUtils.h"
#include "Infos.h"
#include "OptimizerInfo.h"
#include "Model.h"
#include "llvm/ADT/Triple.h"
#include "TaffoDTA.h"
#include "CPUCosts.h"
#include "llvm/IR/DerivedTypes.h"
#include "MetricBaseForward.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#ifndef __TAFFO_DTA_OPTIMIZER_H__
#define __TAFFO_DTA_OPTIMIZER_H__

extern bool hasHalf;
extern bool hasQuad;
extern bool hasPPC128;
extern bool hasFP80;
extern bool hasBF16;



//FIXME: I_COST should absolutely not be constant

#define I_COST 1

//This means how much the casting cost will be relevant for the computation
extern llvm::cl::opt<double> MixedTuningTime;
extern llvm::cl::opt<double> MixedTuningENOB;
extern llvm::cl::opt<double> MixedTuningCastingTime;
extern llvm::cl::opt<bool> MixedDoubleEnabled;
#define TUNING_CASTING (MixedTuningCastingTime)
#define TUNING_MATH (MixedTuningTime)
#define TUNING_ENOB (MixedTuningENOB)

#define FIX_DELTA_MAX 1

#define BIG_NUMBER 10000


using namespace llvm;

namespace tuner {
    template<class T, class U>
    std::shared_ptr<T> dynamic_ptr_cast_or_null(const std::shared_ptr<U> &r) noexcept {
        if (auto p = llvm::dyn_cast_or_null<typename std::shared_ptr<T>::element_type>(r.get())) {
            return std::shared_ptr<T>(r, p);
        } else {
            return std::shared_ptr<T>();
        }
    }


    //This class contains references to phi node that has no been closed yet
    class PhiWatcher {
    private:
        DenseMap<llvm::Value *, vector<PHINode *>> pairsToClose;


    public:
        void openPhiLoop(PHINode *phiNode, Value *requestedValue);

        PHINode *getPhiNodeToClose(Value *value);

        void closePhiLoop(PHINode *phiNode, Value *requestedNode);

        void dumpState();

    };

    class MemWatcher {
    private:
        DenseMap<llvm::Value *, vector<LoadInst *>> pairsToClose;


    public:
        void openPhiLoop(LoadInst *phiNode, Value *requestedValue);

        LoadInst *getPhiNodeToClose(Value *value);

        void closePhiLoop(LoadInst *phiNode, Value *requestedNode);

        void dumpState();

    };

    class Optimizer {
        public:
        ///Data related to function call
        std::unordered_map<std::string, llvm::Function *> known_functions;
        std::unordered_map<std::string, llvm::Function *> functions_still_to_visit;
        std::vector<llvm::Function *> call_stack;
        std::stack<shared_ptr<OptimizerInfo>> retStack;
        std::unique_ptr<MetricBase> metric;


        DenseMap<llvm::Value *, std::shared_ptr<OptimizerInfo>> valueToVariableName;
        Model model;
        llvm::Module &module;
        TaffoTuner *tuner;

        CPUCosts cpuCosts;
        PhiWatcher phiWatcher;
        MemWatcher memWatcher;

        int DisabledSkipped;
        int StatSelectedFixed=0;
        int StatSelectedDouble=0;
        int StatSelectedFloat=0;
        int StatSelectedHalf=0;
        int StatSelectedQuad=0;
        int StatSelectedPPC128=0;
        int StatSelectedFP80=0;
        int StatSelectedBF16=0;


        /*
        bool hasHalf;
        bool hasQuad;
        bool hasFP80 ;
        bool hasPPC128;
        bool hasBF16;*/

    public:


        void handleGlobal(GlobalObject *glob, shared_ptr<ValueInfo> valueInfo);

        bool finish();


        explicit Optimizer(llvm::Module &mm, TaffoTuner *tuner,  MetricBase*  met, std::string modelFile, CPUCosts::CostType cType);

        ~Optimizer();

        void initialize();

        void handleCallFromRoot(Function *f);

        std::shared_ptr<mdutils::MDInfo> getAssociatedMetadata(Value *pValue);

        void printStatInfos();

    public:
        void handleInstruction(Instruction *instruction, shared_ptr<ValueInfo> valueInfo);


        void emitError(const string& stringhina);


        shared_ptr<OptimizerInfo> getInfoOfValue(Value *value);

        void
        handleBinaryInstruction(Instruction *instr, const unsigned int OpCode, const shared_ptr<ValueInfo> &valueInfos);


        void insertTypeEqualityConstraint(shared_ptr<OptimizerScalarInfo> op1, shared_ptr<OptimizerScalarInfo> op2,
                                          bool forceFixBitsConstraint);

        shared_ptr<OptimizerInfo> handleGEPConstant(const ConstantExpr *cexp_i);


        bool valueHasInfo(Value *value);


        void
        processFunction(Function &function, list<shared_ptr<OptimizerInfo>> argInfo, shared_ptr<OptimizerInfo> retInfo);

        void handleTerminators(Instruction *term, shared_ptr<ValueInfo> valueInfo);

        shared_ptr<OptimizerScalarInfo>
        handleBinOpCommon(Instruction *instr, Value *op1, Value *op2, bool forceFixEquality,
                          shared_ptr<ValueInfo> valueInfos);

        void saveInfoForPointer(Value *value, shared_ptr<OptimizerPointerInfo> pointerInfo);


        shared_ptr<mdutils::TType> modelvarToTType(shared_ptr<OptimizerScalarInfo> sharedPtr);

        shared_ptr<mdutils::MDInfo> buildDataHierarchy(shared_ptr<OptimizerInfo> info);

        void handleUnknownFunction(Instruction *call_i, shared_ptr<ValueInfo> valueInfo);


        friend class MetricBase;
    };


}


#endif
