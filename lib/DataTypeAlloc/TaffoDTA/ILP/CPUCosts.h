#include <map>
#include <string>
#include <vector>
#include "llvm/Analysis/TargetTransformInfo.h"


#ifndef __TAFFO_DTA_CPUCOSTS_H__
#define __TAFFO_DTA_CPUCOSTS_H__


using namespace std;

namespace tuner {
    ///This class will manage data about the costs in a cpu model
    ///These data are loaded from a file!
    class CPUCosts {

    public:
        enum class CostType{
            Performance,
            Size,
            LLVM_CodeSize,
            LLVM_Latency,
            LLVM_RecipThroughput,
            LLVM_SizeAndLatency
        };
        
        enum CostsId {
        ADD_FIX=0, ADD_FLOAT, ADD_DOUBLE, ADD_HALF, ADD_QUAD, ADD_FP80, ADD_PPC128, ADD_BF16, 
        SUB_FIX, SUB_FLOAT, SUB_DOUBLE, SUB_HALF, SUB_QUAD, SUB_FP80, SUB_PPC128, SUB_BF16, 
        MUL_FIX, MUL_FLOAT, MUL_DOUBLE, MUL_HALF, MUL_QUAD, MUL_FP80, MUL_PPC128, MUL_BF16, 
        DIV_FIX, DIV_FLOAT, DIV_DOUBLE, DIV_HALF, DIV_QUAD, DIV_FP80, DIV_PPC128, DIV_BF16, 
        REM_FIX, REM_FLOAT, REM_DOUBLE, REM_HALF, REM_QUAD, REM_FP80, REM_PPC128, REM_BF16, 
        CAST_FIX_FLOAT, CAST_FIX_DOUBLE, CAST_FIX_HALF, CAST_FIX_QUAD, CAST_FIX_FP80, CAST_FIX_PPC128, CAST_FIX_BF16, 
        CAST_FLOAT_FIX, CAST_FLOAT_DOUBLE, CAST_FLOAT_HALF, CAST_FLOAT_QUAD, CAST_FLOAT_FP80, CAST_FLOAT_PPC128, CAST_FLOAT_BF16, 
        CAST_DOUBLE_FIX, CAST_DOUBLE_FLOAT, CAST_DOUBLE_HALF, CAST_DOUBLE_QUAD, CAST_DOUBLE_FP80, CAST_DOUBLE_PPC128, CAST_DOUBLE_BF16, 
        CAST_HALF_FIX, CAST_HALF_FLOAT, CAST_HALF_DOUBLE, CAST_HALF_QUAD, CAST_HALF_FP80, CAST_HALF_PPC128, CAST_HALF_BF16, 
        CAST_QUAD_FIX, CAST_QUAD_FLOAT, CAST_QUAD_DOUBLE, CAST_QUAD_HALF, CAST_QUAD_FP80, CAST_QUAD_PPC128, CAST_QUAD_BF16, 
        CAST_FP80_FIX, CAST_FP80_FLOAT, CAST_FP80_DOUBLE, CAST_FP80_HALF, CAST_FP80_QUAD, CAST_FP80_PPC128, CAST_FP80_BF16, 
        CAST_PPC128_FIX, CAST_PPC128_FLOAT, CAST_PPC128_DOUBLE, CAST_PPC128_HALF, CAST_PPC128_QUAD, CAST_PPC128_FP80, CAST_PPC128_BF16, 
        CAST_BF16_FIX, CAST_BF16_FLOAT, CAST_BF16_DOUBLE, CAST_BF16_HALF, CAST_BF16_QUAD, CAST_BF16_FP80, CAST_BF16_PPC128,
        CAST_FIX_FIX
        };

    const int CostsNumber  = 97;

    static const std::vector<std::string> CostsIdValues;
    private:
        std::map<CostsId, double> costsMap;
        std::map<std::string, int> disableNum;
        std::map<CostsId, bool> disableMap;

        CostType cType;
        auto getType(unsigned int n, const string& tmpString ,llvm::LLVMContext& context, llvm::Module& module) -> llvm::Type*;
        void LLVMInizializer(llvm::Module& module, llvm::TargetTransformInfo& TTI, llvm::TargetTransformInfo::TargetCostKind costKind );
        void SizeInizializer(llvm::Module& module, llvm::TargetTransformInfo& TTI);
        void ApplyAttribute(string& attr);
        void InizializeDisabledList();
        void loadInstructionSet();
        int n_types = 0;
    public:
        CPUCosts();

        bool isDisabled(CostsId id) const;

        CPUCosts(string& modelFile, CostType cType = CostType::Performance );

        CPUCosts(llvm::Module& module, llvm::TargetTransformInfo& TTI, CostType cType = CostType::Size );

        void loadModelFile(string& basicString);

        static CostsId decodeId(const string &basicString);

        static string CostsIdToString(CostsId id);

        void dump();

        bool cast_with_fix(CPUCosts::CostsId& cast);

        double getCost(CostsId id);
        std::pair<double,double> MaxMinCosts(const string& ref);

        CPUCosts(const CPUCosts &rhs)
            : costsMap(rhs.costsMap), disableNum(rhs.disableNum),
              disableMap(rhs.disableMap), cType(rhs.cType),
              n_types(rhs.n_types) {}

        CPUCosts(CPUCosts &&rhs) noexcept
            : costsMap(std::move(rhs.costsMap)),
              disableNum(std::move(rhs.disableNum)),
              disableMap(std::move(rhs.disableMap)), cType(rhs.cType),
              n_types(rhs.n_types) {}

        CPUCosts& operator=(const CPUCosts& rhs ){
            if(this != &rhs){
                n_types = rhs.n_types; 
                disableNum.clear();
                disableMap.clear();
                costsMap.clear();
                cType = rhs.cType;
                costsMap = rhs.costsMap;
                disableNum = rhs.disableNum;
                disableMap = rhs.disableMap;
            }
            return *this;
        }

        CPUCosts& operator=(CPUCosts&& rhs ){
            if(this != &rhs){
                n_types = rhs.n_types; 
                disableNum.clear();
                disableMap.clear();                                
                costsMap.clear();
                cType = rhs.cType;
                costsMap = std::move(rhs.costsMap);
                disableNum = std::move(rhs.disableNum);
                disableMap = std::move(rhs.disableMap);
            }
            return *this;
        }




    };


}

#endif
