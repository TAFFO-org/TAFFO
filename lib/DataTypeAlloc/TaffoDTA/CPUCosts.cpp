#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <vector>
#include <llvm/Support/Debug.h>
#include "llvm/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "CPUCosts.h"
#include "llvm/IR/IRBuilder.h"
#include "algorithm"
#include "LLVMVersions.h"

#define HUGE 10110
#define DEBUG_TYPE "taffo-dta"
using namespace tuner;
using namespace std;

extern std::string InstructionSet;
extern bool hasHalf;
extern bool hasQuad;
extern bool hasPPC128;
extern bool hasFP80;
extern bool hasBF16;


const std::vector<std::string> CPUCosts::CostsIdValues = {"ADD_FIX", "ADD_FLOAT", "ADD_DOUBLE", "ADD_HALF", "ADD_QUAD", "ADD_FP80", "ADD_PPC128", "ADD_BF16", 
"SUB_FIX", "SUB_FLOAT", "SUB_DOUBLE", "SUB_HALF", "SUB_QUAD", "SUB_FP80", "SUB_PPC128", "SUB_BF16", 
"MUL_FIX", "MUL_FLOAT", "MUL_DOUBLE", "MUL_HALF", "MUL_QUAD", "MUL_FP80", "MUL_PPC128", "MUL_BF16", 
"DIV_FIX", "DIV_FLOAT", "DIV_DOUBLE", "DIV_HALF", "DIV_QUAD", "DIV_FP80", "DIV_PPC128", "DIV_BF16", 
"REM_FIX", "REM_FLOAT", "REM_DOUBLE", "REM_HALF", "REM_QUAD", "REM_FP80", "REM_PPC128", "REM_BF16", 
"CAST_FIX_FLOAT", "CAST_FIX_DOUBLE", "CAST_FIX_HALF", "CAST_FIX_QUAD", "CAST_FIX_FP80", "CAST_FIX_PPC128", "CAST_FIX_BF16", 
"CAST_FLOAT_FIX", "CAST_FLOAT_DOUBLE", "CAST_FLOAT_HALF", "CAST_FLOAT_QUAD", "CAST_FLOAT_FP80", "CAST_FLOAT_PPC128", "CAST_FLOAT_BF16", 
"CAST_DOUBLE_FIX", "CAST_DOUBLE_FLOAT", "CAST_DOUBLE_HALF", "CAST_DOUBLE_QUAD", "CAST_DOUBLE_FP80", "CAST_DOUBLE_PPC128", "CAST_DOUBLE_BF16", 
"CAST_HALF_FIX", "CAST_HALF_FLOAT", "CAST_HALF_DOUBLE", "CAST_HALF_QUAD", "CAST_HALF_FP80", "CAST_HALF_PPC128", "CAST_HALF_BF16", 
"CAST_QUAD_FIX", "CAST_QUAD_FLOAT", "CAST_QUAD_DOUBLE", "CAST_QUAD_HALF", "CAST_QUAD_FP80", "CAST_QUAD_PPC128", "CAST_QUAD_BF16", 
"CAST_FP80_FIX", "CAST_FP80_FLOAT", "CAST_FP80_DOUBLE", "CAST_FP80_HALF", "CAST_FP80_QUAD", "CAST_FP80_PPC128", "CAST_FP80_BF16", 
"CAST_PPC128_FIX", "CAST_PPC128_FLOAT", "CAST_PPC128_DOUBLE", "CAST_PPC128_HALF", "CAST_PPC128_QUAD", "CAST_PPC128_FP80", "CAST_PPC128_BF16", 
"CAST_BF16_FIX", "CAST_BF16_FLOAT", "CAST_BF16_DOUBLE", "CAST_BF16_HALF", "CAST_BF16_QUAD", "CAST_BF16_FP80", "CAST_BF16_PPC128", 
"CAST_FIX_FIX"};




constexpr int N =  4;


std::string& trim(std::string& tmp){
    tmp.erase(remove(tmp.begin(), tmp.end(), ' '), tmp.end());
    return tmp;
}

std::string trim(std::string&& tmp){
    tmp.erase(remove(tmp.begin(), tmp.end(), ' '), tmp.end());
    return std::move(tmp);
}



void CPUCosts::InizializeDisabledList(){
    LLVM_DEBUG(llvm::dbgs() << "### Disabled List ###\n" );
 for(const auto& tmp : CostsIdValues){
    disableMap[decodeId(tmp)] = false;
    if(tmp.find("ADD_") == 0 ){
        disableNum[trim(tmp.substr(N,std::string::npos))] = 0;        
        LLVM_DEBUG(llvm::dbgs() << tmp.substr(N,std::string::npos) << ": " << disableNum[tmp.substr(N,std::string::npos)] << "\n");
    }
    if(tmp.find("HALF") != string::npos ){
        n_types++;
    }
 }

}

CPUCosts::CPUCosts(string& modelFile, CostType cType) {
    this->cType = cType;
    // File pointer
    InizializeDisabledList();
    loadModelFile(modelFile);
    loadInstructionSet();
}

CPUCosts::CPUCosts(llvm::Module &module, llvm::TargetTransformInfo &TTI,
                   CostType cType) {

  InizializeDisabledList();
  this->cType = cType;
  if (cType == CostType::Size) {
    this->SizeInizializer(module, TTI);
  }
  if (cType == CostType::LLVM_CodeSize) {
    this->LLVMInizializer(module, TTI, llvm::TargetTransformInfo::TCK_CodeSize);
  }
  if (cType == CostType::LLVM_Latency) {
    this->LLVMInizializer(module, TTI, llvm::TargetTransformInfo::TCK_Latency);
  }
  if (cType == CostType::LLVM_RecipThroughput) {
    this->LLVMInizializer(module, TTI,
                          llvm::TargetTransformInfo::TCK_RecipThroughput);
  }
  if (cType == CostType::LLVM_SizeAndLatency) {
    this->LLVMInizializer(module, TTI,
                          llvm::TargetTransformInfo::TCK_SizeAndLatency);
  }
  loadInstructionSet();
}

CPUCosts::CPUCosts() = default;


/*
return pair<max,min> from all the costs that start with ref 
*/
std::pair<double,double> CPUCosts::MaxMinCosts(const string& ref){
    double max;
    double min;
    max = min = getCost(decodeId(
        *std::find_if(CostsIdValues.begin(), CostsIdValues.end(),
                      [this, ref](const string &tmp) { return tmp.find(ref) == 0 && !disableMap[decodeId(tmp)]; 
                      })));
    std::for_each(CostsIdValues.begin(), CostsIdValues.end(),
                  [this, ref, &max, &min](const auto &tmp) {
                    if (tmp.find(ref) == 0 && !disableMap[decodeId(tmp)]) {
                      max = std::max(max, getCost(decodeId(tmp)));
                      min = std::min(min, getCost(decodeId(tmp)));
                    }
                  });
    return {max,min};
}



llvm::Type* CPUCosts::getType(unsigned int n, const string& tmpString ,llvm::LLVMContext& context, llvm::Module& module){
            LLVM_DEBUG(llvm::dbgs() << tmpString << " with n " << n << " to " <<  tmpString.substr(n) <<"\n");
            if(tmpString.find("FIX",n) == n){
            return llvm::Type::getIntNTy(context,module.getDataLayout().getPointerSizeInBits());
            }
            if(tmpString.find("FLOAT",n) == n){
                return llvm::Type::getFloatTy(context);
                }            
            if(tmpString.find("DOUBLE",n) == n){
                return llvm::Type::getDoubleTy(context);
                }
            if(tmpString.find("HALF",n) == n){
                return llvm::Type::getHalfTy(context);
                }
            if(tmpString.find("QUAD",n) == n){   
                return llvm::Type::getFP128Ty(context);
                }       
            if(tmpString.find("FP80",n) == n){
                return llvm::Type::getX86_FP80Ty(context);
                }
            if(tmpString.find("PPC128",n) == n){
                return llvm::Type::getPPC_FP128Ty(context);
                }                
            if(tmpString.find("BF16",n) == n){
                return llvm::Type::getBFloatTy(context);
                } 

            llvm_unreachable("");
}


bool CPUCosts::isDisabled(CostsId id) const{
    return disableMap.at(id);
}

void CPUCosts::SizeInizializer(llvm::Module& module, llvm::TargetTransformInfo& TTI){
    LLVM_DEBUG(llvm::dbgs() << "\n########### Cpu Size ###########\n");
    CPUCosts::LLVMInizializer(module, TTI, llvm::TargetTransformInfo::TargetCostKind::TCK_RecipThroughput);
    auto& context = module.getContext();
    double cost_inst = 0;
    for(const auto & tmpString : CostsIdValues)
    {
    if(tmpString.find("CAST") != 0){
        llvm::Type* type = nullptr;
        type = getType(N, tmpString, context, module);
        LLVM_DEBUG(llvm::dbgs() << "Size of " << tmpString << " " << type->getPrimitiveSizeInBits() << "\n");
        cost_inst = (double) type->getPrimitiveSizeInBits();
    }
    else{
            int first_start = 5;
            int second_start = tmpString.find("_", first_start) + 1;
            llvm::Type* second_type = getType(second_start, tmpString, context, module);
            LLVM_DEBUG(llvm::dbgs() << "Size of " << tmpString  << " " << second_type->getPrimitiveSizeInBits() << "\n");
            cost_inst = second_type->getPrimitiveSizeInBits();
    }
    if(auto finded = costsMap.find(decodeId(tmpString)); finded == costsMap.end()){
    costsMap.insert({decodeId(tmpString) ,cost_inst});
    }else{
        finded->second = finded->second * cost_inst;
        
    }
    }

    


}

void CPUCosts::LLVMInizializer(llvm::Module& module, llvm::TargetTransformInfo& TTI, llvm::TargetTransformInfo::TargetCostKind costKind ){

    LLVM_DEBUG(llvm::dbgs() << "\n########### Cpu LLVM ###########\n");
    auto& context = module.getContext();
    llvm::BasicBlock* TestBlock = llvm::BasicBlock::Create(context, "TO_REMOVE", &(*(module.begin())));
    llvm::IRBuilder<> builder(TestBlock);

    for(const auto & tmpString : CostsIdValues)
    {
        llvm::Type* type = nullptr;
        llvm::Instruction* inst = nullptr;
        int cost_inst = 0;
        if(tmpString.find("CAST") != 0){
        type = getType(N, tmpString, context, module);
        auto*  first_alloca = builder.CreateAlloca(type);
        auto* second_alloca = builder.CreateAlloca(type);
        auto* first_load = builder.CreateLoad(first_alloca);
        auto* second_load = builder.CreateLoad(second_alloca);
        if(tmpString.find("ADD") == 0){        
        if(type->isIntegerTy())
        inst = llvm::cast<llvm::Instruction>(builder.CreateAdd(first_load,second_load));
        else
        inst = llvm::cast<llvm::Instruction>(builder.CreateFAdd(first_load,second_load));
        }
        else if(tmpString.find("SUB") == 0){        
        if(type->isIntegerTy())
        inst = llvm::cast<llvm::Instruction>(builder.CreateSub(first_load,second_load));
        else
        inst = llvm::cast<llvm::Instruction>(builder.CreateFSub(first_load,second_load));
        }
        else if(tmpString.find("MUL") == 0){        
        if(type->isIntegerTy())
        inst = llvm::cast<llvm::Instruction>(builder.CreateMul(first_load,second_load));
        else
        inst = llvm::cast<llvm::Instruction>(builder.CreateFMul(first_load,second_load));
        }
        else if(tmpString.find("DIV") == 0){        
        if(type->isIntegerTy())
        inst = llvm::cast<llvm::Instruction>(builder.CreateSDiv(first_load,second_load));
        else
        inst = llvm::cast<llvm::Instruction>(builder.CreateFDiv(first_load,second_load));

        }
        else if(tmpString.find("REM") == 0){        
        if(type->isIntegerTy())
        inst = llvm::cast<llvm::Instruction>(builder.CreateSRem(first_load,second_load));
        else
        inst = llvm::cast<llvm::Instruction>(builder.CreateFRem(first_load,second_load));

        }
        cost_inst = getInstructionCost(TTI, inst, costKind);
        LLVM_DEBUG(llvm::dbgs() << tmpString << ": " << cost_inst << "\n"); 
        inst->eraseFromParent();
        second_load->eraseFromParent();
        first_load->eraseFromParent();
        second_alloca->eraseFromParent();
        first_alloca->eraseFromParent();
        } else
        {
            int first_start = 5;
            int second_start = tmpString.find("_", first_start) + 1;
            llvm::Type* first_type = getType(first_start, tmpString, context, module);
            llvm::Type* second_type = getType(second_start, tmpString, context, module);
            auto*  first_alloca = builder.CreateAlloca(first_type);
            auto* first_load = builder.CreateLoad(first_alloca);
            if(first_type->isIntegerTy() && second_type->isIntegerTy()){
            inst = llvm::cast<llvm::Instruction>(builder.CreateIntCast(first_load,second_type, false));                                             
            }
            else if(first_type->isIntegerTy() && !second_type->isIntegerTy()){
                inst = llvm::cast<llvm::Instruction>(builder.CreateUIToFP(first_load,second_type));
            }
            else if(!first_type->isIntegerTy() && second_type->isIntegerTy()){
                inst = llvm::cast<llvm::Instruction>(builder.CreateFPToUI(first_load,second_type));
            }
            else if(!first_type->isIntegerTy() && !second_type->isIntegerTy()){
                inst = llvm::cast<llvm::Instruction>(builder.CreateFPCast(first_load,second_type));
            }            
            cost_inst = getInstructionCost( TTI, inst, llvm::TargetTransformInfo::TargetCostKind::TCK_RecipThroughput); 
            LLVM_DEBUG(llvm::dbgs() << tmpString << ": " << cost_inst << "\n");
            if (inst != first_load)
              inst->eraseFromParent();
            if (first_load != nullptr)
              first_load->eraseFromParent();
            if (first_alloca != nullptr)
              first_alloca->eraseFromParent();
            inst = nullptr;
            first_load = nullptr;
            first_alloca = nullptr;
        }
        costsMap.insert({decodeId(tmpString) ,cost_inst});
    }
    TestBlock->eraseFromParent();  
}













void CPUCosts::ApplyAttribute(string& attr){
  switch (static_cast<int>(attr[0])) {
  case static_cast<int>('-'): {
    attr.erase(0, 1);
    auto dec = decodeId(attr);
    disableMap[dec] = true;
    costsMap.at(dec) = HUGE;
    if(attr.find("CAST") != 0){
    disableNum.at(attr.substr(N, string::npos))++;
    }else{
        disableNum.at(attr.substr(N,attr.find("_", N)-N))++;
        disableNum.at(attr.substr(attr.find("_", N)+1, string::npos))++;
    }
  } break;
  case static_cast<int>('N'):
    attr.erase(0, 1);
    for (auto &tmp : CostsIdValues) {
      if (tmp.find(attr) != std::string::npos) {
        {
          disableNum.at(attr)++;
          auto dec = decodeId(tmp);
          disableMap[dec] = true;
          costsMap.at(dec) = HUGE;
          LLVM_DEBUG(llvm::dbgs() << attr << ": " << disableNum.at(attr)
                                  << " cost " << costsMap.at(dec) << "\n");
        }
      }
    }
    break;
  default:
    llvm_unreachable("Attribute not found instruction file invalid");
  }
}

void CPUCosts::loadInstructionSet(){
    LLVM_DEBUG(llvm::dbgs() << "\n### Load Instruction Set ###\n" );
    fstream fin;

    fin.open(InstructionSet, ios::in);
    if(!fin.is_open()){
        LLVM_DEBUG(llvm::dbgs() << "Instruction Set not provided" << "\n");
        return;
    }
    
    string line;
    //for each line in the file
    while (getline(fin, line)) {
        if(line[line.find_first_not_of(" \t\r")] == '#') continue;
        unsigned long start = 0;
        unsigned long end = line.find(',');
        LLVM_DEBUG(llvm::dbgs() << "line: " <<  line << "\n" );
        while(end != std::string::npos ){
            string attr = line.substr(start,end-start);
            LLVM_DEBUG(llvm::dbgs() << "Before trim "<< attr<<  " start: " << start <<  " end: " << end << "\n"); 
            trim(attr);
            LLVM_DEBUG(llvm::dbgs() << "Attribute trimmed " << attr <<"\n");
            ApplyAttribute(attr);
            end = end + 1;
            start = end;           
            end = line.find(',',end);
        }
        string attr = line.substr(start,line.length()-start);
        trim(attr);
        LLVM_DEBUG(llvm::dbgs() << "Attribute trimmed " << attr << "\n");
        ApplyAttribute(attr);
    }
    fin.close();

        LLVM_DEBUG(llvm::dbgs() << "ntype : " << n_types << "\n");
    for(const auto& values: disableNum){
        LLVM_DEBUG(llvm::dbgs() << "Values : ntype  -> " << values.first << " : " << values.second << "\n"); 
        if(values.second == n_types){
            if(values.first.find("HALF") == 0){
                LLVM_DEBUG(llvm::dbgs() << "No half\n");
                hasHalf = false;
            }else
            if(values.first.find("QUAD") == 0){
                LLVM_DEBUG(llvm::dbgs() << "No quad\n");
                hasQuad = false;
            }else
            if(values.first.find("FP80") == 0){
                LLVM_DEBUG(llvm::dbgs() << "No fp80\n");
                hasFP80 = false;
            }else
            if(values.first.find("PPC128") == 0){
                LLVM_DEBUG(llvm::dbgs() << "No ppc128\n");
                hasPPC128 = false;
            }else
            if(values.first.find("BF16") == 0){
                LLVM_DEBUG(llvm::dbgs() << "No bf16\n");
                hasBF16 = false;
            }else{
                llvm_unreachable((string("Not supported disabled type ") + values.first).c_str());
            }              
        }
    }


}


void CPUCosts::loadModelFile(string& modelFile) {
    LLVM_DEBUG(llvm::dbgs() << "### Load Model ###\n" );
    fstream fin;

    fin.open(modelFile, ios::in);

    assert(fin.is_open() && "Cannot open model file!");

    string line, field, temp;
    vector<string> row;

    //for each line in the file
    int nline=0;
    while (getline(fin, line)) {

        //read the file until a newline is found (discarded from final string)
        row.clear();
        double value = 0;
        nline++;


        //Generate a stream in order to be used by getLine
        stringstream lineStream(line);
        //llvm::dbgs() << "Line: " << line << "\n";

        while (getline(lineStream, field, ',')) {
            row.push_back(field);
        }

        if (row.size() != 2) {
            LLVM_DEBUG(llvm::dbgs() << "Malformed line found: [" << line << "] on line"<< nline << ", skipping...\n";);
            continue;
        }

        CostsId id = decodeId(row[0]);
        value = stod(row[1]);

        if(costsMap.find(id) != costsMap.end()){
            LLVM_DEBUG(llvm::dbgs() << "Found duplicated info: [" << line << "], skipping...\n";);
            continue;
        }

        costsMap.insert(make_pair(id, value));
    }


    for (int i = 0; i < CostsNumber; i++){       

        if (costsMap.find(static_cast<CostsId>(i)) == costsMap.end()){

            costsMap.insert(make_pair(static_cast<CostsId>(i), 99999));

        }

    }



}

 CPUCosts::CostsId CPUCosts::decodeId(const string& basicString){
    auto it = std::find(CostsIdValues.cbegin(), CostsIdValues.cend(), basicString);

    if(it != CostsIdValues.cend()){
        int index = it - CostsIdValues.cbegin();
        //LLVM_DEBUG(llvm::dbgs() <<  " found [" << *it <<"]\n" );
        return CostsId(index);
    }

    LLVM_DEBUG( 
        {
            for ( const auto& i : CostsIdValues ){
                LLVM_DEBUG(llvm::dbgs() << i << " = " << basicString << " : " << std::to_string(basicString == i) << "\n";);
            }
        }
    );
    LLVM_DEBUG(llvm::dbgs() << "Unknown value: "<<basicString<<"\n";);

    llvm_unreachable("Unknown cost value!");
}

string CPUCosts::CostsIdToString(CostsId id){
    return CostsIdValues[id];
}

void CPUCosts::dump(){
    LLVM_DEBUG(llvm::dbgs() << "Available model costs:\n";);
    for(auto pair : costsMap){
        LLVM_DEBUG(llvm::dbgs() << "[" << CostsIdToString(pair.first) << ", " << pair.second << "]\n";);
    }
}

bool CPUCosts::cast_with_fix(CPUCosts::CostsId& cast){
return CostsIdToString(cast).find("FIX") !=  string::npos;
}


double CPUCosts::getCost(CPUCosts::CostsId id) {
    //FIXME: the workaround is done due to the model not being able to distiguish between a delta in fixp due to change of datatype or change of bit number
    // in this way should fix the 90% of the cases

    bool fixDoubleCast = false;

    if(id != CAST_FIX_FIX && cType == CPUCosts::CostType::Performance){
        fixDoubleCast = cast_with_fix(id);
    }


    auto it = costsMap.find(id);
    if(it!=costsMap.end()){
        if(fixDoubleCast){
            return it->second - getCost(CAST_FIX_FIX);
        }
        return it->second;
    }

    llvm_unreachable("This cost was not loaded from model file!");
}
