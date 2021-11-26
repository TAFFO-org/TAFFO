#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "InputInfo.h"
#include "Metadata.h"
#include "TypeUtils.h"

#ifndef __TAFFO_TUNER_PASS_H__
#define __TAFFO_TUNER_PASS_H__

#define DEBUG_TYPE "taffo-dta"
#define DEBUG_FUN  "tunerfunction"


llvm::cl::opt<int> FracThreshold("minfractbits", llvm::cl::value_desc("bits"),
    llvm::cl::desc("Threshold of fractional bits in fixed point numbers"), llvm::cl::init(3));
llvm::cl::opt<int> TotalBits("totalbits", llvm::cl::value_desc("bits"),
    llvm::cl::desc("Total amount of bits in fixed point numbers"), llvm::cl::init(32));
llvm::cl::opt<int> SimilarBits("similarbits", llvm::cl::value_desc("bits"),
    llvm::cl::desc("Maximum number of difference bits that leads two fixp formats to merge"), llvm::cl::init(2));
llvm::cl::opt<bool> DisableTypeMerging("notypemerge",
    llvm::cl::desc("Disables adjacent type optimization"), llvm::cl::init(false));
llvm::cl::opt<bool> IterativeMerging("iterative",
    llvm::cl::desc("Enables old iterative merging"), llvm::cl::init(false));


STATISTIC(FixCast, "Number of fixed point format cast");


namespace tuner {

  struct ValueInfo {
    std::shared_ptr<mdutils::MDInfo> metadata;
    std::shared_ptr<mdutils::TType> initialType;
  };

  struct FunInfo {
    llvm::Function* newFun;
    /* {function argument index, type of argument}
     * argument idx is -1 for return value */
    std::vector<std::pair<int, std::shared_ptr<mdutils::MDInfo>>> fixArgs;
  };


  struct TaffoTuner : public llvm::ModulePass {
    static char ID;

    /* to not be accessed directly, use valueInfo() */
    llvm::DenseMap<llvm::Value *, std::shared_ptr<ValueInfo>> info;
    
    /* original function -> cloned function map */
    llvm::DenseMap<llvm::Function*, std::vector<FunInfo>> functionPool;

    TaffoTuner(): ModulePass(ID) { }
    bool runOnModule(llvm::Module &M) override;

    void retrieveAllMetadata(llvm::Module &m, std::vector<llvm::Value *> &vals,
			     llvm::SmallPtrSetImpl<llvm::Value *> &valset);
    bool processMetadataOfValue(llvm::Value *v, mdutils::MDInfo *MDI);
    bool associateFixFormat(mdutils::InputInfo& rng);
    void sortQueue(std::vector<llvm::Value *> &vals, llvm::SmallPtrSetImpl<llvm::Value *> &valset);
    void mergeFixFormat(const std::vector<llvm::Value *> &vals,
			const llvm::SmallPtrSetImpl<llvm::Value *> &valset);

    bool mergeFixFormat(llvm::Value *v, llvm::Value *u);
    bool mergeFixFormatIterative(llvm::Value *v, llvm::Value *u);
    bool isMergeable(mdutils::FPType *fpv, mdutils::FPType *fpu) const;
    std::shared_ptr<mdutils::FPType> merge(mdutils::FPType *fpv, mdutils::FPType *fpu) const;


    void restoreTypesAcrossFunctionCall(llvm::Value *arg_or_call_param);
    void setTypesOnCallArgumentFromFunctionArgument(llvm::Argument *arg, std::shared_ptr<mdutils::MDInfo> finalMd);
    std::vector<llvm::Function*> collapseFunction(llvm::Module &m);
    llvm::Function *findEqFunction(llvm::Function *fun, llvm::Function *origin);
    void attachFPMetaData(std::vector<llvm::Value*> &vals);
    void attachFunctionMetaData(llvm::Module &m);


    std::shared_ptr<ValueInfo> valueInfo(llvm::Value *val) {
      auto vi = info.find(val);
      if (vi == info.end()) {
        LLVM_DEBUG(llvm::dbgs() << "new valueinfo for " << *val << "\n");
        info[val] = std::make_shared<ValueInfo>(ValueInfo());
        return info[val];
      } else {
        return vi->getSecond();
      }
    }

    bool hasInfo(llvm::Value *val) {
      return info.find(val) != info.end();
    }

    bool conversionDisabled(llvm::Value *val) {
      if (llvm::isa<llvm::Constant>(val))
        return false;
      if (llvm::isa<llvm::Argument>(val)) {
        if (!hasInfo(val))
          return true;
        return !(valueInfo(val)->metadata && valueInfo(val)->metadata->getEnableConversion());
      }
      mdutils::MetadataManager &MDManager = mdutils::MetadataManager::getMetadataManager();
      mdutils::MDInfo *mdi = MDManager.retrieveMDInfo(val);
      return !(mdi && mdi->getEnableConversion())
	&& incomingValuesDisabled(val);
    }

    bool incomingValuesDisabled(llvm::Value *v) {
      using namespace llvm;
      if (!taffo::isFloatType(v->getType()))
	return true;

      if (PHINode *phi = dyn_cast<PHINode>(v)) {
	bool disabled = false;
	for (Value *inc : phi->incoming_values()) {
	  if (!isa<PHINode>(inc) && conversionDisabled(inc)) {
	    disabled = true;
	    break;
	  }
        }
	return disabled;
      } else {
	return true;
      }
    }
  };
}


#endif

