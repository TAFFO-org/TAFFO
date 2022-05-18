#include "ExpandEqualValue.h"

#include <list>
#include <unordered_set>

#include "TracingUtils.h"

namespace taffo {

using namespace llvm;

ExpandEqualValue::ExpandEqualValue(std::unordered_map<int, std::list<std::shared_ptr<taffo::ValueWrapper>>> &CCValues)
    : ccValues{CCValues}
{
  expandEqualValues();
}

void ExpandEqualValue::expandEqualValues()
{
  for (auto &pair: ccValues) {
    auto &key = pair.first;
    auto &valueList = pair.second;
    auto &expandedValueList = expandedCCValues[key];
    std::unordered_set<llvm::Value*> visited{};
    std::list<llvm::Value*> queue{};

    auto addToQueue = [&visited, &queue](Value * V) -> void {
      if (visited.count(V) == 0) {
        queue.push_back(V);
      }
    };

    for (auto &wrapper: valueList) {
      expandedValueList.push_back(wrapper);
      addToQueue(wrapper->value);
    }
    while (!queue.empty()) {
      auto *V = queue.front();
      queue.pop_front();
      if (visited.count(V) > 0) {
        continue ;
      }

      if (auto *storeInst = llvm::dyn_cast<llvm::StoreInst>(V)) {
        auto *srcInst = storeInst->getValueOperand();
        if (!srcInst->getType()->isPointerTy()) {
          expandedValueList.push_back(ValueWrapper::wrapValue(srcInst));
          addToQueue(srcInst);
        }
      } else if (auto *loadInst = llvm::dyn_cast<llvm::LoadInst>(V)) {
        if (!loadInst->getType()->isPointerTy()) {
          for (auto &use: loadInst->uses()) {
            auto * user = use.getUser();
            if (isa<CallInst, StoreInst>(user)) {
              expandedValueList.push_back(ValueWrapper::wrapValueUse(&use));
              addToQueue(user);
            } else {
              addToQueue(user);
            }
          }
        }
      } else if (auto *funArg = llvm::dyn_cast<llvm::Argument>(V)) {
//        auto *F = funArg->getParent();
//        for (auto *user: F->users()) {
//          if (isa<CallInst>(user) || isa<InvokeInst>(user)) {
//            auto *call = dyn_cast<CallBase>(user);
//            auto &callArg = call->getArgOperandUse(funArg->getArgNo());
//            if (!callArg.get()->getType()->isPointerTy()) {
//              errs() << "expanding arg: "
//                     << *funArg
//                     << ": " << *(callArg.get()) << "\n";
//              expandedValueList.push_back(ValueWrapper::wrapValueUse(&callArg));
//              expandedValueList.push_back(ValueWrapper::wrapValue(callArg.get()));
//              //            addToQueue(callArg.getUser());
//            }
//          }
//        }
      }

      visited.insert(V);
    }
  }
}

}