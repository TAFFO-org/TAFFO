#include "Optimizer.h"

using namespace tuner;

void PhiWatcher::openPhiLoop(PHINode *phiNode, Value *requestedValue) {
    if (pairsToClose.find(requestedValue) == pairsToClose.end()) {
        pairsToClose.insert(make_pair(requestedValue, vector<PHINode *>()));
    }

    pairsToClose[requestedValue].push_back(phiNode);
}


PHINode *PhiWatcher::getPhiNodeToClose(Value *value) {
    auto workingEntry = pairsToClose.find(value);
    if ( workingEntry == pairsToClose.end()) {
        return nullptr;
    }

    return workingEntry->getSecond().begin().operator*();
}

void PhiWatcher::closePhiLoop(PHINode *phiNode, Value *requestedValue) {
    auto workingEntry = pairsToClose.find(requestedValue);
    if ( workingEntry == pairsToClose.end()) {
        llvm_unreachable("Tried to close an already closed phiLoop!");
    }

    auto toDelete = std::find(workingEntry->getSecond().begin(),
                              workingEntry->getSecond().end(),
                              phiNode);
    workingEntry->getSecond().erase(toDelete);

    if(workingEntry->getSecond().empty()){
        pairsToClose.erase(workingEntry);
    }


}

void PhiWatcher::dumpState() {
    if(pairsToClose.empty()){
        LLVM_DEBUG(dbgs() << "All Phi loops closed!\n";);
    }
    for(auto pair :pairsToClose){
        LLVM_DEBUG(pair.first->print(dbgs()););
        LLVM_DEBUG(dbgs() << " STILL MISSING; will close:\n";);
        for(auto a : pair.second){
            LLVM_DEBUG(a->print(dbgs()););
            LLVM_DEBUG(dbgs() << "\n";);
        }
    }

}
