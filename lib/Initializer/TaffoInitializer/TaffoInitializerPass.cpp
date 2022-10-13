#include "TaffoInitializerPass.h"
#include "IndirectCallPatcher.h"
#include "Metadata.h"
#include "TypeUtils.h"
#include "OpenCLKernelPatcher.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <climits>
#include <cmath>

#define DEBUG_TYPE "taffo-init"

using namespace llvm;
using namespace taffo;

cl::opt<bool> ManualFunctionCloning("manualclone",
    cl::desc("Enables function cloning only for annotated functions"),
    cl::init(false));

cl::opt<bool> OpenCLKernelMode("oclkern",
    cl::desc("Allows cloning of OpenCL kernel functions"),
    cl::init(false));


PreservedAnalyses TaffoInitializer::run(Module &m, ModuleAnalysisManager &AM)
{
  if (OpenCLKernelMode) {
    LLVM_DEBUG(dbgs() << "OpenCLKernelMode == true!\n");
    createOpenCLKernelTrampolines(m);
  }

  LLVM_DEBUG(printAnnotatedObj(m));

  manageIndirectCalls(m);

  ConvQueueT local;
  ConvQueueT global;
  readAllLocalAnnotations(m, local);
  readGlobalAnnotations(m, global, true);
  readGlobalAnnotations(m, global, false);

  Function *startingPoint = findStartingPointFunctionGlobal(m);
  if (startingPoint) {
    LLVM_DEBUG(dbgs() << "Found starting point using global __taffo_vra_starting_function: " << startingPoint->getName() << "\n");
    mdutils::MetadataManager::setStartingPoint(*startingPoint);
  }

  ConvQueueT rootsa;
  rootsa.insert(rootsa.end(), global.begin(), global.end());
  rootsa.insert(rootsa.end(), local.begin(), local.end());
  AnnotationCount = rootsa.size();

  ConvQueueT vals;
  buildConversionQueueForRootValues(rootsa, vals);
  for (auto V : vals) {
    setMetadataOfValue(V->first, V->second);
  }
  removeAnnotationCalls(vals);

  SmallPtrSet<Function *, 10> callTrace;
  generateFunctionSpace(vals, global, callTrace);

  LLVM_DEBUG(printConversionQueue(vals));
  setFunctionArgsMetadata(m, vals);

  return PreservedAnalyses::all();
}


void TaffoInitializer::removeAnnotationCalls(ConvQueueT &q)
{
  for (auto i = q.begin(); i != q.end();) {
    Value *v = i->first;

    if (CallInst *anno = dyn_cast<CallInst>(v)) {
      if (anno->getCalledFunction()) {
        if (anno->getCalledFunction()->getName() == "llvm.var.annotation") {
          i = q.erase(i);
          anno->eraseFromParent();
          continue;
        }
      }
    }

    // TODO: remove global annotations

    i++;
  }
}


void TaffoInitializer::setMetadataOfValue(Value *v, ValueInfo &vi)
{
  std::shared_ptr<mdutils::MDInfo> md = vi.metadata;

  if (vi.bufferID) {
    mdutils::MetadataManager::setBufferIDMetadata(v, *(vi.bufferID));
  }
  if (isa<Instruction>(v) || isa<GlobalObject>(v)) {
    mdutils::MetadataManager::setInputInfoInitWeightMetadata(v, vi.fixpTypeRootDistance);
  }

  if (Instruction *inst = dyn_cast<Instruction>(v)) {
    if (vi.target.hasValue())
      mdutils::MetadataManager::setTargetMetadata(*inst, vi.target.getValue());

    if (auto *ii = dyn_cast<mdutils::InputInfo>(md.get())) {
      if (inst->getMetadata(OMP_DISABLED_METADATA)) {
        LLVM_DEBUG(dbgs() << "Blocking conversion for shared variable" << *inst << "\n");
        ii->IEnableConversion = false;
      }
      mdutils::MetadataManager::setInputInfoMetadata(*inst, *ii);
    } else if (mdutils::StructInfo *si = dyn_cast<mdutils::StructInfo>(md.get())) {
      mdutils::MetadataManager::setStructInfoMetadata(*inst, *si);
    }
  } else if (GlobalObject *con = dyn_cast<GlobalObject>(v)) {
    if (vi.target.hasValue())
      mdutils::MetadataManager::setTargetMetadata(*con, vi.target.getValue());

    if (mdutils::InputInfo *ii = dyn_cast<mdutils::InputInfo>(md.get())) {
      mdutils::MetadataManager::setInputInfoMetadata(*con, *ii);
    } else if (mdutils::StructInfo *si = dyn_cast<mdutils::StructInfo>(md.get())) {
      mdutils::MetadataManager::setStructInfoMetadata(*con, *si);
    }
  }
}


void TaffoInitializer::setFunctionArgsMetadata(Module &m, ConvQueueT &Q)
{
  std::vector<mdutils::MDInfo *> iiPVec;
  std::vector<int> wPVec;
  for (Function &f : m.functions()) {
    LLVM_DEBUG(dbgs() << "Processing function " << f.getName() << "\n");
    iiPVec.reserve(f.arg_size());
    wPVec.reserve(f.arg_size());

    for (Argument &a : f.args()) {
      LLVM_DEBUG(dbgs() << "Processing arg " << a << "\n");
      mdutils::MDInfo *ii = nullptr;
      int weight = -1;
      if (Q.count(&a)) {
        LLVM_DEBUG(dbgs() << "Info found.\n");
        ValueInfo &vi = Q[&a];
        if (vi.bufferID) {
          mdutils::MetadataManager::setBufferIDMetadata(&a, *(vi.bufferID));
        }
        ii = vi.metadata.get();
        weight = vi.fixpTypeRootDistance;
      }
      iiPVec.push_back(ii);
      wPVec.push_back(weight);
    }

    mdutils::MetadataManager::setArgumentInputInfoMetadata(f, iiPVec);
    mdutils::MetadataManager::setInputInfoInitWeightMetadata(&f, wPVec);

    iiPVec.clear();
    wPVec.clear();
  }
}


void TaffoInitializer::buildConversionQueueForRootValues(
    const ConvQueueT &val,
    ConvQueueT &queue)
{
  LLVM_DEBUG(dbgs() << "***** begin " << __PRETTY_FUNCTION__ << "\n"
                    << "Initial ");

  queue.insert(queue.begin(), val.begin(), val.end());
  LLVM_DEBUG(printConversionQueue(queue));

  SmallPtrSet<Value *, 8U> visited;
  size_t prevQueueSize = 0;
  while (prevQueueSize < queue.size()) {
    LLVM_DEBUG(dbgs() << "***** buildConversionQueueForRootValues iter " << prevQueueSize << " < " << queue.size() << "\n";);
    prevQueueSize = queue.size();

    auto next = queue.begin();
    while (next != queue.end()) {
      Value *v = next->first;
      visited.insert(v);

      LLVM_DEBUG(dbgs() << "[V] " << *v);
      if (Instruction *i = dyn_cast<Instruction>(v))
        LLVM_DEBUG(dbgs() << "[ " << i->getFunction()->getName() << "]\n");
      else
        LLVM_DEBUG(dbgs() << "\n");
      LLVM_DEBUG(dbgs() << "    distance = " << next->second.fixpTypeRootDistance << "\n");

      for (auto *u : v->users()) {
        /* ignore u if it is the global annotation array */
        if (GlobalObject *ugo = dyn_cast<GlobalObject>(u)) {
          if (ugo->hasSection() && ugo->getSection() == "llvm.metadata")
            continue;
        }

        if (isa<PHINode>(u) && visited.count(u)) {
          continue;
        }

        /* Insert u at the end of the queue.
         * If u exists already in the queue, *move* it to the end instead. */
        auto UI = queue.find(u);
        ValueInfo UVInfo;
        if (UI != queue.end()) {
          UVInfo = UI->second;
          queue.erase(UI);
        }
        UI = queue.push_back(u, std::move(UVInfo)).first;
        LLVM_DEBUG(dbgs() << "[U] " << *u);
        if (Instruction *i = dyn_cast<Instruction>(u))
          LLVM_DEBUG(dbgs() << "[ " << i->getFunction()->getName() << "]\n");
        else
          LLVM_DEBUG(dbgs() << "\n");

        unsigned int vdepth = std::min(next->second.backtrackingDepthLeft, next->second.backtrackingDepthLeft - 1);
        if (vdepth < 2 && isa<StoreInst>(u)) {
          StoreInst *store = dyn_cast<StoreInst>(u);
          Value *valOp = store->getValueOperand();
          Type *valueType = valOp->getType();
          if (isa<BitCastInst>(valOp) && valueType->isPointerTy() && valueType->getPointerElementType()->isFloatingPointTy()) {
            LLVM_DEBUG(dbgs() << "MALLOC'D POINTER HACK\n");
            vdepth = 2;
          } else if (valueType->isFloatingPointTy()) {
            LLVM_DEBUG(dbgs() << "Backtracking to stored value def instr\n");
            vdepth = 1;
          }
        }
        if (vdepth > 0) {
          unsigned int udepth = UI->second.backtrackingDepthLeft;
          UI->second.backtrackingDepthLeft = std::max(vdepth, udepth);
        }
        createInfoOfUser(v, next->second, u, UI->second);
      }
      ++next;
    }

    for (next = queue.end(); next != queue.begin();) {
      Value *v = (--next)->first;
      unsigned int mydepth = next->second.backtrackingDepthLeft;
      if (mydepth == 0)
        continue;

      Instruction *inst = dyn_cast<Instruction>(v);
      if (!inst)
        continue;

      LLVM_DEBUG(dbgs() << "BACKTRACK " << *v << ", depth left = " << mydepth << "\n");

      int OpIdx = -1;
      for (Value *u : inst->operands()) {
        OpIdx++;
        if (!isa<User>(u) && !isa<Argument>(u)) {
          LLVM_DEBUG(dbgs() << " - " << u->getNameOrAsOperand() << " not a User or an Argument\n");
          continue;
        }
        if (isa<StoreInst>(inst) && OpIdx == 1 && mydepth == 1) {
          LLVM_DEBUG(dbgs() << " - " << u->getNameOrAsOperand() << " is the pointer argument of a store and backtracking depth left is 1, ignoring\n");
          continue;
        }
        if (isa<Function>(u) || isa<BlockAddress>(u)) {
          LLVM_DEBUG(dbgs() << " - " << u->getNameOrAsOperand() << " is a function/block address\n");
          continue;
        }
        LLVM_DEBUG(dbgs() << " - " << *u);

        if (!isFloatType(u->getType())) {
          LLVM_DEBUG(dbgs() << " not a float\n");
          continue;
        }

        bool alreadyIn = false;
        ValueInfo VIU;
        VIU.backtrackingDepthLeft = std::min(mydepth, mydepth - 1);
        auto UI = queue.find(u);
        if (UI != queue.end()) {
          if (UI < next)
            alreadyIn = true;
          else
            UI = queue.erase(UI);
        }
        if (!alreadyIn) {
          LLVM_DEBUG(dbgs() << "  enqueued\n");
          next = UI = queue.insert(next, u, std::move(VIU)).first;
          ++next;
          createInfoOfUser(v, next->second, u, UI->second);
        } else {
          LLVM_DEBUG(dbgs() << " already in, not updating info\n");
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "***** end " << __PRETTY_FUNCTION__ << "\n");
}


void TaffoInitializer::createInfoOfUser(Value *used, const ValueInfo &vinfo, Value *user, ValueInfo &uinfo)
{
  /* Copy metadata from the closest instruction to a root */
  LLVM_DEBUG(dbgs() << "root distances: " << uinfo.fixpTypeRootDistance << " > " << vinfo.fixpTypeRootDistance << " + 1\n");
  if (!(uinfo.fixpTypeRootDistance <= std::max(vinfo.fixpTypeRootDistance, vinfo.fixpTypeRootDistance + 1))) {
    /* Do not copy metadata in case of type conversions from struct to
     * non-struct and vice-versa.
     * We could check the instruction type and copy the correct type
     * contained in the struct type or create a struct type with the
     * correct type in the correct place, but is'a huge mess */
    Type *usedt = fullyUnwrapPointerOrArrayType(used->getType());
    Type *usert = fullyUnwrapPointerOrArrayType(user->getType());
    LLVM_DEBUG(dbgs() << "usedt = " << *usedt << ", vinfo metadata = " << (vinfo.metadata ? vinfo.metadata->toString() : "(null)") << "\n");
    LLVM_DEBUG(dbgs() << "usert = " << *usert << ", uinfo metadata = " << (uinfo.metadata ? uinfo.metadata->toString() : "(null)") << "\n");
    bool copyok = (usedt == usert);
    copyok |= (!usedt->isStructTy() && !usert->isStructTy()) || isa<StoreInst>(user);
    if (isa<GetElementPtrInst>(user) && used != dyn_cast<GetElementPtrInst>(user)->getPointerOperand())
      copyok = false;
    if (copyok) {
      LLVM_DEBUG(dbgs() << "createInfoOfUser copied MD from vinfo (" << *used << ") " << vinfo.metadata->toString() << "\n");
      uinfo.metadata.reset(vinfo.metadata->clone());
    } else {
      LLVM_DEBUG(dbgs() << "createInfoOfUser created MD from uinfo because usedt != usert\n");
      uinfo.metadata = mdutils::StructInfo::constructFromLLVMType(usert);
      if (uinfo.metadata.get() == nullptr) {
        uinfo.metadata.reset(new mdutils::InputInfo(nullptr, nullptr, nullptr, true));
      }
    }

    uinfo.target = vinfo.target;
    uinfo.fixpTypeRootDistance = std::max(vinfo.fixpTypeRootDistance, vinfo.fixpTypeRootDistance + 1);
    LLVM_DEBUG(dbgs() << "[" << *user << "] update fixpTypeRootDistance=" << uinfo.fixpTypeRootDistance << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "[" << *user << "] not updated fixpTypeRootDistance=" << uinfo.fixpTypeRootDistance << "\n");
  }

  /* The conversion enabling flag shall be true if at least one of the parents
   * of the children has it enabled */
  mdutils::InputInfo *iiu = dyn_cast_or_null<mdutils::InputInfo>(uinfo.metadata.get());
  mdutils::InputInfo *iiv = dyn_cast_or_null<mdutils::InputInfo>(vinfo.metadata.get());
  if (iiu && iiv && iiv->IEnableConversion) {
    iiu->IEnableConversion = true;
  }

  // Fix metadata if this is a GetElementPtrInst
  if (std::shared_ptr<mdutils::MDInfo> gepi_mdi =
          extractGEPIMetadata(user, used, uinfo.metadata, vinfo.metadata)) {
    uinfo.metadata = gepi_mdi;
    uinfo.bufferID = vinfo.bufferID;
  }
  // Copy BufferID across bitcasts
  if (isa<BitCastInst>(user)) {
    uinfo.bufferID = vinfo.bufferID;
  }
  // Copy BufferID across loads
  if (isa<LoadInst>(user)) {
    uinfo.bufferID = vinfo.bufferID;
  }
}

std::shared_ptr<mdutils::MDInfo>
TaffoInitializer::extractGEPIMetadata(const llvm::Value *user,
                                      const llvm::Value *used,
                                      std::shared_ptr<mdutils::MDInfo> user_mdi,
                                      std::shared_ptr<mdutils::MDInfo> used_mdi)
{
  using namespace mdutils;
  if (!used_mdi)
    return nullptr;
  assert(user && used);
  const GetElementPtrInst *gepi = dyn_cast<GetElementPtrInst>(user);
  if (!gepi)
    return nullptr;

  if (gepi->getPointerOperand() != used) {
    /* if the used value is not the pointer, then it must be one of the
     * indices; keep everything as is */
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "[extractGEPIMetadata] begin\n");
  LLVM_DEBUG(dbgs() << "Initial GEP object metadata is " << used_mdi->toString() << "\n");

  Type *source_element_type = gepi->getSourceElementType();
  for (auto idx_it = gepi->idx_begin() + 1; // skip first index
       idx_it != gepi->idx_end(); ++idx_it) {
    LLVM_DEBUG(dbgs() << "[extractGEPIMetadata] source_element_type=" << *source_element_type << "\n");
    if (isa<llvm::ArrayType>(source_element_type) || isa<llvm::VectorType>(source_element_type))
      continue;

    if (const llvm::ConstantInt *int_i = dyn_cast<llvm::ConstantInt>(*idx_it)) {
      int n = static_cast<int>(int_i->getSExtValue());
      used_mdi = cast<StructInfo>(used_mdi.get())->getField(n);
      source_element_type =
          cast<StructType>(source_element_type)->getTypeAtIndex(n);
    } else {
      LLVM_DEBUG(dbgs() << "[extractGEPIMetadata] fail, non-const index encountered\n");
      return nullptr;
    }
  }
  if (used_mdi)
    LLVM_DEBUG(dbgs() << "[extractGEPIMetadata] end, used_mdi=" << used_mdi->toString() << "\n");
  else
    LLVM_DEBUG(dbgs() << "[extractGEPIMetadata] end, used_mdi=NULL\n");
  return (used_mdi)
             ? std::shared_ptr<mdutils::MDInfo>(used_mdi.get()->clone())
             : nullptr;
}


void TaffoInitializer::generateFunctionSpace(ConvQueueT &vals,
                                             ConvQueueT &global, SmallPtrSet<Function *, 10> &callTrace)
{
  LLVM_DEBUG(dbgs() << "***** begin " << __PRETTY_FUNCTION__ << "\n");

  for (auto VVI : vals) {
    Value *v = VVI->first;
    if (!(isa<CallInst>(v) || isa<InvokeInst>(v)))
      continue;
    CallBase *call = dyn_cast<CallBase>(v);

    Function *oldF = call->getCalledFunction();
    if (!oldF) {
      LLVM_DEBUG(dbgs() << "found bitcasted funcptr in " << *v << ", skipping\n");
      continue;
    }
    if (isSpecialFunction(oldF))
      continue;
    if (ManualFunctionCloning) {
      if (enabledFunctions.count(oldF) == 0) {
        LLVM_DEBUG(dbgs() << "skipped cloning of function from call " << *v << ": function disabled\n");
        continue;
      }
    }

    std::vector<llvm::Value *> newVals;

    Function *newF = createFunctionAndQueue(call, vals, global, newVals);
    call->setCalledFunction(newF);
    enabledFunctions.insert(newF);

    // Attach metadata
    MDNode *newFRef = MDNode::get(call->getContext(), ValueAsMetadata::get(newF));
    MDNode *oldFRef = MDNode::get(call->getContext(), ValueAsMetadata::get(oldF));

    call->setMetadata(ORIGINAL_FUN_METADATA, oldFRef);
    if (MDNode *cloned = oldF->getMetadata(CLONED_FUN_METADATA)) {
      cloned = cloned->concatenate(cloned, newFRef);
      oldF->setMetadata(CLONED_FUN_METADATA, cloned);
    } else {
      oldF->setMetadata(CLONED_FUN_METADATA, newFRef);
    }
    newF->setMetadata(CLONED_FUN_METADATA, NULL);
    newF->setMetadata(SOURCE_FUN_METADATA, oldFRef);

    mdutils::MetadataManager &mm = mdutils::MetadataManager::getMetadataManager();
    for (auto v : newVals) {
      Instruction *i = dyn_cast<Instruction>(v);
      if (!i || !mm.retrieveInputInfo(*i))
        setMetadataOfValue(v, vals[v]);
    }

    /* Reconstruct the value info for the values which are in the top-level
     * conversion queue and in the oldF
     * Allows us to properly process call functions */
    // TODO: REWRITE USING THE VALUE MAP RETURNED BY CloneFunctionInto
    for (BasicBlock &bb : *newF) {
      for (Instruction &i : bb) {
        if (mdutils::MDInfo *mdi = mm.retrieveMDInfo(&i)) {
          ValueInfo &vi = vals.insert(vals.end(), &i, ValueInfo()).first->second;
          vi.metadata.reset(mdi->clone());
          int weight = mm.retrieveInputInfoInitWeightMetadata(&i);
          if (weight >= 0)
            vi.fixpTypeRootDistance = weight;
          vals.push_back(&i, vi);
          LLVM_DEBUG(dbgs() << "  enqueued & rebuilt valueInfo of " << i << " in " << newF->getName() << "\n");
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "***** end " << __PRETTY_FUNCTION__ << "\n");
}


Function *TaffoInitializer::createFunctionAndQueue(CallBase *call, ConvQueueT &vals, ConvQueueT &global, std::vector<llvm::Value *> &convQueue)
{
  LLVM_DEBUG(dbgs() << "***** begin " << __PRETTY_FUNCTION__ << "\n");

  /* vals: conversion queue of caller
   * global: global values to copy in all converison queues
   * convQueue: output conversion queue of this function */

  Function *oldF = call->getCalledFunction();
  Function *newF = Function::Create(
      oldF->getFunctionType(), oldF->getLinkage(),
      oldF->getName(), oldF->getParent());

  ValueToValueMapTy mapArgs; // Create Val2Val mapping and clone function
  Function::arg_iterator newArgumentI = newF->arg_begin();
  Function::arg_iterator oldArgumentI = oldF->arg_begin();
  for (; oldArgumentI != oldF->arg_end(); oldArgumentI++, newArgumentI++) {
    newArgumentI->setName(oldArgumentI->getName());
    mapArgs.insert(std::make_pair(oldArgumentI, newArgumentI));
  }
  SmallVector<ReturnInst *, 100> returns;
  CloneFunctionInto(newF, oldF, mapArgs, CloneFunctionChangeType::GlobalChanges, returns);
  if (!OpenCLKernelMode)
    newF->setLinkage(GlobalVariable::LinkageTypes::InternalLinkage);
  FunctionCloned++;

  ConvQueueT roots;
  oldArgumentI = oldF->arg_begin();
  newArgumentI = newF->arg_begin();
  LLVM_DEBUG(dbgs() << "Create function from " << oldF->getName() << " to " << newF->getName() << "\n");
  LLVM_DEBUG(dbgs() << "  CallBase instr " << *call << " [" << call->getFunction()->getName() << "]\n");
  for (int i = 0; oldArgumentI != oldF->arg_end(); oldArgumentI++, newArgumentI++, i++) {
    auto user_begin = newArgumentI->user_begin();
    if (user_begin == newArgumentI->user_end()) {
      LLVM_DEBUG(dbgs() << "  Arg nr. " << i << " skipped, value has no uses\n");
      continue;
    }



    Value *callOperand = call->getOperand(i);
    Value *allocaOfArgument = user_begin->getOperand(1);


    for (; user_begin != newArgumentI->user_end(); ++user_begin) {
      if (llvm::StoreInst *store = llvm::dyn_cast<StoreInst>(*user_begin)) {
        allocaOfArgument = store->getOperand(1);
      }
    }

    if (!isa<AllocaInst>(allocaOfArgument))
      allocaOfArgument = nullptr;

    if (!vals.count(callOperand)) {
      LLVM_DEBUG(dbgs() << "  Arg nr. " << i << " skipped, callOperand has no valueInfo\n");
      continue;
    }

    ValueInfo &callVi = vals[callOperand];

    ValueInfo &argumentVi = vals.insert(vals.end(), newArgumentI, ValueInfo()).first->second;
    // Mark the argument itself (set it as a new root as well in VRA-less mode)
    argumentVi.metadata.reset(callVi.metadata->clone());
    argumentVi.fixpTypeRootDistance = std::max(callVi.fixpTypeRootDistance, callVi.fixpTypeRootDistance + 1);
    if (!allocaOfArgument) {
      roots.push_back(newArgumentI, argumentVi);
    }

    if (allocaOfArgument) {
      ValueInfo &allocaVi = vals.insert(vals.end(), allocaOfArgument, ValueInfo()).first->second;
      // Mark the alloca used for the argument (in O0 opt lvl)
      // let it be a root in VRA-less mode
      allocaVi.metadata.reset(callVi.metadata->clone());
      allocaVi.fixpTypeRootDistance = std::max(callVi.fixpTypeRootDistance, callVi.fixpTypeRootDistance + 2);
      roots.push_back(allocaOfArgument, allocaVi);
    }

    // Propagate BufferID
    argumentVi.bufferID = callVi.bufferID;

    LLVM_DEBUG(dbgs() << "  Arg nr. " << i << " processed\n");
    LLVM_DEBUG(dbgs() << "    md = " << argumentVi.metadata->toString() << "\n");
    if (allocaOfArgument)
      LLVM_DEBUG(dbgs() << "    enqueued alloca of argument " << *allocaOfArgument << "\n");
  }

  ConvQueueT tmpVals;
  roots.insert(roots.begin(), global.begin(), global.end());
  ConvQueueT localFix;
  readLocalAnnotations(*newF, localFix);
  roots.insert(roots.begin(), localFix.begin(), localFix.end());
  buildConversionQueueForRootValues(roots, tmpVals);
  for (auto val : tmpVals) {
    if (Instruction *inst = dyn_cast<Instruction>(val.first)) {
      if (inst->getFunction() == newF) {
        vals.push_back(val);
        convQueue.push_back(val.first);
        LLVM_DEBUG(dbgs() << "  enqueued " << *inst << " in " << newF->getName() << "\n");
      }
    }
  }

  LLVM_DEBUG(dbgs() << "***** end " << __PRETTY_FUNCTION__ << "\n");
  return newF;
}


void TaffoInitializer::printConversionQueue(ConvQueueT &vals)
{
  if (vals.size() < 1000) {
    dbgs() << "conversion queue:\n";
    for (auto val : vals) {
      dbgs() << "bt=" << val.second.backtrackingDepthLeft << " ";
      dbgs() << "md=" << val.second.metadata->toString() << " ";
      if (Instruction *I = dyn_cast<Instruction>(val.first))
        dbgs() << "fun=" << I->getFunction()->getName() << " ";
      dbgs() << *(val.first) << "\n";
    }
    dbgs() << "\n";
  } else {
    dbgs() << "not printing the conversion queue because it exceeds 1000 items";
  }
}
