#include "MetadataManager.hpp"

#include "SerializationUtils.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/InlineAsm.h>

#define DEBUG_TYPE "TaffoUtils"
#define TAFFO_GLOBAL_ID_LIST_METADATA "taffo.globalIds"
#define TAFFO_ID_METADATA "taffo.id"
#define TAFFO_ARGUMENTS_ID_METADATA "taffo.argIds"
#define TAFFO_LOOP_ID_METADATA "taffo.loopId"
#define TAFFO_TYPE_LIST_METADATA "taffo.types"
#define CUDA_KERNEL_METADATA "nvvm.annotations"

using namespace taffo;
using namespace llvm;

MetadataManager &MetadataManager::getInstance() {
  static MetadataManager Instance;
  return Instance;
}

void MetadataManager::setIdValueMapping(const BiMap<std::string, Value*> &idValueMapping, Module &m) {
  NamedMDNode *globalMappingMd = m.getOrInsertNamedMetadata(TAFFO_GLOBAL_ID_LIST_METADATA);
  // Clear old metadata
  globalMappingMd->clearOperands();
  for (Function &f : m)
    f.eraseMetadata(f.getContext().getMDKindID(TAFFO_ARGUMENTS_ID_METADATA));
  // Set new metadata
  for (const auto &[id, value] : idValueMapping) {
    LLVMContext &ctx = value->getContext();
    MDString *idMd = MDString::get(ctx, id);
    if (isa<GlobalValue>(value) || isa<Constant>(value)) {
      // Global values and constants' ids remain in the module-level metadata.
      ValueAsMetadata *valueMd = ValueAsMetadata::get(value);
      MDNode *entryMd = MDNode::get(ctx, {idMd, valueMd});
      globalMappingMd->addOperand(entryMd);
    }
    else if (auto *inst = dyn_cast<Instruction>(value)) {
      // Instructions' ids are attached to instructions directly
      inst->setMetadata(TAFFO_ID_METADATA, MDNode::get(ctx, idMd));
    }
    else if (auto *arg = dyn_cast<Argument>(value)) {
      // Function arguments' ids are attached their parent function
      Function *parentF = arg->getParent();
      unsigned argIndex = arg->getArgNo();
      auto *argIndexMD = ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx), argIndex));
      MDString *idMD = MDString::get(ctx, id);
      MDNode *newEntry = MDNode::get(ctx, {argIndexMD, idMD});
      // Retrieve any existing arguments metadata
      SmallVector<Metadata*, 4> elems;
      if (MDNode *existingMD = parentF->getMetadata(TAFFO_ARGUMENTS_ID_METADATA))
        for (const MDOperand &mdOperand : existingMD->operands())
          elems.push_back(mdOperand);
      elems.push_back(newEntry);
      parentF->setMetadata(TAFFO_ARGUMENTS_ID_METADATA, MDNode::get(ctx, elems));
    }
    else
      llvm_unreachable("Unrecognized local value");
  }
}

BiMap<std::string, Value*> MetadataManager::getIdValueMapping(Module &m) {
  BiMap<std::string, Value*> idValueMapping;
  // Global values:
  NamedMDNode *globalMappingMd = m.getNamedMetadata(TAFFO_GLOBAL_ID_LIST_METADATA);
  if (globalMappingMd) {
    for (MDNode *entryMd : globalMappingMd->operands()) {
      std::string id = cast<MDString>(entryMd->getOperand(0))->getString().str();
      ValueAsMetadata *valueMd = dyn_cast_or_null<ValueAsMetadata>(entryMd->getOperand(1));
      if (!valueMd) {
        LLVM_DEBUG(dbgs() << "Value with taffoId " << id << " probably deleted by dce pass: ignoring\n");
        continue;
      }
      Value *value = valueMd->getValue();
      idValueMapping[id] = value;
    }
  }
  // Local values:
  for (Function &f : m) {
    // Instructions:
    for (BasicBlock &bb : f) {
      for (Instruction &inst : bb) {
        MDNode *idMd = inst.getMetadata(TAFFO_ID_METADATA);
        if (idMd) {
          std::string id = cast<MDString>(idMd->getOperand(0))->getString().str();
          idValueMapping[id] = &inst;
        }
      }
    }
    // Function arguments:
    MDNode *argsMd = f.getMetadata(TAFFO_ARGUMENTS_ID_METADATA);
    if (argsMd) {
      for (const MDOperand &mdOperand : argsMd->operands()) {
        MDNode *entry = cast<MDNode>(mdOperand);
        unsigned argIndex = cast<ConstantInt>(cast<ConstantAsMetadata>(entry->getOperand(0))->getValue())->getZExtValue();
        std::string id = cast<MDString>(entry->getOperand(1))->getString().str();
        Argument &arg = *std::next(f.arg_begin(), argIndex);
        idValueMapping[id] = &arg;
      }
    }
  }
  return idValueMapping;
}

void MetadataManager::setIdLoopMapping(const BiMap<std::string, Loop*> &idLoopMapping, Module &m) {
  for (const auto &[id, loop] : idLoopMapping) {
    // Get a representative instruction from the loop header.
    Instruction &headerFirstInst = *loop->getHeader()->getFirstNonPHI();
    LLVMContext &ctx = headerFirstInst.getContext();
    MDString *idMd = MDString::get(ctx, id);
    headerFirstInst.setMetadata(TAFFO_ID_METADATA, MDNode::get(ctx, idMd));
  }
}

BiMap<std::string, Loop*> MetadataManager::getIdLoopMapping(Module &m) {
  BiMap<std::string, Loop*> idLoopMapping;
  for (Function &f : m) {
    for (BasicBlock &bb : f) {
      for (Instruction &inst : bb) {
        if (MDNode *loopMd = inst.getMetadata(TAFFO_LOOP_ID_METADATA)) {
          // Build loopInfo for the function to recover loops.
          DominatorTree dominatorTree(f);
          LoopInfo loopInfo(dominatorTree);
          std::string id = cast<MDString>(loopMd)->getString().str();
          BasicBlock *header = inst.getParent();
          Loop *loop = loopInfo.getLoopFor(header);
          idLoopMapping[id] = loop;
        }
      }
    }
  }
  return idLoopMapping;
}

void MetadataManager::setIdTypeMapping(const BiMap<std::string, Type*> &idTypeMapping, Module &m) {
  NamedMDNode *typesListMd = m.getOrInsertNamedMetadata(TAFFO_TYPE_LIST_METADATA);
  typesListMd->clearOperands();
  for (const auto &[id, type] : idTypeMapping) {
    LLVMContext &ctx = type->getContext();
    MDString *idMd = MDString::get(ctx, id);
    Metadata *typeMd;
    if (type->isVoidTy())
      typeMd = MDString::get(ctx, "void");
    else
      typeMd = ConstantAsMetadata::get(Constant::getNullValue(type));
    MDNode *entryMd = MDNode::get(ctx, {idMd, typeMd});
    typesListMd->addOperand(entryMd);
  }
}

BiMap<std::string, Type*> MetadataManager::getIdTypeMapping(Module &m) {
  BiMap<std::string, Type*> idTypeMapping;
  NamedMDNode *typesListMd = m.getNamedMetadata(TAFFO_TYPE_LIST_METADATA);
  for (MDNode *entryMd : typesListMd->operands()) {
    std::string id = cast<MDString>(entryMd->getOperand(0))->getString().str();
    if (auto *nullConstMd = dyn_cast<ConstantAsMetadata>(entryMd->getOperand(1)))
      idTypeMapping[id] = nullConstMd->getType();
    else {
      auto *stringMd = cast<MDString>(entryMd->getOperand(1));
      if (stringMd->getString() == "void")
        idTypeMapping[id] = Type::getVoidTy(entryMd->getContext());
      else
        llvm_unreachable("Unknown type identifier");
    }
  }
  return idTypeMapping;
}

void MetadataManager::getCudaKernels(Module &m, SmallVectorImpl<Function*> kernels) {
  NamedMDNode *cudaMd = m.getNamedMetadata(CUDA_KERNEL_METADATA);
  if (!cudaMd)
    return;
  for (MDNode *entry : cudaMd->operands()) {
    if (entry->getNumOperands() < 2)
      continue;
    MDString *kindId = dyn_cast<MDString>(entry->getOperand(1));
    if (!kindId || kindId->getString() != "kernel")
      continue;
    Function *kernelF = mdconst::dyn_extract_or_null<Function>(entry->getOperand(0));
    if (!kernelF)
      continue;
    kernels.push_back(kernelF);
  }
}
