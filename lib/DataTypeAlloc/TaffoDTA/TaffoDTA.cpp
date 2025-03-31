#include "TaffoDTA.h"

#include "Logger.hpp"
#include "TaffoInfo/TaffoInfo.hpp"
#include "DTAConfig.h"
#include "BufferIDFiles.h"
#include "PtrCasts.hpp"
#ifdef TAFFO_BUILD_ILP_DTA
#include "ILP/MetricBase.h"
#include "ILP/Optimizer.h"
#endif // TAFFO_BUILD_ILP_DTA

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/AbstractCallSite.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Debug.h>
#include <llvm/Analysis/ScalarEvolution.h>

using namespace llvm;
using namespace taffo;
using namespace tuner;

#define DEBUG_TYPE "taffo-dta"

STATISTIC(FixCast, "Number of fixed point format cast");

PreservedAnalyses TaffoTuner::run(Module &m, ModuleAnalysisManager &AM) {
  LLVM_DEBUG(Logger::getInstance().logln("[DataTypeAllocationPass]", raw_ostream::Colors::MAGENTA));
  TaffoInfo::getInstance().initializeFromFile("taffo_info_vra.json", m);
  
  MAM = &AM;

  std::vector<Value *> vals;
  SmallPtrSet<Value *, 8U> valset;
  retrieveAllMetadata(m, vals, valset);

#ifdef TAFFO_BUILD_ILP_DTA
  if (MixedMode) {
    LLVM_DEBUG(dbgs() << "Model " << CostModelFilename << "\n");
    LLVM_DEBUG(dbgs() << "Inst " << InstructionSet << "\n");
    buildModelAndOptimze(m, vals, valset);
  } else {
    mergeFixFormat(vals, valset);
  }
#else
  mergeFixFormat(vals, valset);
#endif

  mergeBufferIDSets();

  std::vector<Function *> toDel;
  toDel = collapseFunction(m);

  LLVM_DEBUG(dbgs() << "attaching metadata\n");
  attachFPMetaData(vals);
  attachFunctionMetaData(m);

  for (Function *f : toDel) {
    f->eraseFromParent();
    TaffoInfo::getInstance().eraseValue(*f);
  }

  TaffoInfo::getInstance().dumpToFile("taffo_info_dta.json", m);
  LLVM_DEBUG(Logger::getInstance().logln("[End of DataTypeAllocationPass]", raw_ostream::Colors::MAGENTA));
  return PreservedAnalyses::all();
}

/**
 * Reads metadata for the program and DOES THE ACTUAL DATA TYPE ALLOCATION.
 * Yes you read that right.
 */
void TaffoTuner::retrieveAllMetadata(Module &m, std::vector<Value *> &vals,
                                     SmallPtrSetImpl<Value *> &valset) {
  LLVM_DEBUG(dbgs() << "**********************************************************\n");
  LLVM_DEBUG(dbgs() << __PRETTY_FUNCTION__ << " BEGIN\n");
  LLVM_DEBUG(dbgs() << "**********************************************************\n");

  LLVM_DEBUG(dbgs() << "=============>>>>  " << __FUNCTION__ << " GLOBALS  <<<<===============\n");
  for (GlobalObject &globObj : m.globals()) {
    std::shared_ptr<ValueInfo> valueInfo = TaffoInfo::getInstance().getValueInfo(globObj);
    if (processMetadataOfValue(&globObj, valueInfo)) {
      vals.push_back(&globObj);
      retrieveBufferID(&globObj);
    }
  }
  LLVM_DEBUG(dbgs() << "\n");

  for (Function &f : m.functions()) {
    if (f.isIntrinsic())
      continue;
    LLVM_DEBUG(dbgs() << "=============>>>>  " << __FUNCTION__ << " FUNCTION " << f.getNameOrAsOperand() << "  <<<<===============\n");

    for (Argument &arg : f.args()) {
      std::shared_ptr<ValueInfo> argInfo = TaffoInfo::getInstance().getValueInfo(arg);
      if (processMetadataOfValue(&arg, argInfo)) {
        vals.push_back(&arg);
        retrieveBufferID(&arg);
      }
    }

    for (Instruction &inst : instructions(f)) {
      std::shared_ptr<ValueInfo> valueInfo = TaffoInfo::getInstance().getValueInfo(inst);
      if (processMetadataOfValue(&inst, valueInfo)) {
        vals.push_back(&inst);
        retrieveBufferID(&inst);
      }
    }
    LLVM_DEBUG(dbgs() << "\n");
  }

  LLVM_DEBUG(dbgs() << "=============>>>>  SORTING QUEUE  <<<<===============\n");
  sortQueue(vals, valset);

  LLVM_DEBUG(dbgs() << "**********************************************************\n");
  LLVM_DEBUG(dbgs() << __PRETTY_FUNCTION__ << " END\n");
  LLVM_DEBUG(dbgs() << "**********************************************************\n");
}

/**
 * Reads metadata for a value and DOES THE ACTUAL DATA TYPE ALLOCATION.
 * Yes you read that right.
 */
void TaffoTuner::retrieveBufferID(Value *V) {
  LLVM_DEBUG(dbgs() << "Looking up buffer id of " << *V << "\n");
  auto MaybeBID = TaffoInfo::getInstance().getValueInfo(*V)->getBufferId();
  if (MaybeBID.has_value()) {
    std::string Tag = *MaybeBID;
    auto& Set = bufferIDSets[Tag];
    Set.insert(V);
    LLVM_DEBUG(dbgs() << "Found buffer ID '" << Tag << "' for " << *V << "\n");
    if (hasTunerInfo(V))
      getTunerInfo(V)->bufferID = Tag;
  } else {
    LLVM_DEBUG(dbgs() << "No buffer ID for " << *V << "\n");
  }
}

bool TaffoTuner::processMetadataOfValue(Value *v, const std::shared_ptr<ValueInfo> &valueInfo) {
  LLVM_DEBUG(dbgs() << "\n" << __FUNCTION__ << " v=" << *v << " valueInfo=" << (valueInfo ? valueInfo->toString() : "(null)") << "\n");
  if (!valueInfo) {
    LLVM_DEBUG(dbgs() << "no metadata... bailing out!\n");
    return false;
  }
  std::shared_ptr<ValueInfo> newValueInfo = valueInfo->clone();

  if (v->getType()->isVoidTy()) {
    LLVM_DEBUG(dbgs() << "[Info] Value " << *v << " has void type, leaving metadata unchanged\n");
    getTunerInfo(v)->metadata = newValueInfo;
    return true;
  }

  /* HACK to set the enabled status on phis which compensates for a bug in vra.
   * Affects axbench/sobel. */
  bool forceEnableConv = false;
  if (isa<PHINode>(v) && !conversionDisabled(v) && isa<ScalarInfo>(newValueInfo.get())) {
    forceEnableConv = true;
  }

  bool skippedAll = true;
  Type *fuwt = getUnwrappedType(v);
  SmallVector<std::pair<std::shared_ptr<ValueInfo>, Type*>, 8> queue({std::make_pair(newValueInfo, fuwt)});

  while (!queue.empty()) {
    std::pair<std::shared_ptr<ValueInfo>, Type*> elem = queue.pop_back_val();

    if (std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast<ScalarInfo>(elem.first)) {
      if (forceEnableConv)
        scalarInfo->conversionEnabled = true;

      // FIXME: hack to propagate itofp metadata
      if (MixedMode && (isa<UIToFPInst>(v) || isa<SIToFPInst>(v))) {
        LLVM_DEBUG(dbgs() << "FORCING CONVERSION OF A ITOFP!\n";);
        scalarInfo->conversionEnabled = true;
      }

      if (!elem.first->getUnwrappedType()->isFloatTy()) {
        LLVM_DEBUG(dbgs() << "[Info] Skipping a member of " << *v << " because not a float\n");
        continue;
      }

      // TODO: insert logic here to associate different types in a clever way
      if (associateFixFormat(scalarInfo, v)) {
        skippedAll = false;
      }

    } else if (std::shared_ptr<StructInfo> structInfo = dynamic_ptr_cast<StructInfo>(elem.first)) {
      if (!elem.second->isStructTy()) {
        LLVM_DEBUG(dbgs() << "[ERROR] found non conforming structinfo " << structInfo->toString() << " on value " << *v
                          << "\n");
        LLVM_DEBUG(dbgs() << "contained type " << *elem.second << " is not a struct type\n");
        LLVM_DEBUG(dbgs() << "The top-level MDInfo was " << valueInfo->toString() << "\n");
        llvm_unreachable("Non-conforming StructInfo.");
      }
      for (const std::shared_ptr<ValueInfo> &field : *structInfo)
        if (field)
          queue.push_back(std::make_pair(field, field->getUnwrappedType()));

    } else {
      llvm_unreachable("unknown mdinfo subclass");
    }
  }

  if (!skippedAll) {
    std::shared_ptr<TunerInfo> tunerInfo = getTunerInfo(v);
    tunerInfo->metadata = newValueInfo;
    LLVM_DEBUG(dbgs() << "associated metadata '" << newValueInfo->toString() << "' to value " << *v);
    if (auto *i = dyn_cast<Instruction>(v))
      LLVM_DEBUG(dbgs() << " (parent function = " << i->getFunction()->getName() << ")");
    LLVM_DEBUG(dbgs() << "\n");
    if (std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast<ScalarInfo>(newValueInfo))
      tunerInfo->initialType = scalarInfo->numericType;
  }
  return !skippedAll;
}


bool TaffoTuner::associateFixFormat(std::shared_ptr<ScalarInfo> &scalarInfo, Value *value) {
  if (!scalarInfo->isConversionEnabled()) {
    LLVM_DEBUG(dbgs() << "[Info] Skipping " << scalarInfo->toString() << ", conversion disabled\n");
    return false;
  }

  if (scalarInfo->numericType) {
    LLVM_DEBUG(dbgs() << "[Info] Type of " << scalarInfo->toString() << " already assigned\n");
    return true;
  }

  Range *rng = scalarInfo->range.get();
  if (rng == nullptr) {
    LLVM_DEBUG(dbgs() << "[Info] Skipping " << scalarInfo->toString() << ", no range\n");
    return false;
  }

  double greatest = std::max(std::abs(rng->Min), std::abs(rng->Max));
  auto *I = dyn_cast<Instruction>(value);
  if (I) {
    if (I->isBinaryOp() || I->isUnaryOp()) {
      std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast_or_null<ScalarInfo>(TaffoInfo::getInstance().getValueInfo(*I->getOperand(0U)));
      if (scalarInfo && scalarInfo->range)
        greatest = std::max(greatest, std::max(std::abs(scalarInfo->range->Max), std::abs(scalarInfo->range->Min)));
      else
        LLVM_DEBUG(dbgs() << "[Warning] No range metadata found on first arg of " << *I << "\n");
    }
    if (I->isBinaryOp()) {
      std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast_or_null<ScalarInfo>(TaffoInfo::getInstance().getValueInfo(*I->getOperand(1U)));
      if (scalarInfo && scalarInfo->range)
        greatest = std::max(greatest, std::max(std::abs(scalarInfo->range->Max), std::abs(scalarInfo->range->Min)));
      else
        LLVM_DEBUG(dbgs() << "[Warning] No range metadata found on second arg of " << *I << "\n");
    }
  }
  LLVM_DEBUG(dbgs() << "[Info] Maximum value involved in " << *value << " = " << greatest << "\n");

  if (!UseFloat.empty()) {
    FloatType::FloatStandard standard;
    if (UseFloat == "f16")
      standard = FloatType::Float_half;
    else if (UseFloat == "f32")
      standard = FloatType::Float_float;
    else if (UseFloat == "f64")
      standard = FloatType::Float_double;
    else if (UseFloat == "bf16")
      standard = FloatType::Float_bfloat;
    else {
      errs() << "[DTA] Invalid format " << UseFloat << " specified to the -usefloat argument.\n";
      abort();
    }
    //auto standard = static_cast<mdutils::FloatType::FloatStandard>(ForceFloat.getValue());

    auto res = std::make_shared<FloatType>(FloatType(standard, greatest));
    double maxRep = std::max(std::abs(res->getMaxValueBound().convertToDouble()), std::abs(res->getMinValueBound().convertToDouble()));
    LLVM_DEBUG(dbgs() << "[Info] Maximum value representable in " << res->toString() << " = " << maxRep << "\n");

    if (greatest >= maxRep) {
      LLVM_DEBUG(dbgs() << "[Info] CANNOT force conversion to float " << res->toString() << " because max value is not representable\n");
    } else {
      LLVM_DEBUG(dbgs() << "[Info] Forcing conversion to float " << res->toString() << "\n");
      scalarInfo->numericType = res;
      return true;
    }
  } else {
    FixedPointTypeGenError fpgerr;

    /* Testing maximum type for operands, not deciding type yet */
    fixedPointTypeFromRange(Range(0, greatest), &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
    if (fpgerr == FixedPointTypeGenError::NoError) {
      FixpType res = fixedPointTypeFromRange(*rng, &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
      if (fpgerr == FixedPointTypeGenError::NoError) {
        LLVM_DEBUG(dbgs() << "[Info] Converting to " << res.toString() << "\n");
        scalarInfo->numericType = res.clone();
        return true;
      }

      LLVM_DEBUG(dbgs() << "[Info] Error when generating fixed point type\n");
      switch (fpgerr) {
        case FixedPointTypeGenError::InvalidRange:
          LLVM_DEBUG(dbgs() << "[Info] Invalid range\n");
          break;
        case FixedPointTypeGenError::UnboundedRange:
          LLVM_DEBUG(dbgs() << "[Info] Unbounded range\n");
          break;
        case FixedPointTypeGenError::NotEnoughIntAndFracBits:
        case FixedPointTypeGenError::NotEnoughFracBits:
          LLVM_DEBUG(dbgs() << "[Info] Result not representable\n");
          break;
        default:
          LLVM_DEBUG(dbgs() << "[Info] error code unknown\n");
      }
    } else {
      LLVM_DEBUG(dbgs() << "[Info] The operands of " << *value << " are not representable as fixed point with specified constraints\n");
    }
  }

  /* We failed, try to keep original type */
  Type *Ty = getUnwrappedType(value);
  if (Ty->isFloatingPointTy()) {
    auto res = std::make_shared<FloatType>(FloatType(Ty->getTypeID(), greatest));
    scalarInfo->numericType = res;
    LLVM_DEBUG(dbgs() << "[Info] Keeping original type which was " << res->toString() << "\n");
    return true;
  }
  
  LLVM_DEBUG(dbgs() << "[Info] The original type was not floating point, skipping (fingers crossed!)\n");
  return false;
}


void TaffoTuner::sortQueue(std::vector<Value *> &vals,
                           SmallPtrSetImpl<Value *> &valset) {
  // Topological sort by means of a reversed DFS.
  enum VState {
    Visited,
    Visiting
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
          if (!hasTunerInfo(u)) {
            LLVM_DEBUG(dbgs() << "[WARNING] Found Value " << *u << " without range! (uses " << *c << ")\n");
            Type *utype = getUnwrappedType(u);
            Type *ctype = getUnwrappedType(c);
            if (!utype->isStructTy() && !ctype->isStructTy()) {
              std::shared_ptr<ScalarInfo> scalarInfo = static_ptr_cast<ScalarInfo>(getTunerInfo(c)->metadata->clone());
              scalarInfo->range.reset();
              std::shared_ptr<TunerInfo> viu = getTunerInfo(u);
              viu->metadata = scalarInfo;
              viu->initialType = scalarInfo->numericType;
            } else if (utype->isStructTy() && ctype->isStructTy() && ctype->canLosslesslyBitCastTo(utype)) {
              getTunerInfo(u)->metadata = getTunerInfo(c)->metadata->clone();
            } else {
              if (utype->isStructTy())
                getTunerInfo(u)->metadata = StructInfo::constructFromLLVMType(utype);
              else
                getTunerInfo(u)->metadata = std::make_shared<ScalarInfo>(getUnwrappedType(u));
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
    if (Argument *Arg = dyn_cast<Argument>(*i)) {
      LLVM_DEBUG(dbgs() << "Restoring consistency of argument " << *Arg << " of function " << Arg->getParent()->getNameOrAsOperand() << "\n");
      restoreTypesAcrossFunctionCall(Arg);
    }
  }
}

void TaffoTuner::mergeFixFormat(const std::vector<Value *> &vals,
                                const SmallPtrSetImpl<Value *> &valset) {
  if (DisableTypeMerging)
    return;

  assert(vals.size() == valset.size() && "They must contain the same elements.");
  bool merged = false;
  for (Value *v : vals) {
    for (Value *u : v->users()) {
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

bool TaffoTuner::mergeFixFormat(Value *v, Value *u) {
  std::shared_ptr<TunerInfo> valueTunerInfo = getTunerInfo(v);
  std::shared_ptr<TunerInfo> userTunerInfo = getTunerInfo(u);
  std::shared_ptr<ScalarInfo> valueInfo = dynamic_ptr_cast<ScalarInfo>(valueTunerInfo->metadata);
  std::shared_ptr<ScalarInfo> userInfo = dynamic_ptr_cast<ScalarInfo>(userTunerInfo->metadata);
  if (!valueInfo || !userInfo) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a struct\n");
    return false;
  }
  if (!valueInfo->numericType || !valueTunerInfo->initialType || !userInfo->numericType || !userTunerInfo->initialType) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u
                      << " because at least one does not change to a fixed point type\n");
    return false;
  }
  if (v->getType()->isPointerTy() || u->getType()->isPointerTy()) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a pointer\n");
    return false;
  }
  std::shared_ptr<FixpType> valueFixpType = dynamic_ptr_cast<FixpType>(valueTunerInfo->initialType);
  std::shared_ptr<FixpType> userFixpType = dynamic_ptr_cast<FixpType>(userTunerInfo->initialType);
  if (!valueFixpType || !userFixpType) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because one is not a FPType\n");
    return false;
  }
  if (valueFixpType != userFixpType) {
    if (isMergeable(valueFixpType, userFixpType)) {
      std::shared_ptr<FixpType> fp = merge(valueFixpType, userFixpType);
      if (!fp) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u
                          << " because resulting type is invalid\n");
        return false;
      }
      LLVM_DEBUG(dbgs() << "Merged fixp : \n"
                        << "\t" << *v << " fix type " << valueFixpType->toString() << "\n"
                        << "\t" << *u << " fix type " << userFixpType->toString() << "\n"
                        << "Final format " << fp->toString() << "\n";);

      valueInfo->numericType = fp->clone();
      userInfo->numericType = fp->clone();
      return true;
    } else {
      FixCast++;
    }
  }
  return false;
}

bool TaffoTuner::mergeFixFormatIterative(Value *v, Value *u) {
  std::shared_ptr<TunerInfo> viv = getTunerInfo(v);
  std::shared_ptr<TunerInfo> viu = getTunerInfo(u);
  std::shared_ptr<ScalarInfo> iiv = dynamic_ptr_cast<ScalarInfo>(viv->metadata);
  std::shared_ptr<ScalarInfo> iiu = dynamic_ptr_cast<ScalarInfo>(viu->metadata);
  if (!iiv || !iiu) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a struct\n");
    return false;
  }
  if (!iiv->numericType || !iiu->numericType) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u
                      << " because at least one does not change to a fixed point type\n");
    return false;
  }
  if (v->getType()->isPointerTy() || u->getType()->isPointerTy()) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because at least one is a pointer\n");
    return false;
  }
  std::shared_ptr<FixpType> fpv = dynamic_ptr_cast<FixpType>(iiv->numericType);
  std::shared_ptr<FixpType> fpu = dynamic_ptr_cast<FixpType>(iiu->numericType);
  if (!fpv || !fpu) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because one is not a FPType\n");
    return false;
  }
  if (*fpv != *fpu) {
    if (isMergeable(fpv, fpu)) {
      std::shared_ptr<FixpType> fp = merge(fpv, fpu);
      if (!fp) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because resulting type "
                          << fp->toString() << " is invalid\n");
        return false;
      }
      LLVM_DEBUG(dbgs() << "Merged fixp : \n"
                        << "\t" << *v << " fix type " << fpv->toString() << "\n"
                        << "\t" << *u << " fix type " << fpu->toString() << "\n"
                        << "Final format " << fp->toString() << "\n";);

      iiv->numericType = fp->clone();
      iiu->numericType = fp->clone();
      return true;
    } else {
      FixCast++;
    }
  }
  return false;
}

bool tuner::isMergeable(const std::shared_ptr<FixpType> &fpv, const std::shared_ptr<FixpType> &fpu) {
  return fpv->getWidth() == fpu->getWidth()
         && (std::abs(int(fpv->getPointPos()) - int(fpu->getPointPos())) + (fpv->isSigned() == fpu->isSigned() ? 0 : 1)) <= SimilarBits;
}

std::shared_ptr<FixpType> tuner::merge(const std::shared_ptr<FixpType> &fpv, const std::shared_ptr<FixpType> &fpu) {
  int sign_v = fpv->isSigned() ? 1 : 0;
  int int_v = int(fpv->getWidth()) - fpv->getPointPos() - sign_v;
  int sign_u = fpu->isSigned() ? 1 : 0;
  int int_u = int(fpu->getWidth()) - fpu->getPointPos() - sign_u;

  int sign_res = std::max(sign_u, sign_v);
  int int_res = std::max(int_u, int_v);
  int size_res = std::max(fpv->getWidth(), fpu->getWidth());
  int frac_res = size_res - int_res - sign_res;
  if (sign_res + int_res + frac_res != size_res || frac_res < 0)
    return nullptr; // Invalid format.
  else
    return std::make_shared<FixpType>(size_res, frac_res, sign_res);
}

std::shared_ptr<NumericType> tuner::merge(const std::shared_ptr<NumericType> &fpv, const std::shared_ptr<NumericType> &fpu) {
  if (isa<FixpType>(fpv.get()) && isa<FixpType>(fpu.get()))
    return merge(dynamic_ptr_cast<FixpType>(fpv), dynamic_ptr_cast<FixpType>(fpu));
  if (isa<FixpType>(fpv.get()) && isa<FloatType>(fpu.get()))
    return dynamic_ptr_cast<FloatType>(fpu)->clone();
  if (isa<FixpType>(fpu.get()) && isa<FloatType>(fpv.get()))
    return dynamic_ptr_cast<FloatType>(fpv)->clone();
  if (isa<FloatType>(fpu.get()) && isa<FloatType>(fpv.get())) {
    std::shared_ptr<FloatType> a = dynamic_ptr_cast<FloatType>(fpu);
    std::shared_ptr<FloatType> b = dynamic_ptr_cast<FloatType>(fpv);
    FloatType::FloatStandard maxStd = std::max(a->getStandard(), b->getStandard());
    double maxMax = std::max(a->getGreatestNumber(), b->getGreatestNumber());
    return std::make_shared<FloatType>(maxStd, maxMax);
  }
  llvm_unreachable("unknown numericType subclass");
}

void TaffoTuner::mergeBufferIDSets() {
  LLVM_DEBUG(dbgs() << "\n" << __PRETTY_FUNCTION__ << " BEGIN\n\n");
  BufferIDTypeMap InMap, OutMap;
  if (!BufferIDImport.empty()) {
    LLVM_DEBUG(dbgs() << "Importing Buffer ID sets from " << BufferIDImport << "\n\n");
    ReadBufferIDFile(BufferIDImport, InMap);
  }

  for (auto& Set: bufferIDSets) {
    LLVM_DEBUG(dbgs() << "Merging Buffer ID set " << Set.first << "\n");

    std::shared_ptr<NumericType> DestType;
    if (InMap.find(Set.first) != InMap.end()) {
      LLVM_DEBUG(dbgs() << "Set has type specified in file\n");
      DestType = InMap.at(Set.first)->clone();
    } else {
      for (auto *V: Set.second) {
        std::shared_ptr<TunerInfo> tunerInfo = getTunerInfo(V);
        std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast<ScalarInfo>(tunerInfo->metadata);
        if (!scalarInfo) {
          LLVM_DEBUG(dbgs() << "Metadata is null or struct, not handled, bailing out! Value='" << *V << "'\n");
          goto nextSet;
        }
        std::shared_ptr<NumericType> T = scalarInfo->numericType;
        if (T) {
          LLVM_DEBUG(dbgs() << "Type=" << T->toString() << " Value='" << *V << "'\n");
        } else {
          LLVM_DEBUG(dbgs() << "Type is null, not handled, bailing out! Value='" << *V << "'\n");
          continue;
        }
        
        if (!DestType) {
          DestType = T->clone();
        } else {
          DestType = merge(DestType, T);
        }
      }
    }
    LLVM_DEBUG(dbgs() << "Computed merged type: " << DestType->toString() << "\n");

    for (auto *V: Set.second) {
      std::shared_ptr<TunerInfo> tunerInfo = getTunerInfo(V);
      std::shared_ptr<ScalarInfo> scalarInfo = dynamic_ptr_cast<ScalarInfo>(tunerInfo->metadata);
      scalarInfo->numericType = DestType->clone();
      restoreTypesAcrossFunctionCall(V);
    }
    OutMap[Set.first] = DestType->clone();

nextSet:
    LLVM_DEBUG(dbgs() << "Merging Buffer ID set " << Set.first << " DONE\n\n");
  }

  if (!BufferIDExport.empty()) {
    LLVM_DEBUG(dbgs() << "Exporting Buffer ID sets to " << BufferIDExport << "\n\n");
    WriteBufferIDFile(BufferIDExport, OutMap);
  }

  LLVM_DEBUG(dbgs() << __PRETTY_FUNCTION__ << " END\n\n");
}


void TaffoTuner::restoreTypesAcrossFunctionCall(Value *v) {
  LLVM_DEBUG(dbgs() << "restoreTypesAcrossFunctionCall(" << *v << ")\n");
  if (!hasTunerInfo(v)) {
    LLVM_DEBUG(dbgs() << " --> skipping restoring types because value is not converted\n");
    return;
  }

  std::shared_ptr<ValueInfo> finalMd = getTunerInfo(v)->metadata;

  if (auto *arg = dyn_cast<Argument>(v)) {
    LLVM_DEBUG(dbgs() << "Is a function argument, propagating to calls\n");
    setTypesOnCallArgumentFromFunctionArgument(arg, finalMd);
  } else {
    LLVM_DEBUG(dbgs() << "Not a function argument, propagating to function arguments\n");
    setTypesOnFunctionArgumentFromCallArgument(v, finalMd);
  }
  
  LLVM_DEBUG(dbgs() << "restoreTypesAcrossFunctionCall ended\n");
}


void TaffoTuner::setTypesOnFunctionArgumentFromCallArgument(Value *v, std::shared_ptr<ValueInfo> finalMd) {
  for (Use &use : v->uses()) {
    User *user = use.getUser();
    auto *call = dyn_cast<CallBase>(user);
    if (call == nullptr)
      continue;
    LLVM_DEBUG(dbgs() << "restoreTypesAcrossFunctionCall: processing user " << *(user) << ")\n");

    auto *fun = dyn_cast<Function>(call->getCalledFunction());
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
    if (hasTunerInfo(arg)) {
      getTunerInfo(arg)->metadata = finalMd->clone();
      setTypesOnCallArgumentFromFunctionArgument(arg, finalMd);
      LLVM_DEBUG(dbgs() << " --> set new metadata, now checking uses of the argument... (hope there's no recursion!)\n");
      setTypesOnFunctionArgumentFromCallArgument(arg, finalMd);
    } else {
      LLVM_DEBUG(dbgs() << "Not looking good, formal arg #" << use.getOperandNo() << " (" << *arg << ") has no valueInfo, but actual argument does...\n");
    }
  }
}


void TaffoTuner::setTypesOnCallArgumentFromFunctionArgument(Argument *arg, std::shared_ptr<ValueInfo> finalMd) {
  Function *fun = arg->getParent();
  int n = arg->getArgNo();
  LLVM_DEBUG(dbgs() << " --> setting types to " << finalMd->toString() << " on call arguments from function "
                    << fun->getName() << " argument " << n << "\n");
  for (auto it = fun->user_begin(); it != fun->user_end(); it++) {
    if (isa<CallInst>(*it) || isa<InvokeInst>(*it)) {
      Value *callarg = it->getOperand(n);
      LLVM_DEBUG(dbgs() << " --> target " << *callarg << ", CallBase " << **it << "\n");

      if (!hasTunerInfo(callarg)) {
        if (!isa<Argument>(callarg)) {
          LLVM_DEBUG(dbgs() << " --> actual argument doesn't get converted; skipping\n");
          continue;
        } else {
          LLVM_DEBUG(dbgs() << " --> actual argument IS AN ARGUMENT ITSELF! not skipping even if it doesn't get converted\n");
        }
      }
      getTunerInfo(callarg)->metadata = finalMd->clone();
      if (auto *Arg = dyn_cast<Argument>(callarg)) {
        LLVM_DEBUG(dbgs() << " --> actual argument IS AN ARGUMENT ITSELF, recursing\n");
        setTypesOnCallArgumentFromFunctionArgument(Arg, finalMd);
      }
    }
  }
}

std::vector<Function *> TaffoTuner::collapseFunction(Module &m) {
  std::vector<Function *> toDel;
  for (Function &f : m.functions()) {
    if (ranges::find(toDel, &f) != toDel.end())
      continue;
    LLVM_DEBUG(dbgs() << "Analyzing original function " << f.getName() << "\n");

    SmallPtrSet<Function*, 2> taffoFunctions;
    TaffoInfo::getInstance().getTaffoFunctions(f, taffoFunctions);
    for (Function *cloneF : taffoFunctions) {
      LLVM_DEBUG(dbgs() << "\t Clone: " << *cloneF << "\n");
      if (cloneF->user_empty()) {
        LLVM_DEBUG(dbgs() << "\t Ignoring " << cloneF->getName()
            << " because it's not used anywhere\n");
      } else if (Function *eqFun = findEqFunction(cloneF, &f)) {
        LLVM_DEBUG(dbgs() << "\t Replace function " << cloneF->getName()
            << " with " << eqFun->getName() << "\n";);
        cloneF->replaceAllUsesWith(eqFun);
        toDel.push_back(cloneF);
      }
    }
  }
  return toDel;
}

bool compareTypesOfMDInfo(const std::shared_ptr<ValueInfo> &mdi1, const std::shared_ptr<ValueInfo> &mdi2) {
  if (mdi1->getKind() != mdi2->getKind())
    return false;

  if (isa<ScalarInfo>(mdi1.get())) {
    std::shared_ptr<ScalarInfo> ii1 = static_ptr_cast<ScalarInfo>(mdi1);
    std::shared_ptr<ScalarInfo> ii2 = static_ptr_cast<ScalarInfo>(mdi2);
    if (ii1->numericType && ii2->numericType) {
      return *ii1->numericType == *ii2->numericType;
    } else
      return false;

  } else if (isa<StructInfo>(mdi1.get())) {
    std::shared_ptr<StructInfo> si1 = static_ptr_cast<StructInfo>(mdi1);
    std::shared_ptr<StructInfo> si2 = static_ptr_cast<StructInfo>(mdi2);
    if (si1->numFields() == si2->numFields()) {
      int numFields = si1->numFields();
      for (int i = 0; i < numFields; i++) {
        std::shared_ptr<ValueInfo> p1 = si1->getField(i);
        std::shared_ptr<ValueInfo> p2 = si1->getField(i);
        if ((p1.get() == nullptr) != (p2.get() == nullptr))
          return false;
        if (p1.get() != nullptr) {
          if (!compareTypesOfMDInfo(p1, p2))
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
  std::vector<std::pair<int, std::shared_ptr<ValueInfo>>> fixSign;

  LLVM_DEBUG(dbgs() << "\t\t Search eq function for " << fun->getName()
      << " in " << origin->getName() << " pool\n";);

  if (getUnwrappedType(fun)->isFloatTy() && hasTunerInfo(*fun->user_begin())) {
    std::shared_ptr<ValueInfo> retval = getTunerInfo(*fun->user_begin())->metadata;
    if (retval) {
      fixSign.push_back(std::pair(-1, retval)); // ret value in signature
      LLVM_DEBUG(dbgs() << "\t\t Return type : "
          << getTunerInfo(*fun->user_begin())->metadata->toString() << "\n";);
    }
  }

  int i = 0;
  for (Argument &arg : fun->args()) {
    if (hasTunerInfo(&arg) && getTunerInfo(&arg)->metadata) {
      fixSign.push_back(std::pair(i, getTunerInfo(&arg)->metadata));
      LLVM_DEBUG(dbgs() << "\t\t Arg " << i << " type : "
          << getTunerInfo(&arg)->metadata->toString() << "\n";);
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
          if (!compareTypesOfMDInfo(fcheck->second, fthis->second))
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
  LLVM_DEBUG(dbgs() << "\t Function " << fun->getName() << " used\n";);
  return nullptr;
}

void TaffoTuner::attachFPMetaData(std::vector<Value *> &vals)
{
  for (Value *v : vals) {
    assert(info[v] && "Every value should have info");
    assert(getTunerInfo(v)->metadata.get() && "every value should have metadata");

    if (isa<Instruction>(v) || isa<GlobalObject>(v)) {
      TaffoInfo::getInstance().setValueInfo(*v, getTunerInfo(v)->metadata);
    } else {
      LLVM_DEBUG(dbgs() << "[WARNING] Cannot attach MetaData to " << *v << " (normal for function args)\n");
    }
  }
}

void TaffoTuner::attachFunctionMetaData(Module &m) {
  for (Function &f : m.functions()) {
    if (f.isIntrinsic())
      continue;

    for (Argument &arg : f.args())
      if (TaffoInfo::getInstance().getValueInfo(arg) && hasTunerInfo(&arg))
        TaffoInfo::getInstance().setValueInfo(arg, getTunerInfo(&arg)->metadata);
  }
}

#ifdef TAFFO_BUILD_ILP_DTA
void TaffoTuner::buildModelAndOptimze(Module &m, const vector<Value *> &vals,
                                      const SmallPtrSetImpl<Value *> &valset) {
  assert(vals.size() == valset.size() && "They must contain the same elements.");

  Optimizer optimizer(m, this, new MetricPerf(), CostModelFilename, CPUCosts::CostType::Performance);
  // Optimizer optimizer(m, this, new MetricPerf(),"", CPUCosts::CostType::Size);
  optimizer.initialize();

  LLVM_DEBUG(dbgs() << "\n============ GLOBALS ============\n");

  for (GlobalObject &globObj : m.globals()) {
    LLVM_DEBUG(globObj.print(dbgs()););
    LLVM_DEBUG(dbgs() << "     -having-     ");
    if (!hasTunerInfo(&globObj)) {
      LLVM_DEBUG(dbgs() << "No info available, skipping.");
    } else {
      LLVM_DEBUG(dbgs() << getTunerInfo(&globObj)->metadata->toString() << "\n");

      optimizer.handleGlobal(&globObj, getTunerInfo(&globObj));
    }
    LLVM_DEBUG(dbgs() << "\n\n";);
  }

  // FIXME: this is an hack to prevent multiple visit of the same function if it will be called somewhere from the program
  for (Function &f : m.functions()) {
    // Skip compiler provided functions
    if (f.isIntrinsic() || f.isDeclaration())
      continue;

    if (!f.isIntrinsic() && !f.empty() && f.getName().equals("main")) {
      LLVM_DEBUG(dbgs() << "========== GLOBAL ENTRY POINT main ==========";);

      optimizer.handleCallFromRoot(&f);
      break;
    }
  }

  // Looking for remaining functions
  for (Function &f : m.functions()) {
    // Skip compiler provided functions
    if (f.isIntrinsic()) {
      LLVM_DEBUG(dbgs() << "Skipping intrinsic function " << f.getName() << "\n";);
      continue;
    }

    // Skip empty functions
    if (f.empty()) {
      LLVM_DEBUG(dbgs() << "Skipping empty function " << f.getName() << "\n";);
      continue;
    }

    optimizer.handleCallFromRoot(&f);
  }

  bool result = optimizer.finish();
  assert(result && "Optimizer did not find a solution!");

  for (Value *v : vals) {
    LLVM_DEBUG(dbgs() << "Processing " << *v << "...\n");

    if (!valset.count(v)) {
      LLVM_DEBUG(dbgs() << "Not in the conversion queue! Skipping!\n\n";);
      continue;
    }

    std::shared_ptr<TunerInfo> viu = getTunerInfo(v);

    // Read from the model, search for the data type associated with that value and convert it!
    auto fp = optimizer.getAssociatedMetadata(v);
    if (!fp) {
      LLVM_DEBUG(dbgs() << "Invalid datatype returned!\n";);
      continue;
    }
    LLVM_DEBUG(dbgs() << "Datatype: " << fp->toString() << "\n");

    // Write the datatype
    bool result = overwriteType(viu->metadata, fp);
    if (result) {
      // Some datatype has changed, restore in function call
      LLVM_DEBUG(dbgs() << "Restoring call type because of mergeDataTypes()...\n";);
      restoreTypesAcrossFunctionCall(v);
    }

    LLVM_DEBUG(dbgs() << "done with [" << *v << "]\n\n");
    /*auto *iiv = dyn_cast<InputInfo>(viu->metadata.get());

    iiv->IType.reset(fp->clone());*/
  }

  optimizer.printStatInfos();
}

bool TaffoTuner::overwriteType(shared_ptr<ValueInfo> old, shared_ptr<ValueInfo> model)
{
  if (!old || !model)
    return false;

  if (old->getKind() == ValueInfo::K_Scalar) {
    assert(model->getKind() == ValueInfo::K_Scalar && "Mismatching metadata infos!!!");

    std::shared_ptr<ScalarInfo> old1 = dynamic_ptr_cast<ScalarInfo>(old);
    std::shared_ptr<ScalarInfo> model1 = dynamic_ptr_cast<ScalarInfo>(model);

    if (!old1->numericType)
      return false;
    LLVM_DEBUG(dbgs() << "model1: " << model1->numericType->toString() << "\n";);
    LLVM_DEBUG(dbgs() << "old1: " << old1->numericType->toString() << "\n";);
    if (*old1->numericType == *model1->numericType) {
      return false;
    }

    old1->numericType = model1->numericType->clone();
    return true;
  } else if (old->getKind() == ValueInfo::K_Struct) {
    std::shared_ptr<StructInfo> old1 = dynamic_ptr_cast<StructInfo>(old);
    std::shared_ptr<StructInfo> model1 = dynamic_ptr_cast<StructInfo>(model);

    bool changed = false;
    for (unsigned int i = 0; i < old1->numFields(); i++) {
      changed |= overwriteType(old1->getField(i), model1->getField(i));
    }
    return changed;
  }

  llvm_unreachable("unknown data type");
}
#endif // TAFFO_BUILD_ILP_DTA
