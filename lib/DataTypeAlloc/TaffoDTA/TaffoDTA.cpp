#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "DTAConfig.h"
#include "Metadata.h"
#include "TaffoDTA.h"
#include "BufferIDFiles.h"
#ifdef TAFFO_BUILD_ILP_DTA
#include "ILP/MetricBase.h"
#include "ILP/Optimizer.h"
#endif // TAFFO_BUILD_ILP_DTA

using namespace llvm;
using namespace tuner;
using namespace mdutils;
using namespace taffo;

#define DEBUG_TYPE "taffo-dta"

STATISTIC(FixCast, "Number of fixed point format cast");


PreservedAnalyses TaffoTuner::run(Module &m, ModuleAnalysisManager &AM)
{
  MAM = &AM;

  std::vector<llvm::Value *> vals;
  llvm::SmallPtrSet<llvm::Value *, 8U> valset;
  retrieveAllMetadata(m, vals, valset);

#ifdef TAFFO_BUILD_ILP_DTA
  if (MixedMode) {
    LLVM_DEBUG(llvm::dbgs() << "Model " << CostModelFilename << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Inst " << InstructionSet << "\n");
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

  for (Function *f : toDel)
    f->eraseFromParent();

  return PreservedAnalyses::all();
}


/**
 * Reads metadata for the program and DOES THE ACTUAL DATA TYPE ALLOCATION.
 * Yes you read that right.
 */
void TaffoTuner::retrieveAllMetadata(Module &m, std::vector<llvm::Value *> &vals,
                                     llvm::SmallPtrSetImpl<llvm::Value *> &valset)
{
  LLVM_DEBUG(dbgs() << "**********************************************************\n");
  LLVM_DEBUG(dbgs() << __PRETTY_FUNCTION__ << " BEGIN\n");
  LLVM_DEBUG(dbgs() << "**********************************************************\n");

  mdutils::MetadataManager &MDManager = mdutils::MetadataManager::getMetadataManager();

  LLVM_DEBUG(dbgs() << "=============>>>>  " << __FUNCTION__ << " GLOBALS  <<<<===============\n");
  for (GlobalObject &globObj : m.globals()) {
    MDInfo *MDI = MDManager.retrieveMDInfo(&globObj);
    if (processMetadataOfValue(&globObj, MDI)) {
      vals.push_back(&globObj);
      retrieveBufferID(&globObj);
    }
  }
  LLVM_DEBUG(dbgs() << "\n");

  for (Function &f : m.functions()) {
    if (f.isIntrinsic())
      continue;
    LLVM_DEBUG(dbgs() << "=============>>>>  " << __FUNCTION__ << " FUNCTION " << f.getNameOrAsOperand() << "  <<<<===============\n");

    SmallVector<mdutils::MDInfo *, 5> argsII;
    MDManager.retrieveArgumentInputInfo(f, argsII);
    auto arg = f.arg_begin();
    for (auto itII = argsII.begin(); itII != argsII.end(); itII++) {
      if (processMetadataOfValue(arg, *itII)) {
        vals.push_back(arg);
        retrieveBufferID(arg);
      }
      arg++;
    }

    for (inst_iterator iIt = inst_begin(&f), iItEnd = inst_end(&f); iIt != iItEnd; iIt++) {
      MDInfo *MDI = MDManager.retrieveMDInfo(&(*iIt));
      if (processMetadataOfValue(&(*iIt), MDI)) {
        vals.push_back(&*iIt);
        retrieveBufferID(&(*iIt));
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
void TaffoTuner::retrieveBufferID(llvm::Value *V)
{
  LLVM_DEBUG(dbgs() << "Looking up buffer id of " << *V << "\n");
  auto MaybeBID = mdutils::MetadataManager::retrieveBufferIDMetadata(V);
  if (MaybeBID.hasValue()) {
    std::string Tag = *MaybeBID;
    auto& Set = bufferIDSets[Tag];
    Set.insert(V);
    LLVM_DEBUG(dbgs() << "Found buffer ID '" << Tag << "' for " << *V << "\n");
    if (hasInfo(V))
      valueInfo(V)->bufferID = Tag;
  } else {
    LLVM_DEBUG(dbgs() << "No buffer ID for " << *V << "\n");
  }
}


bool TaffoTuner::processMetadataOfValue(Value *v, MDInfo *MDI)
{
  LLVM_DEBUG(dbgs() << "\n" << __FUNCTION__ << " v=" << *v << " MDI=" << (MDI ? MDI->toString() : std::string("(null)"))
                    << "\n");
  if (!MDI) {
    LLVM_DEBUG(dbgs() << "no metadata... bailing out!\n");
    return false;
  }
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

      // FIXME: hack to propagate itofp metadata
      if (MixedMode && (isa<UIToFPInst>(v) || isa<SIToFPInst>(v))) {
        LLVM_DEBUG(dbgs() << "FORCING CONVERSION OF A ITOFP!\n";);
        II->IEnableConversion = true;
      }

      if (!isFloatType(elem.second)) {
        LLVM_DEBUG(dbgs() << "[Info] Skipping a member of " << *v << " because not a float\n");
        continue;
      }

      // TODO: insert logic here to associate different types in a clever way
      if (associateFixFormat(*II, v)) {
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
      for (std::shared_ptr<MDInfo> se : *SI) {
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


bool TaffoTuner::associateFixFormat(InputInfo &II, Value *V)
{
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

  double greatest = std::max(std::abs(rng->Min), std::abs(rng->Max));
  Instruction *I = dyn_cast<Instruction>(V);
  if (I) {
    if (I->isBinaryOp() || I->isUnaryOp()) {
      InputInfo *II = dyn_cast_or_null<InputInfo>(MetadataManager::getMetadataManager().retrieveMDInfo(I->getOperand(0U)));
      if (II && II->IRange.get())
        greatest = std::max(greatest, std::max(std::abs(II->IRange->Max), std::abs(II->IRange->Min)));
      else
        LLVM_DEBUG(dbgs() << "[Warning] No range metadata found on first arg of " << *I << "\n");
    }
    if (I->isBinaryOp()) {
      InputInfo *II = dyn_cast_or_null<InputInfo>(MetadataManager::getMetadataManager().retrieveMDInfo(I->getOperand(1U)));
      if (II && II->IRange.get())
        greatest = std::max(greatest, std::max(std::abs(II->IRange->Max), std::abs(II->IRange->Min)));
      else
        LLVM_DEBUG(dbgs() << "[Warning] No range metadata found on second arg of " << *I << "\n");
    }
  }
  LLVM_DEBUG(dbgs() << "[Info] Maximum value involved in " << *V << " = " << greatest << "\n");

  if (UseFloat != "") {
    mdutils::FloatType::FloatStandard standard;
    if (UseFloat == "f16")
      standard = mdutils::FloatType::Float_half;
    else if (UseFloat == "f32")
      standard = mdutils::FloatType::Float_float;
    else if (UseFloat == "f64")
      standard = mdutils::FloatType::Float_double;
    else if (UseFloat == "bf16")
      standard = mdutils::FloatType::Float_bfloat;
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
      II.IType = res;
      return true;
    }
  } else {
    FixedPointTypeGenError fpgerr;

    /* Testing maximum type for operands, not deciding type yet */
    fixedPointTypeFromRange(Range(0, greatest), &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
    if (fpgerr == FixedPointTypeGenError::NoError) {
      FPType res = fixedPointTypeFromRange(*rng, &fpgerr, TotalBits, FracThreshold, MaxTotalBits, TotalBits);
      if (fpgerr == FixedPointTypeGenError::NoError) {
        LLVM_DEBUG(dbgs() << "[Info] Converting to " << res.toString() << "\n");
        II.IType.reset(res.clone());
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
      LLVM_DEBUG(dbgs() << "[Info] The operands of " << *V << " are not representable as fixed point with specified constraints\n");
    }
  }

  /* We failed, try to keep original type */
  Type *Ty = fullyUnwrapPointerOrArrayType(V->getType());
  if (Ty->isFloatingPointTy()) {
    auto res = std::make_shared<FloatType>(FloatType(Ty->getTypeID(), greatest));
    II.IType = res;
    LLVM_DEBUG(dbgs() << "[Info] Keeping original type which was " << res->toString() << "\n");
    return true;
  }
  
  LLVM_DEBUG(dbgs() << "[Info] The original type was not floating point, skipping (fingers crossed!)\n");
  return false;
}


void TaffoTuner::sortQueue(std::vector<llvm::Value *> &vals,
                           llvm::SmallPtrSetImpl<llvm::Value *> &valset)
{
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
            } else if (utype->isStructTy() && ctype->isStructTy() && ctype->canLosslesslyBitCastTo(utype)) {
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
    if (Argument *Arg = dyn_cast<Argument>(*i)) {
      LLVM_DEBUG(dbgs() << "Restoring consistency of argument " << *Arg << " of function " << Arg->getParent()->getNameOrAsOperand() << "\n");
      restoreTypesAcrossFunctionCall(Arg);
    }
  }
}

void TaffoTuner::mergeFixFormat(const std::vector<llvm::Value *> &vals,
                                const llvm::SmallPtrSetImpl<llvm::Value *> &valset)
{
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

bool TaffoTuner::mergeFixFormat(llvm::Value *v, llvm::Value *u)
{
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

bool TaffoTuner::mergeFixFormatIterative(llvm::Value *v, llvm::Value *u)
{
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
  FPType *fpv = dyn_cast<FPType>(iiv->IType.get());
  FPType *fpu = dyn_cast<FPType>(iiu->IType.get());
  if (!fpv || !fpu) {
    LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because one is not a FPType\n");
    return false;
  }
  if (!(*fpv == *fpu)) {
    if (isMergeable(fpv, fpu)) {
      std::shared_ptr<mdutils::FPType> fp = merge(fpv, fpu);
      if (!fp) {
        LLVM_DEBUG(dbgs() << "not attempting merge of " << *v << ", " << *u << " because resulting type is invalid\n");
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

bool tuner::isMergeable(mdutils::FPType *fpv, mdutils::FPType *fpu)
{
  return fpv->getWidth() == fpu->getWidth() && (std::abs((int)fpv->getPointPos() - (int)fpu->getPointPos()) + (fpv->isSigned() == fpu->isSigned() ? 0 : 1)) <= SimilarBits;
}

std::shared_ptr<mdutils::FPType> tuner::merge(mdutils::FPType *fpv, mdutils::FPType *fpu)
{
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

std::shared_ptr<mdutils::TType> tuner::merge(mdutils::TType *fpv, mdutils::TType *fpu)
{
  if (isa<FPType>(fpv) && isa<FPType>(fpu))
    return merge(dyn_cast<FPType>(fpv), dyn_cast<FPType>(fpu));
  if (isa<FPType>(fpv) && isa<FloatType>(fpu))
    return std::shared_ptr<FloatType>(new FloatType(*dyn_cast<FloatType>(fpu)));
  if (isa<FPType>(fpu) && isa<FloatType>(fpv))
    return std::shared_ptr<FloatType>(new FloatType(*dyn_cast<FloatType>(fpv)));
  if (isa<FloatType>(fpu) && isa<FloatType>(fpv)) {
    FloatType *a = dyn_cast<FloatType>(fpu);
    FloatType *b = dyn_cast<FloatType>(fpv);
    FloatType::FloatStandard MaxStd = std::max(a->getStandard(), b->getStandard());
    double MaxMax = std::max(a->getGreatestNumber(), b->getGreatestNumber());
    return std::shared_ptr<FloatType>(new FloatType(MaxStd, MaxMax));
  }
  llvm_unreachable("unknown TType subclass");
}

void TaffoTuner::mergeBufferIDSets()
{
  LLVM_DEBUG(dbgs() << "\n" << __PRETTY_FUNCTION__ << " BEGIN\n\n");
  BufferIDTypeMap InMap, OutMap;
  if (BufferIDImport != "") {
    LLVM_DEBUG(dbgs() << "Importing Buffer ID sets from " << BufferIDImport << "\n\n");
    ReadBufferIDFile(BufferIDImport, InMap);
  }

  for (auto& Set: bufferIDSets) {
    LLVM_DEBUG(dbgs() << "Merging Buffer ID set " << Set.first << "\n");

    auto DestType = std::shared_ptr<mdutils::TType>(nullptr);
    if (InMap.find(Set.first) != InMap.end()) {
      LLVM_DEBUG(dbgs() << "Set has type specified in file\n");
      DestType.reset(InMap.at(Set.first).get()->clone());
    } else {
      for (auto *V: Set.second) {
        auto VInfo = valueInfo(V);
        auto IInfo = dyn_cast<InputInfo>(VInfo->metadata.get());
        if (!IInfo) {
          LLVM_DEBUG(dbgs() << "Metadata is null or struct, not handled, bailing out! Value='" << *V << "'\n");
          goto nextSet;
        }
        TType *T = IInfo->IType.get();
        if (T) {
          LLVM_DEBUG(dbgs() << "Type=" << T->toString() << " Value='" << *V << "'\n");
        } else {
          LLVM_DEBUG(dbgs() << "Type is null, not handled, bailing out! Value='" << *V << "'\n");
          continue;
        }
        
        if (!DestType.get()) {
          DestType.reset(T->clone());
        } else {
          DestType = merge(DestType.get(), T);
        }
      }
    }
    LLVM_DEBUG(dbgs() << "Computed merged type: " << DestType->toString() << "\n");

    for (auto *V: Set.second) {
      auto VInfo = valueInfo(V);
      auto IInfo = dyn_cast<InputInfo>(VInfo->metadata.get());
      IInfo->IType.reset(DestType->clone());
      restoreTypesAcrossFunctionCall(V);
    }
    OutMap[Set.first].reset(DestType->clone());

nextSet:
    LLVM_DEBUG(dbgs() << "Merging Buffer ID set " << Set.first << " DONE\n\n");
  }

  if (BufferIDExport != "") {
    LLVM_DEBUG(dbgs() << "Exporting Buffer ID sets to " << BufferIDExport << "\n\n");
    WriteBufferIDFile(BufferIDExport, OutMap);
  }

  LLVM_DEBUG(dbgs() << __PRETTY_FUNCTION__ << " END\n\n");
}


void TaffoTuner::restoreTypesAcrossFunctionCall(Value *v)
{
  LLVM_DEBUG(dbgs() << "restoreTypesAcrossFunctionCall(" << *v << ")\n");
  if (!hasInfo(v)) {
    LLVM_DEBUG(dbgs() << " --> skipping restoring types because value is not converted\n");
    return;
  }

  std::shared_ptr<MDInfo> finalMd = valueInfo(v)->metadata;

  if (Argument *arg = dyn_cast<Argument>(v)) {
    LLVM_DEBUG(dbgs() << "Is a function argument, propagating to calls\n");
    setTypesOnCallArgumentFromFunctionArgument(arg, finalMd);
  } else {
    LLVM_DEBUG(dbgs() << "Not a function argument, propagating to function arguments\n");
    setTypesOnFunctionArgumentFromCallArgument(v, finalMd);
  }
  
  LLVM_DEBUG(dbgs() << "restoreTypesAcrossFunctionCall ended\n");
}


void TaffoTuner::setTypesOnFunctionArgumentFromCallArgument(Value *v, std::shared_ptr<MDInfo> finalMd)
{
  for (Use &use : v->uses()) {
    User *user = use.getUser();
    CallBase *call = dyn_cast<CallBase>(user);
    if (call == nullptr)
      continue;
    LLVM_DEBUG(dbgs() << "restoreTypesAcrossFunctionCall: processing user " << *(user) << ")\n");

    Function *fun = dyn_cast<Function>(call->getCalledFunction());
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
      LLVM_DEBUG(dbgs() << " --> set new metadata, now checking uses of the argument... (hope there's no recursion!)\n");
      setTypesOnFunctionArgumentFromCallArgument(arg, finalMd);
    } else {
      LLVM_DEBUG(dbgs() << "Not looking good, formal arg #" << use.getOperandNo() << " (" << *arg << ") has no valueInfo, but actual argument does...\n");
    }
  }
}


void TaffoTuner::setTypesOnCallArgumentFromFunctionArgument(Argument *arg, std::shared_ptr<MDInfo> finalMd)
{
  Function *fun = arg->getParent();
  int n = arg->getArgNo();
  LLVM_DEBUG(dbgs() << " --> setting types to " << finalMd->toString() << " on call arguments from function "
                    << fun->getName() << " argument " << n << "\n");
  for (auto it = fun->user_begin(); it != fun->user_end(); it++) {
    if (isa<CallInst>(*it) || isa<InvokeInst>(*it)) {
      Value *callarg = it->getOperand(n);
      LLVM_DEBUG(dbgs() << " --> target " << *callarg << ", CallBase " << **it << "\n");

      if (!hasInfo(callarg)) {
        if (!isa<Argument>(callarg)) {
          LLVM_DEBUG(dbgs() << " --> actual argument doesn't get converted; skipping\n");
          continue;
        } else {
          LLVM_DEBUG(dbgs() << " --> actual argument IS AN ARGUMENT ITSELF! not skipping even if it doesn't get converted\n");
        }
      }
      valueInfo(callarg)->metadata.reset(finalMd->clone());
      if (Argument *Arg = dyn_cast<Argument>(callarg)) {
        LLVM_DEBUG(dbgs() << " --> actual argument IS AN ARGUMENT ITSELF, recursing\n");
        setTypesOnCallArgumentFromFunctionArgument(Arg, finalMd);
      }
    }
  }
}


std::vector<Function *> TaffoTuner::collapseFunction(Module &m)
{
  std::vector<Function *> toDel;
  for (Function &f : m.functions()) {
    if (MDNode *mdNode = f.getMetadata(CLONED_FUN_METADATA)) {
      if (std::find(toDel.begin(), toDel.end(), &f) != toDel.end())
        continue;
      LLVM_DEBUG(dbgs() << "Analyzing original function " << f.getName() << "\n";);

      for (auto mdIt = mdNode->op_begin(); mdIt != mdNode->op_end(); mdIt++) {
        LLVM_DEBUG(dbgs() << "\t Clone : " << **mdIt << "\n";);

        ValueAsMetadata *md = dyn_cast<ValueAsMetadata>(*mdIt);
        Function *fClone = dyn_cast<Function>(md->getValue());
        if (fClone->user_begin() == fClone->user_end()) {
          LLVM_DEBUG(dbgs() << "\t Ignoring " << fClone->getName()
              << " because it's not used anywhere\n");
        } else if (Function *eqFun = findEqFunction(fClone, &f)) {
          LLVM_DEBUG(dbgs() << "\t Replace function " << fClone->getName()
              << " with " << eqFun->getName() << "\n";);
          fClone->replaceAllUsesWith(eqFun);
          toDel.push_back(fClone);
        }
      }
    }
  }
  return toDel;
}


bool compareTypesOfMDInfo(MDInfo &mdi1, MDInfo &mdi2)
{
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


Function *TaffoTuner::findEqFunction(Function *fun, Function *origin)
{
  std::vector<std::pair<int, std::shared_ptr<MDInfo>>> fixSign;

  LLVM_DEBUG(dbgs() << "\t\t Search eq function for " << fun->getName()
      << " in " << origin->getName() << " pool\n";);

  if (isFloatType(fun->getReturnType()) && hasInfo(*fun->user_begin())) {
    std::shared_ptr<MDInfo> retval = valueInfo(*fun->user_begin())->metadata;
    if (retval) {
      fixSign.push_back(std::pair<int, std::shared_ptr<MDInfo>>(-1, retval)); // ret value in signature
      LLVM_DEBUG(dbgs() << "\t\t Return type : "
          << valueInfo(*fun->user_begin())->metadata->toString() << "\n";);
    }
  }

  int i = 0;
  for (Argument &arg : fun->args()) {
    if (hasInfo(&arg) && valueInfo(&arg)->metadata) {
      fixSign.push_back(std::pair<int, std::shared_ptr<MDInfo>>(i, valueInfo(&arg)->metadata));
      LLVM_DEBUG(dbgs() << "\t\t Arg " << i << " type : "
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
  LLVM_DEBUG(dbgs() << "\t Function " << fun->getName() << " used\n";);
  return nullptr;
}


void TaffoTuner::attachFPMetaData(std::vector<llvm::Value *> &vals)
{
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


void TaffoTuner::attachFunctionMetaData(llvm::Module &m)
{
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

#ifdef TAFFO_BUILD_ILP_DTA
void TaffoTuner::buildModelAndOptimze(Module &m, const vector<llvm::Value *> &vals,
                                      const SmallPtrSetImpl<llvm::Value *> &valset)
{
  assert(vals.size() == valset.size() && "They must contain the same elements.");


  Optimizer optimizer(m, this, new MetricPerf(), CostModelFilename, CPUCosts::CostType::Performance);
  // Optimizer optimizer(m, this, new MetricPerf(),"", CPUCosts::CostType::Size);
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

    std::shared_ptr<ValueInfo> viu = valueInfo(v);

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

bool TaffoTuner::overwriteType(shared_ptr<mdutils::MDInfo> old, shared_ptr<mdutils::MDInfo> model)
{
  if (!old || !model)
    return false;

  if (old->getKind() == mdutils::MDInfo::K_Field) {
    assert(model->getKind() == mdutils::MDInfo::K_Field && "Mismatching metadata infos!!!");

    auto old1 = dynamic_ptr_cast_or_null<InputInfo>(old);
    auto model1 = dynamic_ptr_cast_or_null<InputInfo>(model);

    if (!old1->IType)
      return false;
    LLVM_DEBUG(dbgs() << "model1: " << model1->IType->toString() << "\n";);
    LLVM_DEBUG(dbgs() << "old1: " << old1->IType->toString() << "\n";);
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
      changed |= overwriteType(old1->getField(i), model1->getField(i));
    }
    return changed;
  }

  llvm_unreachable("unknown data type");
}
#endif // TAFFO_BUILD_ILP_DTA
