//===-- Metadata.cpp - Metadata Utils for ErrorPropagator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definitions of utility functions that handle metadata in Error Propagator.
///
//===----------------------------------------------------------------------===//

#include "Metadata.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <sstream>

namespace mdutils
{

using namespace llvm;

MetadataManager &MetadataManager::getMetadataManager()
{
  static MetadataManager Instance;
  return Instance;
}

MDInfo *MetadataManager::retrieveMDInfo(const Value *v)
{
  if (const Instruction *i = dyn_cast<Instruction>(v)) {
    if (MDNode *mdn = i->getMetadata(INPUT_INFO_METADATA)) {
      return retrieveInputInfo(mdn).get();
    } else if (MDNode *mdn = i->getMetadata(STRUCT_INFO_METADATA)) {
      return retrieveStructInfo(mdn).get();
    } else
      return nullptr;
  } else if (const GlobalObject *go = dyn_cast<GlobalObject>(v)) {
    if (MDNode *mdn = go->getMetadata(INPUT_INFO_METADATA)) {
      return retrieveInputInfo(mdn).get();
    } else if (MDNode *mdn = go->getMetadata(STRUCT_INFO_METADATA)) {
      return retrieveStructInfo(mdn).get();
    } else
      return nullptr;
  } else if (const Argument *arg = dyn_cast<Argument>(v)) {
    const Function *fun = arg->getParent();
    llvm::SmallVector<MDInfo *, 2> famd;
    retrieveArgumentInputInfo(*fun, famd);
    return famd[arg->getArgNo()];
  }
  return nullptr;
}

InputInfo *MetadataManager::retrieveInputInfo(const Instruction &I)
{
  return retrieveInputInfo(I.getMetadata(INPUT_INFO_METADATA)).get();
}

InputInfo *MetadataManager::retrieveInputInfo(const GlobalObject &V)
{
  return retrieveInputInfo(V.getMetadata(INPUT_INFO_METADATA)).get();
}

void MetadataManager::
    retrieveArgumentInputInfo(const Function &F, SmallVectorImpl<MDInfo *> &ResII)
{
  MDNode *ArgsMD = F.getMetadata(FUNCTION_ARGS_METADATA);
  if (ArgsMD == nullptr)
    return;

  assert((ArgsMD->getNumOperands() % 2) == 0 && "invalid funinfo");
  unsigned nfunargs = ArgsMD->getNumOperands() / 2;
  assert(nfunargs == F.getFunctionType()->getNumParams() && "invalid funinfo");
  ResII.reserve(nfunargs);
  for (auto ArgMDOp = ArgsMD->op_begin(), ArgMDOpEnd = ArgsMD->op_end();
       ArgMDOp != ArgMDOpEnd;) {
    Constant *mdtid = cast<ConstantAsMetadata>(ArgMDOp->get())->getValue();
    ArgMDOp++;
    int tid = cast<ConstantInt>(mdtid)->getZExtValue();
    switch (tid) {
    case 0:
      ResII.push_back(nullptr);
      break;
    case 1:
      ResII.push_back(retrieveInputInfo(cast<MDNode>(ArgMDOp->get())).get());
      break;
    case 2:
      ResII.push_back(retrieveStructInfo(cast<MDNode>(ArgMDOp->get())).get());
      break;
    default:
      assert(0 && "invalid funinfo type id");
    }
    ArgMDOp++;
  }
}

void MetadataManager::
    retrieveConstInfo(const llvm::Instruction &I,
                      llvm::SmallVectorImpl<InputInfo *> &ResII)
{
  MDNode *ArgsMD = I.getMetadata(CONST_INFO_METADATA);
  if (ArgsMD == nullptr)
    return;

  ResII.reserve(ArgsMD->getNumOperands());
  for (const MDOperand &MDOp : ArgsMD->operands()) {
    if (ConstantAsMetadata *CMD = dyn_cast<ConstantAsMetadata>(MDOp)) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(CMD->getValue())) {
        if (CI->isZero()) {
          ResII.push_back(nullptr);
          continue;
        }
      }
    }
    ResII.push_back(retrieveInputInfo(cast<MDNode>(MDOp)).get());
  }
}

void MetadataManager::
    setMDInfoMetadata(llvm::Value *u, const MDInfo *mdinfo)
{
  StringRef mdid;

  if (isa<InputInfo>(mdinfo)) {
    mdid = INPUT_INFO_METADATA;
  } else if (isa<StructInfo>(mdinfo)) {
    mdid = STRUCT_INFO_METADATA;
  } else {
    assert(false && "unknown MDInfo class");
  }

  if (Instruction *instr = dyn_cast<Instruction>(u)) {
    instr->setMetadata(mdid, mdinfo->toMetadata(u->getContext()));
  } else if (GlobalObject *go = dyn_cast<GlobalObject>(u)) {
    go->setMetadata(mdid, mdinfo->toMetadata(u->getContext()));
  } else {
    assert(false && "parameter not an instruction or a global object");
  }
}

void MetadataManager::
    setInputInfoMetadata(Instruction &I, const InputInfo &IInfo)
{
  I.setMetadata(INPUT_INFO_METADATA, IInfo.toMetadata(I.getContext()));
}

void MetadataManager::
    setInputInfoMetadata(GlobalObject &V, const InputInfo &IInfo)
{
  V.setMetadata(INPUT_INFO_METADATA, IInfo.toMetadata(V.getContext()));
}

void MetadataManager::
    setArgumentInputInfoMetadata(Function &F, const ArrayRef<MDInfo *> AInfo)
{
  LLVMContext &Context = F.getContext();
  SmallVector<Metadata *, 2U> AllArgsMD;
  AllArgsMD.reserve(AInfo.size());

  for (MDInfo *info : AInfo) {
    int tid = -1;
    Metadata *val;
    if (info == nullptr) {
      tid = 0;
      val = ConstantAsMetadata::get(Constant::getNullValue(Type::getInt1Ty(Context)));
    } else if (InputInfo *IInfo = dyn_cast<InputInfo>(info)) {
      tid = 1;
      val = IInfo->toMetadata(Context);
    } else if (StructInfo *SInfo = dyn_cast<StructInfo>(info)) {
      tid = 2;
      val = SInfo->toMetadata(Context);
    } else {
      llvm_unreachable("invalid MDInfo in array");
    }
    ConstantInt *ctid = ConstantInt::get(IntegerType::getInt32Ty(Context), tid);
    ConstantAsMetadata *mdtid = ConstantAsMetadata::get(ctid);
    AllArgsMD.push_back(mdtid);
    AllArgsMD.push_back(val);
  }

  assert(AllArgsMD.size() / 2 == F.getFunctionType()->getNumParams() && "writing malformed funinfo");

  F.setMetadata(FUNCTION_ARGS_METADATA, MDNode::get(Context, AllArgsMD));
}

void MetadataManager::
    setConstInfoMetadata(llvm::Instruction &I,
                         const llvm::ArrayRef<InputInfo *> CInfo)
{
  assert(I.getNumOperands() == CInfo.size() && "Must provide InputInfo or nullptr for each operand.");
  LLVMContext &Context = I.getContext();
  SmallVector<Metadata *, 2U> ConstMDs;
  ConstMDs.reserve(CInfo.size());

  for (InputInfo *II : CInfo) {
    if (II) {
      ConstMDs.push_back(II->toMetadata(Context));
    } else {
      ConstMDs.push_back(ConstantAsMetadata::get(ConstantInt::getFalse(Context)));
    }
  }

  I.setMetadata(CONST_INFO_METADATA, MDNode::get(Context, ConstMDs));
}

StructInfo *MetadataManager::retrieveStructInfo(const Instruction &I)
{
  return retrieveStructInfo(I.getMetadata(STRUCT_INFO_METADATA)).get();
}

StructInfo *MetadataManager::retrieveStructInfo(const GlobalObject &V)
{
  return retrieveStructInfo(V.getMetadata(STRUCT_INFO_METADATA)).get();
}

void MetadataManager::setStructInfoMetadata(Instruction &I, const StructInfo &SInfo)
{
  I.setMetadata(STRUCT_INFO_METADATA, SInfo.toMetadata(I.getContext()));
}

void MetadataManager::setStructInfoMetadata(GlobalObject &V, const StructInfo &SInfo)
{
  V.setMetadata(STRUCT_INFO_METADATA, SInfo.toMetadata(V.getContext()));
}


void MetadataManager::setInputInfoInitWeightMetadata(Value *v, int weight)
{
  assert((isa<Instruction>(v) || isa<GlobalObject>(v)) && "v not an instruction or a global object");
  ConstantInt *cweight = ConstantInt::get(IntegerType::getInt32Ty(v->getContext()), weight);
  ConstantAsMetadata *mdweight = ConstantAsMetadata::get(cweight);
  if (Instruction *i = dyn_cast<Instruction>(v)) {
    i->setMetadata(INIT_WEIGHT_METADATA, MDNode::get(v->getContext(), mdweight));
  } else if (GlobalObject *go = dyn_cast<GlobalObject>(v)) {
    go->setMetadata(INIT_WEIGHT_METADATA, MDNode::get(v->getContext(), mdweight));
  }
}


int MetadataManager::retrieveInputInfoInitWeightMetadata(const Value *v)
{
  MDNode *node;
  if (const Instruction *i = dyn_cast<Instruction>(v)) {
    node = i->getMetadata(INIT_WEIGHT_METADATA);
  } else if (const GlobalObject *go = dyn_cast<GlobalObject>(v)) {
    node = go->getMetadata(INIT_WEIGHT_METADATA);
  } else {
    return INT_MAX;
  }
  if (!node)
    return -1;
  assert(node->getNumOperands() == 1 && "malformed " INIT_WEIGHT_METADATA " metadata node");
  ConstantAsMetadata *mdweight = cast<ConstantAsMetadata>(node->getOperand(0U));
  ConstantInt *cweight = cast<ConstantInt>(mdweight->getValue());
  return cweight->getZExtValue();
}

void MetadataManager::setOpenCLCloneTrampolineMetadata(Function *F, Function *KernF)
{
  F->setMetadata(INIT_OCL_TRAMPOLINE_METADATA, MDNode::get(F->getContext(), {ValueAsMetadata::get(KernF)}));
}

bool MetadataManager::retrieveOpenCLCloneTrampolineMetadata(Function *F, Function **KernF)
{
  MDNode *MDN = F->getMetadata(INIT_OCL_TRAMPOLINE_METADATA);
  if (!MDN)
    return false;
  ValueAsMetadata *VAM = cast<ValueAsMetadata>(MDN->getOperand(0U));
  Function *OutF = cast<Function>(VAM->getValue());
  if (KernF)
    *KernF = OutF;
  return true;
}

void MetadataManager::setBufferIDMetadata(Value *V, std::string BufID)
{
  Metadata *TheString = MDString::get(V->getContext(), BufID);

  if (Argument *Arg = dyn_cast<Argument>(V)) {
    Function *F = Arg->getParent();
    MDNode *Node = F->getMetadata(INIT_FUN_ARGS_BUFFER_ID_METADATA);
    SmallVector<Metadata *> MDs;
    if (!Node) {
      MDs = SmallVector<Metadata *>(F->arg_size(), nullptr);
    } else {
      MDs = SmallVector<Metadata *>(Node->op_begin(), Node->op_end());
    }
    MDs[Arg->getArgNo()] = TheString;
    MDNode *NewNode = MDTuple::get(V->getContext(), MDs);
    F->setMetadata(INIT_FUN_ARGS_BUFFER_ID_METADATA, NewNode);
  } else {
    MDNode *NewNode = MDNode::get(V->getContext(), {TheString});
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      I->setMetadata(INIT_FUN_ARGS_BUFFER_ID_METADATA, NewNode);
    } else if (GlobalObject *GO = dyn_cast<GlobalObject>(V)) {
      GO->setMetadata(INIT_FUN_ARGS_BUFFER_ID_METADATA, NewNode);
    } else {
      llvm_unreachable("attempted to attach metadata to unsupported value type");
    }
  }
}

std::optional<std::string> MetadataManager::retrieveBufferIDMetadata(Value *V)
{
  MDString *String;

  if (Argument *Arg = dyn_cast<Argument>(V)) {
    Function *F = Arg->getParent();
    MDNode *Node = F->getMetadata(INIT_FUN_ARGS_BUFFER_ID_METADATA);
    if (!Node || Node->getNumOperands() != F->arg_size())
      return std::optional<std::string>();
    String = dyn_cast_or_null<MDString>(Node->getOperand(Arg->getArgNo()));
  } else {
    MDNode *Node;
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      Node = I->getMetadata(INIT_FUN_ARGS_BUFFER_ID_METADATA);
    } else if (GlobalObject *GO = dyn_cast<GlobalObject>(V)) {
      Node = GO->getMetadata(INIT_FUN_ARGS_BUFFER_ID_METADATA);
    } else {
      llvm_unreachable("attempted to attach metadata to unsupported value type");
    }
    if (!Node || Node->getNumOperands() != 1)
      return std::optional<std::string>();
    String = dyn_cast_or_null<MDString>(Node->getOperand(0U));
  }

  if (String)
    return std::optional<std::string>(std::string(String->getString()));
  return std::optional<std::string>();
}

void MetadataManager::getCudaKernels(Module &M, SmallVectorImpl<Function *> &Fs) {
  NamedMDNode *MD = M.getNamedMetadata(CUDA_KERNEL_METADATA);
 
  if (!MD)
    return;
 
  for (auto *Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;
    MDString *KindID = dyn_cast<MDString>(Op->getOperand(1));
    if (!KindID || KindID->getString() != "kernel")
      continue;
 
    Function *KernelFn = mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
    if (!KernelFn)
      continue;
 
    Fs.append({KernelFn});
  }
 
  return;
}

bool MetadataManager::isCudaKernel(llvm::Module &m, llvm::Function *f){
  llvm::MDNode *mdn = f->getMetadata(SOURCE_FUN_METADATA); 
  if(mdn)
    f = mdconst::dyn_extract_or_null<Function>(mdn->getOperand(0));
  SmallVector<Function *, 2> KernFs;
  mdutils::MetadataManager::getCudaKernels(m, KernFs);
  for (Function *kernel: KernFs) {
    if(f == kernel) 
      return true;
  }
  return false;
}

void MetadataManager::setInputInfoInitWeightMetadata(llvm::Function *f,
                                                     const llvm::ArrayRef<int> weights)
{
  SmallVector<Metadata *, 4U> wmds;
  wmds.reserve(weights.size());
  for (int w : weights) {
    ConstantInt *cweight = ConstantInt::get(IntegerType::getInt32Ty(f->getContext()), w);
    ConstantAsMetadata *mdweight = ConstantAsMetadata::get(cweight);
    wmds.push_back(mdweight);
  }
  f->setMetadata(INIT_WEIGHT_METADATA, MDNode::get(f->getContext(), wmds));
}

void MetadataManager::retrieveInputInfoInitWeightMetadata(llvm::Function *f,
                                                          llvm::SmallVectorImpl<int> &ResWs)
{
  MDNode *node = f->getMetadata(INIT_WEIGHT_METADATA);
  if (!node)
    return;

  ResWs.reserve(f->arg_size());
  for (unsigned i = 0; i < f->arg_size(); ++i) {
    if (i < node->getNumOperands()) {
      ConstantAsMetadata *mdweight = cast<ConstantAsMetadata>(node->getOperand(i));
      ConstantInt *cweight = cast<ConstantInt>(mdweight->getValue());
      ResWs.push_back(cweight->getZExtValue());
    } else {
      ResWs.push_back(-1);
    }
  }
}

void MetadataManager::
    setMaxRecursionCountMetadata(Function &F, unsigned MaxRecursionCount)
{
  ConstantInt *CIRC = ConstantInt::get(Type::getInt32Ty(F.getContext()),
                                       MaxRecursionCount,
                                       false);
  ConstantAsMetadata *CMRC = ConstantAsMetadata::get(CIRC);
  MDNode *RCNode = MDNode::get(F.getContext(), CMRC);
  F.setMetadata(MAX_REC_METADATA, RCNode);
}

unsigned MetadataManager::
    retrieveMaxRecursionCount(const Function &F)
{
  MDNode *RecC = F.getMetadata(MAX_REC_METADATA);
  if (RecC == nullptr)
    return 0U;

  assert(RecC->getNumOperands() > 0 && "Must contain the recursion count.");
  ConstantAsMetadata *CMRC = cast<ConstantAsMetadata>(RecC->getOperand(0U));
  ConstantInt *CIRC = cast<ConstantInt>(CMRC->getValue());
  return CIRC->getZExtValue();
}

void MetadataManager::
    setLoopUnrollCountMetadata(Loop &L, unsigned UnrollCount)
{
  // Get Loop header terminating instruction
  BasicBlock *Header = L.getHeader();
  assert(Header && "Loop with no header.");

  Instruction *HTI = Header->getTerminator();
  assert(HTI && "Block with no terminator.");

  // Prepare MD Node
  ConstantInt *CIUC = ConstantInt::get(Type::getInt32Ty(HTI->getContext()),
                                       UnrollCount,
                                       false);
  ConstantAsMetadata *CMUC = ConstantAsMetadata::get(CIUC);
  MDNode *UCNode = MDNode::get(HTI->getContext(), CMUC);

  HTI->setMetadata(UNROLL_COUNT_METADATA, UCNode);
}

void MetadataManager::
    setLoopUnrollCountMetadata(Function &F,
                               const SmallVectorImpl<std::optional<unsigned>> &LUCs)
{
  std::ostringstream EncLUCs;
  for (const std::optional<unsigned> &LUC : LUCs) {
    if (LUC.has_value())
      EncLUCs << LUC.value() << " ";
    else
      EncLUCs << "U ";
  }

  F.setMetadata(UNROLL_COUNT_METADATA,
                MDNode::get(F.getContext(),
                            MDString::get(F.getContext(), EncLUCs.str())));
}

std::optional<unsigned>
MetadataManager::retrieveLoopUnrollCount(const Loop &L, LoopInfo *LI)
{
  std::optional<unsigned> MDLUC = retrieveLUCFromHeaderMD(L);

  if (!MDLUC.has_value() && LI)
    return retrieveLUCFromFunctionMD(L, *LI);

  return MDLUC;
}

std::optional<unsigned>
MetadataManager::retrieveLUCFromHeaderMD(const Loop &L)
{
  // Get Loop header terminating instruction
  BasicBlock *Header = L.getHeader();
  assert(Header && "Loop with no header.");

  Instruction *HTI = Header->getTerminator();
  assert(HTI && "Block with no terminator.");

  MDNode *UCNode = HTI->getMetadata(UNROLL_COUNT_METADATA);
  if (UCNode == nullptr)
    return std::nullopt;

  assert(UCNode->getNumOperands() > 0 && "Must contain the unroll count.");
  ConstantAsMetadata *CMUC = cast<ConstantAsMetadata>(UCNode->getOperand(0U));
  ConstantInt *CIUC = cast<ConstantInt>(CMUC->getValue());
  return CIUC->getZExtValue();
}

std::optional<unsigned>
MetadataManager::retrieveLUCFromFunctionMD(const Loop &L, LoopInfo &LI)
{
  unsigned LIdx = getLoopIndex(L, LI);

  Function *F = L.getHeader()->getParent();
  assert(F);
  SmallVector<std::optional<unsigned>, 4U> LUCs = retrieveLUCListFromFunctionMD(*F);

  if (LIdx >= LUCs.size())
    return std::nullopt;

  return LUCs[LIdx];
}

unsigned
MetadataManager::getLoopIndex(const Loop &L, LoopInfo &LI)
{
  unsigned LIdx = 0;
  for (const Loop *CLoop : LI.getLoopsInPreorder()) {
    if (&L == CLoop)
      return LIdx;
    else
      ++LIdx;
  }
  llvm_unreachable("User-provided loop not found in LoopInfo.");
}

SmallVector<std::optional<unsigned>, 4U>
MetadataManager::retrieveLUCListFromFunctionMD(Function &F)
{
  SmallVector<std::optional<unsigned>, 4U> LUCList;

  MDNode *LUCListMDN = F.getMetadata(UNROLL_COUNT_METADATA);
  if (!LUCListMDN)
    return LUCList;

  MDString *LUCListMDS = dyn_cast<MDString>(LUCListMDN->getOperand(0U).get());
  if (!LUCListMDS)
    return LUCList;

  SmallVector<StringRef, 4U> LUCSRefs;
  LUCListMDS->getString().split(LUCSRefs, ' ', -1, false);
  for (const StringRef &LUCSR : LUCSRefs) {
    errs() << LUCSR;
    unsigned LUC;
    if (!LUCSR.getAsInteger(10U, LUC)) {
      LUCList.push_back(LUC);
      errs() << " done\n";
    } else {
      LUCList.push_back(std::nullopt);
      errs() << " nope\n";
    }
  }
  return LUCList;
}

void MetadataManager::setErrorMetadata(Instruction &I, double Error)
{
  I.setMetadata(COMP_ERROR_METADATA, createDoubleMDNode(I.getContext(), Error));
}

double MetadataManager::retrieveErrorMetadata(const Instruction &I)
{
  return retrieveDoubleMDNode(I.getMetadata(COMP_ERROR_METADATA));
}

void MetadataManager::
    setCmpErrorMetadata(Instruction &I, const CmpErrorInfo &CEI)
{
  if (!CEI.MayBeWrong)
    return;

  I.setMetadata(WRONG_CMP_METADATA, CEI.toMetadata(I.getContext()));
}

std::unique_ptr<CmpErrorInfo> MetadataManager::
    retrieveCmpError(const Instruction &I)
{
  return CmpErrorInfo::createFromMetadata(I.getMetadata(WRONG_CMP_METADATA));
}

void MetadataManager::setStartingPoint(Function &F)
{
  Metadata *MD[] = {ConstantAsMetadata::get(ConstantInt::getTrue(F.getContext()))};
  F.setMetadata(START_FUN_METADATA, MDNode::get(F.getContext(), MD));
}

bool MetadataManager::isStartingPoint(const Function &F)
{
  return F.getMetadata(START_FUN_METADATA) != nullptr;
}

void MetadataManager::defaultStartingPoint(Module &M)
{
  auto main = llvm::find_if(M.functions(), [](const Function &F) { return F.getName().equals("main"); } );
  if (main != M.end()) {
    setStartingPoint(*main);
  }
}

bool MetadataManager::hasStartingPoint(const Module &M)
{
  bool hasStartingPoint = false;
  hasStartingPoint = llvm::any_of(M.functions(), [](const auto &F) { return isStartingPoint(F); });
  return hasStartingPoint;
}


void MetadataManager::setTargetMetadata(Instruction &I, StringRef Name)
{
  MDNode *TMD = MDNode::get(I.getContext(), MDString::get(I.getContext(), Name));
  I.setMetadata(TARGET_METADATA, TMD);
}

std::optional<StringRef> MetadataManager::retrieveTargetMetadata(const Instruction &I)
{
  MDNode *MD = I.getMetadata(TARGET_METADATA);
  if (MD == nullptr)
    return std::nullopt;

  MDString *MDName = cast<MDString>(MD->getOperand(0U).get());
  return MDName->getString();
}

void MetadataManager::setTargetMetadata(GlobalObject &I, StringRef Name)
{
  MDNode *TMD = MDNode::get(I.getContext(), MDString::get(I.getContext(), Name));
  I.setMetadata(TARGET_METADATA, TMD);
}

std::optional<StringRef> MetadataManager::retrieveTargetMetadata(const GlobalObject &I)
{
  MDNode *MD = I.getMetadata(TARGET_METADATA);
  if (MD == nullptr)
    return std::nullopt;

  MDString *MDName = cast<MDString>(MD->getOperand(0U).get());
  return MDName->getString();
}


std::shared_ptr<TType> MetadataManager::retrieveTType(MDNode *MDN)
{
  if (MDN == nullptr)
    return nullptr;

  auto CachedTT = TTypes.find(MDN);
  if (CachedTT != TTypes.end())
    return CachedTT->second;

  std::shared_ptr<TType> TT(TType::createFromMetadata(MDN));

  TTypes.insert(std::make_pair(MDN, TT));
  return TT;
}

std::shared_ptr<Range> MetadataManager::retrieveRange(MDNode *MDN)
{
  if (MDN == nullptr)
    return nullptr;

  auto CachedRange = Ranges.find(MDN);
  if (CachedRange != Ranges.end())
    return CachedRange->second;

  std::shared_ptr<Range> NRange(Range::createFromMetadata(MDN));

  Ranges.insert(std::make_pair(MDN, NRange));
  return NRange;
}

std::shared_ptr<double> MetadataManager::retrieveError(MDNode *MDN)
{
  if (MDN == nullptr)
    return nullptr;

  auto CachedError = IErrors.find(MDN);
  if (CachedError != IErrors.end())
    return CachedError->second;

  std::shared_ptr<double> NError(CreateInitialErrorFromMetadata(MDN));

  IErrors.insert(std::make_pair(MDN, NError));
  return NError;
}

std::shared_ptr<InputInfo> MetadataManager::retrieveInputInfo(MDNode *MDN)
{
  if (MDN == nullptr)
    return nullptr;

  auto CachedIInfo = IInfos.find(MDN);
  if (CachedIInfo != IInfos.end())
    return CachedIInfo->second;

  std::shared_ptr<InputInfo> NIInfo(createInputInfoFromMetadata(MDN));

  IInfos.insert(std::make_pair(MDN, NIInfo));
  return NIInfo;
}

std::shared_ptr<StructInfo> MetadataManager::retrieveStructInfo(MDNode *MDN)
{
  if (MDN == nullptr)
    return nullptr;

  auto CachedStructInfo = StructInfos.find(MDN);
  if (CachedStructInfo != StructInfos.end())
    return CachedStructInfo->second;

  std::shared_ptr<StructInfo> NSInfo(createStructInfoFromMetadata(MDN));

  StructInfos.insert(std::make_pair(MDN, NSInfo));
  return NSInfo;
}

std::unique_ptr<InputInfo> MetadataManager::
    createInputInfoFromMetadata(MDNode *MDN)
{
  assert(MDN != nullptr);
  assert(MDN->getNumOperands() >= 3U && "Must have Type, Range, Initial Error, [Flags]");

  Metadata *ITypeMDN = MDN->getOperand(0U).get();
  std::shared_ptr<TType> IType = (IsNullInputInfoField(ITypeMDN))
                                     ? nullptr
                                     : retrieveTType(cast<MDNode>(ITypeMDN));

  Metadata *IRangeMDN = MDN->getOperand(1U).get();
  std::shared_ptr<Range> IRange = (IsNullInputInfoField(IRangeMDN))
                                      ? nullptr
                                      : retrieveRange(cast<MDNode>(IRangeMDN));

  Metadata *IErrorMDN = MDN->getOperand(2U).get();
  std::shared_ptr<double> IError = (IsNullInputInfoField(IErrorMDN))
                                       ? nullptr
                                       : retrieveError(cast<MDNode>(IErrorMDN));

  bool IEnabled = true;
  bool IFinal = false;
  if (MDN->getNumOperands() >= 4U) {
    Metadata *IFlagsMDN = MDN->getOperand(3U).get();
    if (IFlagsMDN) {
      ConstantAsMetadata *tmpmd = cast<ConstantAsMetadata>(IFlagsMDN);
      uint64_t tmpint = cast<ConstantInt>(tmpmd->getValue())->getZExtValue();
      IEnabled = tmpint & 1U;
      IFinal = tmpint & 2U;
    }
  }

  return std::unique_ptr<InputInfo>(new InputInfo(IType, IRange, IError, IEnabled, IFinal));
}

std::unique_ptr<StructInfo> MetadataManager::
    createStructInfoFromMetadata(MDNode *MDN)
{
  assert(MDN != nullptr);

  SmallVector<std::shared_ptr<MDInfo>, 4U> Fields;
  Fields.reserve(MDN->getNumOperands());
  for (const MDOperand &MDO : MDN->operands()) {
    Metadata *MDField = MDO.get();
    assert(MDField != nullptr);
    if (IsNullInputInfoField(MDField)) {
      Fields.push_back(nullptr);
    } else if (InputInfo::isInputInfoMetadata(MDField)) {
      Fields.push_back(retrieveInputInfo(cast<MDNode>(MDField)));
    } else if (MDNode *MDNField = dyn_cast<MDNode>(MDField)) {
      Fields.push_back(retrieveStructInfo(MDNField));
    } else {
      llvm_unreachable("Malformed structinfo Metadata.");
    }
  }

  return std::unique_ptr<StructInfo>(new StructInfo(Fields));
}


} // namespace mdutils
