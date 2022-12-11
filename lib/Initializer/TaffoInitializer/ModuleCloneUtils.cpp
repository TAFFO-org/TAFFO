#include "ModuleCloneUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <unordered_map>

std::unique_ptr<llvm::Module> getModule(llvm::StringRef Filename, llvm::LLVMContext &cntx)
{
  llvm::SMDiagnostic diagnostic;
  return llvm::parseIRFile(Filename, diagnostic, cntx);
}


//Clone all global but not the one that start with hero as they are target dependant (contains assembly code)
void cloneGlobalVariable(llvm::Module &host_module, llvm::Module &dev_module, llvm::ValueToValueMapTy &GtoG, llvm::Twine prefix)
{

  using namespace llvm;
  //Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (const llvm::GlobalVariable &I : dev_module.globals()) {
    if (I.getName().startswith("llvm.used"))
      continue;
    llvm::GlobalVariable *NewGV = new GlobalVariable(
        host_module, I.getValueType(), I.isConstant(), I.getLinkage(),
        (llvm::Constant *)nullptr, prefix + I.getName(), (llvm::GlobalVariable *)nullptr,
        I.getThreadLocalMode(), I.getType()->getAddressSpace());
    NewGV->copyAttributesFrom(&I);
    GtoG[&I] = NewGV;
  }

  // Loop over the functions in the module, making external functions as before
  for (const Function &I : dev_module) {
    // if (I.getName().startswith("hero_"))
    //   continue;
    Function *NF = Function::Create(cast<FunctionType>(I.getValueType()), I.getLinkage(),
                                    I.getAddressSpace(), prefix + I.getName(), &host_module);
    NF->copyAttributesFrom(&I);
    GtoG[&I] = NF;
  }


  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (const GlobalVariable &G : dev_module.globals()) {
    if (G.getName().startswith("llvm.used"))
      continue;
    GlobalVariable *GV = cast<GlobalVariable>(GtoG[&G]);

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    G.getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(MD.first, *MapMetadata(MD.second, GtoG));

    if (G.isDeclaration())
      continue;
    if (G.hasInitializer())
      GV->setInitializer(MapValue(G.getInitializer(), GtoG));
  }

  // Similarly, copy over function bodies now...
  //
  for (const Function &I : dev_module) {
    // if (I.getName().startswith("hero_"))
    //   continue;
    Function *F = cast<Function>(GtoG[&I]);

    if (I.isDeclaration()) {
      // Copy over metadata for declarations since we're not doing it below in
      // CloneFunctionInto().
      SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
      I.getAllMetadata(MDs);
      for (auto MD : MDs)
        F->addMetadata(MD.first, *MapMetadata(MD.second, GtoG));
      continue;
    }

    Function::arg_iterator DestI = F->arg_begin();
    for (const Argument &J : I.args()) {
      DestI->setName(J.getName());
      GtoG[&J] = &*DestI++;
    }

    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(F, &I, GtoG, true,
                      Returns);

    if (I.hasPersonalityFn())
      F->setPersonalityFn(MapValue(I.getPersonalityFn(), GtoG));
  }


  // // And named metadata....
  // for (const NamedMDNode &NMD : dev_module.named_metadata()) {
  //   NamedMDNode *NewNMD = host_module.getOrInsertNamedMetadata(NMD.getName());
  //   for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
  //     NewNMD->addOperand(MapMetadata(NMD.getOperand(i), GtoG));
  // }
}

std::unique_ptr<llvm::Module> cloneModuleInto(llvm::StringRef Filename, llvm::Module &host_module, llvm::Twine prefix)
{
  if (Filename.find('/') != llvm::StringRef::npos) {
    Filename = Filename.split('/').second;
  }

  auto new_module = getModule(Filename, host_module.getContext());
  llvm::ValueToValueMapTy GtoG{};
  cloneGlobalVariable(host_module, *new_module, GtoG, prefix);
  return new_module;
}
