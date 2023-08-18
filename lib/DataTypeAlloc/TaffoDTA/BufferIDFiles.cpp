#include "BufferIDFiles.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include <fstream>

using namespace llvm;
using namespace tuner;
using namespace mdutils;


static std::unique_ptr<TType> ReadTypeFromYAMLNode(SourceMgr& SM, yaml::Node *Node)
{
  yaml::MappingNode *MapNode = dyn_cast<yaml::MappingNode>(Node);
  if (!MapNode) {
    SM.PrintMessage(Node->getSourceRange().Start, SourceMgr::DiagKind::DK_Error, "Expected mapping specifying the type");
    return std::unique_ptr<TType>(nullptr);
  }
  
  std::map<std::string, std::string> Map;
  for (auto& Item: *MapNode) {
    yaml::ScalarNode *Key = dyn_cast<yaml::ScalarNode>(Item.getKey());
    if (!Key) {
      SM.PrintMessage(Item.getKey()->getSourceRange().Start, SourceMgr::DiagKind::DK_Error, "Expected scalar");
      return std::unique_ptr<TType>(nullptr);
    }
    SmallVector<char> KeyStorage;
    StringRef KeyStr = Key->getValue(KeyStorage);

    yaml::ScalarNode *Value = dyn_cast<yaml::ScalarNode>(Item.getValue());
    if (!Value) {
      SM.PrintMessage(Item.getValue()->getSourceRange().Start, SourceMgr::DiagKind::DK_Error, "Expected scalar");
      return std::unique_ptr<TType>(nullptr);
    }
    SmallVector<char> ValueStorage;
    StringRef ValueStr = Value->getValue(ValueStorage);

    Map[KeyStr.str()] = ValueStr.str();
  }

  int ClassID = atoi(Map["Class"].c_str());
  if (ClassID == TType::K_FPType) {
    int SWidth = atoi(Map["SWidth"].c_str());
    int PointPos = atoi(Map["PointPos"].c_str());
    return std::unique_ptr<TType>(new FPType(SWidth, PointPos));
  }
  
  if (ClassID == TType::K_FloatType) {
    FloatType::FloatStandard FloatStandard = (FloatType::FloatStandard)atoi(Map["FloatStandard"].c_str());
    double GreatestNumber = atof(Map["GreatestNumber"].c_str());
    return std::unique_ptr<TType>(new FloatType(FloatStandard, GreatestNumber));
  }
  
  SM.PrintMessage(Node->getSourceRange().Start, SourceMgr::DiagKind::DK_Error, "Invalid Class");
  return std::unique_ptr<TType>(nullptr);
}

static bool ReadBufferIDFileImpl(std::string Fn, BufferIDTypeMap& OutMap)
{
  SourceMgr SM;

  auto MBOrErr = MemoryBuffer::getFile(Twine(Fn));
  if (!MBOrErr) {
    SM.PrintMessage(errs(), SMDiagnostic(Fn, SourceMgr::DiagKind::DK_Error, "Error reading file"));
    return false;
  }
  yaml::Stream YamlStm(MemoryBufferRef(*(MBOrErr->get())), SM);
  yaml::Document& Doc = *(YamlStm.begin());
  yaml::Node *RootN = Doc.getRoot();

  yaml::MappingNode *RootMap = dyn_cast<yaml::MappingNode>(RootN);
  if (!RootMap) {
    SM.PrintMessage(RootN->getSourceRange().Start, SourceMgr::DiagKind::DK_Error, "Expected mapping, one for buffer ID tag");
    return false;
  }

  for (auto& Item: *RootMap) {
    yaml::ScalarNode *Key = dyn_cast<yaml::ScalarNode>(Item.getKey());
    if (!Key) {
      SM.PrintMessage(Key->getSourceRange().Start, SourceMgr::DiagKind::DK_Error, "Expected bufferID tag string");
      return false;
    }
    SmallVector<char> KeyStorage;
    StringRef KeyStr = Key->getValue(KeyStorage);

    std::unique_ptr<TType> T = ReadTypeFromYAMLNode(SM, Item.getValue());
    if (T.get()) {
      OutMap[KeyStr.str()] = std::move(T);
    } else {
      return false;
    }
  }

  return true;
}

void tuner::ReadBufferIDFile(std::string Fn, BufferIDTypeMap& OutMap)
{
  // TODO: "replace most of report_fatal_error with llvm::Error and propagate them correctly"
  // Not my todo, it's a quote from a comment to Phabricator patchset D67847 :)
  // Is llvm::Error used anywhere btw?
  if (!ReadBufferIDFileImpl(Fn, OutMap))
    report_fatal_error("Error reading bufferID file!");
}


void tuner::WriteBufferIDFile(std::string Fn, BufferIDTypeMap& Map)
{
  std::ofstream Stm(Fn);
  Stm << "---" << std::endl;
  for (auto& Pair: Map) {
    Stm << "\"" << Pair.first << "\" : ";
    TType *T = Pair.second.get();
    if (FloatType *FloatT = dyn_cast<FloatType>(T)) {
      Stm << "{ " <<
        "Class: " << FloatT->getKind() << ", " <<
        "FloatStandard: " << FloatT->getStandard() << ", " <<
        "GreatestNumber: " << FloatT->getGreatestNumber() << " }";
    } else if (FPType *FPT = dyn_cast<FPType>(T)) {
      Stm << "{ " <<
        "Class: " << FPT->getKind() << ", " <<
        "SWidth: " << FPT->getSWidth() << ", " <<
        "PointPos: " << FPT->getPointPos() << " }";
    } else {
      llvm_unreachable("unknown type class");
    }
    Stm << std::endl;
  }
}

