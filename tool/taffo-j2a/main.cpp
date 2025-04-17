#include "taffoAnnotations.hpp"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/JSON.h>

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Refactoring.h>
#include <clang/Tooling/Tooling.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace clang;
using namespace tooling;
using namespace llvm;
using namespace ast_matchers;

static llvm::cl::OptionCategory AnnotationInserterCategory("Annotation Inserter options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nSomething about this tool \n");
static cl::opt<bool> Inplace("i", cl::desc("Overwrite edited files."), cl::cat(AnnotationInserterCategory));

static cl::opt<std::string>
  AnnotationFile("f", cl::desc("Annotation file"), cl::init("./annotations.json"), cl::cat(AnnotationInserterCategory));

static cl::opt<std::string>
  AnnotationJSON("j", cl::desc("Inline annotation JSON"), cl::init(""), cl::cat(AnnotationInserterCategory));

class DeclarationPrinter : public MatchFinder::MatchCallback {
public:
  DeclarationPrinter(std::map<std::string, Replacements>& replacements, taffo::AnnotationMap& annotations)
  : replacements(replacements), annotations(annotations) {}

  void run(const MatchFinder::MatchResult& result) override {
    if (auto dec = result.Nodes.getNodeAs<VarDecl>("GlobalDecl")) {
      const std::string& declName = dec->getDeclName().getAsString();
      if (annotations.globalExists(declName)) {
        SourceRange range(dec->getSourceRange().getBegin(), dec->getSourceRange().getBegin());
        addHeadReplacement(range, *result.Context, annotations.globalToStr(declName));
      }
    }
    if (auto dec = result.Nodes.getNodeAs<VarDecl>("LocalDecl")) {
      if (!dec->isLocalVarDeclOrParm())
        return;
      const std::string& declName = dec->getDeclName().getAsString();
      if (auto fun = dyn_cast<FunctionDecl>(dec->getParentFunctionOrMethod())) {
        std::string functionName = fun->getNameInfo().getAsString();
        if (annotations.localExists(declName, functionName)) {
          SourceRange range(dec->getSourceRange().getBegin(), dec->getSourceRange().getBegin());
          addHeadReplacement(range, *result.Context, annotations.localToStr(declName, functionName));
        }
      }
    }
    if (auto dec = result.Nodes.getNodeAs<FunctionDecl>("FunctionDecl")) {
      const std::string& declName = dec->getName().str();
      if (annotations.functionExists(declName)) {
        SourceRange range(dec->getSourceRange().getBegin(), dec->getSourceRange().getBegin());

        addFunctionReplacement(range, *result.Context, annotations.functionToStr(declName));
      }
    }
  }

  void addReplacement(SourceRange Old, const ASTContext& Context, std::string ann) {
    std::string NewText;
    NewText +=
      Lexer::getSourceText(CharSourceRange::getTokenRange(Old), Context.getSourceManager(), Context.getLangOpts());

    NewText += " __attribute((annotate(\"" + ann + "\"))) ";

    tooling::Replacement R(
      Context.getSourceManager(), CharSourceRange::getTokenRange(Old), NewText, Context.getLangOpts());

    consumeError(replacements[R.getFilePath()].add(R));
  }

  void addHeadReplacement(SourceRange Old, const ASTContext& Context, std::string ann) {
    std::string NewText;
    NewText += " __attribute((annotate(\"" + ann + "\"))) ";
    NewText +=
      Lexer::getSourceText(CharSourceRange::getTokenRange(Old), Context.getSourceManager(), Context.getLangOpts());

    tooling::Replacement R(
      Context.getSourceManager(), CharSourceRange::getTokenRange(Old), NewText, Context.getLangOpts());

    consumeError(replacements[R.getFilePath()].add(R));
  }

  void addFunctionReplacement(SourceRange Old, const ASTContext& Context, std::string anno) {
    std::string NewText =
      Lexer::getSourceText(CharSourceRange::getTokenRange(Old), Context.getSourceManager(), Context.getLangOpts());
    NewText += " __attribute((annotate(\"" + anno + "\"))) ";

    tooling::Replacement R(
      Context.getSourceManager(), CharSourceRange::getTokenRange(Old), NewText, Context.getLangOpts());

    consumeError(replacements[R.getFilePath()].add(R));
  }

private:
  std::map<std::string, Replacements>& replacements;
  taffo::AnnotationMap& annotations;
};

int main(int argc, const char** argv) {
  CommonOptionsParser OptionParser(argc, argv, AnnotationInserterCategory);

  std::string str(AnnotationJSON);
  if (AnnotationJSON.empty()) {
    std::ifstream inFile;
    inFile.open(AnnotationFile);

    std::stringstream strStream;
    strStream << inFile.rdbuf();
    str = strStream.str();
  }

  auto js = llvm::json::parse(str);
  auto parsed = js->getAsArray();
  if (parsed == nullptr) {
    llvm::errs() << "Could not parse " << AnnotationFile;
    return -1;
  }
  taffo::AnnotationMap ann(*parsed);

  auto Files = OptionParser.getSourcePathList();
  tooling::RefactoringTool tool(OptionParser.getCompilations(), Files);

  DeclarationPrinter printer(tool.getReplacements(), ann);
  MatchFinder finder;

  DeclarationMatcher localDlc = varDecl(isExpansionInMainFile()).bind("LocalDecl");

  DeclarationMatcher globalDlc = varDecl(hasGlobalStorage(), isExpansionInMainFile()).bind("GlobalDecl");

  DeclarationMatcher functionDlc = functionDecl(hasBody(isExpansionInMainFile())).bind("FunctionDecl");

  finder.addMatcher(localDlc, &printer);
  finder.addMatcher(globalDlc, &printer);
  finder.addMatcher(functionDlc, &printer);
  auto factory = tooling::newFrontendActionFactory(&finder);

  if (Inplace) {
    int out = tool.runAndSave(factory.get());
    return out;
  }

  int ExitCode = tool.run(factory.get());
  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
    IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts, &DiagnosticPrinter, false);

  auto& FileMgr = tool.getFiles();
  SourceManager Sources(Diagnostics, FileMgr);
  Rewriter Rewrite(Sources, DefaultLangOptions);
  tool.applyAllReplacements(Rewrite);

  for (const auto& File : Files) {
    const auto* Entry = FileMgr.getFile(File);
    const auto ID = Sources.getOrCreateFileID(Entry, SrcMgr::C_User);
    Rewrite.getEditBuffer(ID).write(outs());
  }

  return ExitCode;
}
