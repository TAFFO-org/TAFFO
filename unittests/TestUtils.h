#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "TaffoInitializer/AnnotationParser.h"

/// Creates a llvm::Module object starting from a LLVM-IR string.
static std::unique_ptr<llvm::Module> makeLLVMModule(llvm::LLVMContext &Context, const std::string &code)
{
  llvm::StringRef ModuleStr(code);
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad LLVM IR?");
  return M;
}

/// Creates a FatalErrorHandler that throws an exception instead of exiting.
static void FatalErrorHandler(void *user_data, const std::string &reason, bool gen_crash_diag)
{
  throw std::runtime_error(reason.c_str());
}

/**
 * Creates and returns a global variable with the required annotations
 *
 * @param[out] ret
 * @param[in] type the type of the variable (only float for now)
 * @param[in] name the variable name
 * @param[in] anno the annotation string
 */
static void generateGlobalVariable(taffo::MultiValueMap<llvm::Value*, taffo::ValueInfo> ret, const std::string& type, const std::string& name, const std::string& anno) {
  std::string code;
  std::string filename = "filename.c";
  std::string annoLen = std::to_string(anno.length()+1);
  std::string nameLen = std::to_string(name.length()+1);
  std::string fileLen = std::to_string(filename.length()+1);

  //TODO: generalize? if needed
  std::string initVal = "0.000000e+00";
  std::string align = "4";

  code += "@" + name + " = dso_local global " + type + " " + initVal + ", align " + align + "\n";
  code += "@.str = private unnamed_addr constant [" + annoLen + R"( x i8] c")" + anno + R"(\00", section "llvm.metadata")" + "\n";
  code += "@.str.1 = private unnamed_addr constant [" + fileLen + R"( x i8] c")" + filename + R"(\00", section "llvm.metadata")" + "\n";
  code += "@llvm.global.annotations = appending global\n  [1 x { i8*, i8*, i8*, i32, i8* }]\n [{ i8*, i8*, i8*, i32, i8* } {\n";
  code += "i8* bitcast (" + type +"* @" + name + " to i8*), \n";
  code += "i8* getelemptr inbounds ([" + annoLen + " x 18], [" + annoLen + " x i8]* @.str, i32 0, i32 0),\n";
  code += "i8* getelemptr inbounds ([" + fileLen + " x 18], [" + fileLen + " x i8]* @.str1, i32 0, i32 0),\n";
  code += R"(i32 1, i8* null}], section "llvm.metadata")";
  code += "define dso_local i32 @main() #0 {ret i32 0}";

  llvm::LLVMContext C;
  std::unique_ptr<llvm::Module> M = makeLLVMModule(C, code);

  llvm::Value* v;
  taffo::ValueInfo vi;
  v = M->getGlobalVariable(name);

  taffo::AnnotationParser parser;
  parser.parseAnnotationString(code);
  vi.backtrackingDepthLeft = parser.backtracking ? parser.backtrackingDepth : 0;
  vi.target = parser.target;
  vi.metadata = parser.metadata;

  ret.insert(ret.begin(), v, vi);
}
