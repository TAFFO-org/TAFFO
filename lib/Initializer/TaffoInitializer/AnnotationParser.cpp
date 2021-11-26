#include <cctype>
#include <climits>
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "AnnotationParser.h"


using namespace llvm;
using namespace taffo;
using namespace mdutils;


void AnnotationParser::reset()
{
  target = None;
  startingPoint = false;
  backtracking = false;
  metadata.reset();
}


bool AnnotationParser::parseAnnotationString(StringRef annstr)
{
  reset();
  sstream = std::istringstream(annstr.substr(0, annstr.size()));
  
  bool res;
  if (annstr.find('(') == StringRef::npos)
    res = parseOldSyntax();
  else
    res = parseNewSyntax();
  if (res)
    error = "";
  return res;
}


StringRef AnnotationParser::lastError()
{
  return error;
}


bool AnnotationParser::parseOldSyntax()
{
  error = "Somebody used the old syntax and they should stop.";
  bool readNumBits = true;
  std::string head;
  sstream >> head;
  if (head.find("target:") == 0) {
    target = head.substr(7); // strlen("target:") == 7
    startingPoint = true;
    sstream >> head;
  }
  if (head == "no_float" || head == "force_no_float") {
    if (head == "no_float") {
      backtracking = false;
    } else {
      backtracking = true;
      backtrackingDepth = UINT_MAX;
    }
    sstream >> head;
  }
  if (head == "range")
    readNumBits = false;
  else
    return false;
  
  mdutils::InputInfo *info = new mdutils::InputInfo(nullptr, nullptr, nullptr, true);
  metadata.reset(info);

  if (readNumBits) {
    int intbits, fracbits;
    sstream >> intbits >> fracbits;
    if (!sstream.fail()) {
      std::string signedflg;
      sstream >> signedflg;
      if (!sstream.fail() && signedflg == "unsigned") {
        info->IType.reset(new mdutils::FPType(intbits + fracbits, fracbits, false));
      } else {
        info->IType.reset(new mdutils::FPType(intbits + fracbits, fracbits, true));
      }
    }
  }

  // Look for Range info
  double Min, Max;
  sstream >> Min >> Max;
  if (!sstream.fail()) {
    info->IRange.reset(new mdutils::Range(Min, Max));
    LLVM_DEBUG(dbgs() << "Range found: [" << Min << ", " << Max << "]\n");

    // Look for initial error
    double Error;
    sstream >> Error;
    if (!sstream.fail()) {
      LLVM_DEBUG(dbgs() << "Initial error found " << Error << "\n");
      info->IError.reset(new double(Error));
    }
  }
  
  return true;
}


bool AnnotationParser::parseNewSyntax()
{
  sstream.unsetf(std::ios_base::skipws);
  char next = skipWhitespace();
  sstream.putback(next);
  
  while (next != '\0') {
    if (peek("target")) {
      std::string tgt;
      if (!expect("(")) return false;
      if (!expectString(tgt)) return false;
      if (!expect(")")) return false;
      target = tgt;
      startingPoint = true;
      
    } if (peek("errtarget")) {
      std::string tgt;
      if (!expect("(")) return false;
      if (!expectString(tgt)) return false;
      if (!expect(")")) return false;
      target = tgt;

    } else if (peek("backtracking")) {
      if (peek("(")) {
        if (!expectBoolean(backtracking)) {
          int64_t tmp;
          if (!expectInteger(tmp)) return false;
          backtrackingDepth = tmp;
          backtracking = !(backtrackingDepth == 0);
        }
        if (!expect(")")) return false;
      } else {
        backtracking = true;
        backtrackingDepth = UINT_MAX;
      }
      
    } else if (peek("struct")) {
      if (!parseStruct(metadata)) return false;
      
    } else if (peek("scalar")) {
      if (!parseScalar(metadata)) return false;
      
    } else {
      error = "Unknown identifier at character index " + std::to_string(sstream.tellg());
      return false;
    }
    
    next = skipWhitespace();
    sstream.putback(next);
  }
  
  if (metadata.get() == nullptr) {
    error = "scalar() or struct() top-level specifiers missing";
    return false;
  }
  return true;
}


bool AnnotationParser::parseScalar(std::shared_ptr<MDInfo>& thisMd)
{
  if (!expect("(")) return false;
  
  if (thisMd.get() != nullptr) {
    error = "Duplicated content definition in this context";
    return false;
  }
  InputInfo *ii = new InputInfo(nullptr, nullptr, nullptr, true);
  thisMd.reset(ii);
  
  while (!peek(")")) {
    if (peek("range")) {
      ii->IRange.reset(new Range());
      if (!expect("(")) return false;
      if (!expectReal(ii->IRange->Min)) return false;
      if (!expect(",")) return false;
      if (!expectReal(ii->IRange->Max)) return false;
      if (!expect(")")) return false;
      
    } else if (peek("type")) {
      if (!expect("(")) return false;
      bool isSignd = true;
      int64_t total, frac;
      if (!peek("signed")) {
        if (peek("unsigned")) {
          isSignd = false;
        }
      }
      if (!expectInteger(total)) return false;
      if (!expectInteger(frac)) return false;
      if (!expect(")")) return false;
      ii->IType.reset(new FPType(total, frac, isSignd));
      
    } else if (peek("error")) {
      ii->IError = std::make_shared<double>(0);
      if (!expect("(")) return false;
      if (!expectReal(*(ii->IError))) return false;
      if (!expect(")")) return false;
      
    } else if (peek("disabled")) {
      ii->IEnableConversion = false;
    } else if (peek("final")) {
      ii->IFinal = true;
    } else {
      error = "Unknown identifier at character index " + std::to_string(sstream.tellg());
      return false;
    }
  }
  return true;
}


bool AnnotationParser::parseStruct(std::shared_ptr<MDInfo>& thisMd)
{
  if (!expect("[")) return false;
  if (thisMd.get() != nullptr) {
    error = "Duplicated content definition in this context";
    return false;
  }
  std::vector<std::shared_ptr<MDInfo>> elems;
  
  bool first = true;
  while (!peek("]")) {
    if (first) {
      first = false;
    } else {
      if (!expect(",")) return false;
    }
    
    if (peek("scalar")) {
      std::shared_ptr<MDInfo> tmp;
      if (!parseScalar(tmp)) return false;
      elems.push_back(tmp);
      
    } else if (peek("struct")) {
      std::shared_ptr<MDInfo> tmp;
      if (!parseStruct(tmp)) return false;
      elems.push_back(tmp);
      
    } else if (peek("void")) {
      elems.push_back(nullptr);
      
    } else {
      error = "Unknown identifier at character index " + std::to_string(sstream.tellg());
      return false;
    }
  }
  
  if (elems.size() == 0) {
    error = "Empty structures not allowed";
    return false;
  }
  StructInfo *si = new StructInfo(elems);
  thisMd.reset(si);
  return true;
}


char AnnotationParser::skipWhitespace()
{
  char tmp = '\0';
  sstream >> tmp;
  while (!sstream.eof() && tmp != '\0' && (isblank(tmp) || iscntrl(tmp)))
    sstream >> tmp;
  if (sstream.eof())
    return '\0';
  return tmp;
}


bool AnnotationParser::expect(std::string kw)
{
  char next = skipWhitespace();
  error = "Expected " + kw + " at character index " + std::to_string((int)(sstream.tellg())-1);
  if (next == '\0')
    return false;
  int i = 0;
  while (i < kw.size() && next != '\0' && next == kw[i]) {
    i++;
    sstream >> next;
  }
  sstream.putback(next);
  return i == kw.size();
}


bool AnnotationParser::expectString(std::string& res)
{
  char next = skipWhitespace();
  error = "Expected string at character index " + std::to_string((int)(sstream.tellg())-1);
  res = "";
  if (next != '\'')
    return false;
  sstream >> next;
  while (next != '\'' && next != '\0') {
    if (next == '@') {
      sstream >> next;
      if (next != '@' && next != '\'')
        return false;
    }
    res.append(&next, 1);
    sstream >> next;
  }
  if (next == '\'')
    return true;
  return false;
}


bool AnnotationParser::expectInteger(int64_t& res)
{
  char next = skipWhitespace();
  error = "Expected integer at character index " + std::to_string((int)(sstream.tellg())-1);
  bool neg = false;
  int base = 10;
  if (next == '+') {
    sstream >> next;
  } else if (next == '-') {
    neg = true;
    sstream >> next;
  }
  if (next == '0') {
    base = 8;
    sstream >> next;
    if (next == 'x') {
      base = 16;
      sstream >> next;
    }
  }
  if (!isdigit(next))
    return false;
  res = 0;
  while (isdigit(next) || (base == 16 ? isxdigit(next) : false)) {
    res *= base;
    if (next > '9')
      res += toupper(next) - 'A' + 10;
    else
      res += next - '0';
    sstream >> next;
  }
  sstream.putback(next);
  if (neg)
    res = -res;
  return true;
}


bool AnnotationParser::expectReal(double& res)
{
  char next = skipWhitespace();
  error = "Expected real at character index " + std::to_string((int)(sstream.tellg())-1);
  sstream.putback(next);
  std::streampos pos = sstream.tellg();
  sstream >> res;
  if (sstream.fail()) {
    sstream.clear();
    sstream.seekg(pos);
    return false;
  }
  return true;
}


bool AnnotationParser::expectBoolean(bool& res)
{
  char next = skipWhitespace();
  error = "Expected boolean at character index " + std::to_string((int)(sstream.tellg())-1);
  sstream.putback(next);
  if (peek("true") || peek("yes")) {
    res = true;
    return true;
  } else if (peek("false") || peek("no")) {
    res = false;
    return true;
  }
  return false;
}
