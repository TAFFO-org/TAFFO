#include "AnnotationParser.hpp"

#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <cctype>
#include <memory>

using namespace taffo;
using namespace llvm;

#define DEBUG_TYPE "taffo-init"

void AnnotationParser::reset() {
  startingPoint = false;
  backtracking = false;
  valueInfo.reset();
}

bool AnnotationParser::parseAnnotationString(StringRef annotationStr, Type *type) {
  reset();
  stringStream = std::istringstream((annotationStr.substr(0, annotationStr.size())).str());

  bool res;
  if (annotationStr.find('(') == StringRef::npos && !annotationStr.contains("struct"))
    res = parseOldSyntax(type);
  else
    res = parseNewSyntax(type);
  if (res)
    error = "";
  return res;
}

StringRef AnnotationParser::lastError() {
  return error;
}

bool AnnotationParser::parseOldSyntax(Type *type) {
  error = "Somebody used the old syntax and they should stop.";
  bool readNumBits = true;
  std::optional<std::string> target;
  std::string head;
  stringStream >> head;
  if (head.find("target:") == 0) {
    target = head.substr(7); // strlen("target:") == 7
    startingPoint = true;
    stringStream >> head;
  }
  if (head == "no_float" || head == "force_no_float") {
    if (head == "no_float") {
      backtracking = false;
    } else {
      backtracking = true;
      backtrackingDepth = std::numeric_limits<unsigned int>::max();
    }
    stringStream >> head;
  }
  if (head == "range")
    readNumBits = false;
  else
    return false;

  auto info = std::make_shared<ScalarInfo>(type, nullptr, nullptr, nullptr, true);
  info->target = target;
  valueInfo = info;

  if (readNumBits) {
    int intbits, fracbits;
    stringStream >> intbits >> fracbits;
    if (!stringStream.fail()) {
      std::string signedflg;
      stringStream >> signedflg;
      if (!stringStream.fail() && signedflg == "unsigned")
        info->numericType = std::make_shared<FixpType>(intbits + fracbits, fracbits, false);
      else
        info->numericType = std::make_shared<FixpType>(intbits + fracbits, fracbits, true);

    }
  }

  // Look for Range info
  double Min, Max;
  stringStream >> Min >> Max;
  if (!stringStream.fail()) {
    info->range = std::make_shared<Range>(Min, Max);
    LLVM_DEBUG(dbgs() << "Range found: [" << Min << ", " << Max << "]\n");

    // Look for initial error
    double Error;
    stringStream >> Error;
    if (!stringStream.fail()) {
      LLVM_DEBUG(dbgs() << "Initial error found " << Error << "\n");
      info->error = std::make_shared<double>(Error);
    }
  }

  return true;
}


bool AnnotationParser::parseNewSyntax(Type *type) {
  stringStream.unsetf(std::ios_base::skipws);
  char next = skipWhitespace();
  stringStream.putback(next);

  std::optional<std::string> target;
  std::optional<std::string> bufferId;

  while (next != '\0') {
    if (peek("target")) {
      std::string tgt;
      if (!expect("("))
        return false;
      if (!expectString(tgt))
        return false;
      if (!expect(")"))
        return false;
      target = tgt;
      startingPoint = true;

    } else if (peek("errtarget")) {
      std::string tgt;
      if (!expect("("))
        return false;
      if (!expectString(tgt))
        return false;
      if (!expect(")"))
        return false;
      target = tgt;

    } else if (peek("backtracking")) {
      if (peek("(")) {
        if (!expectBoolean(backtracking)) {
          int64_t tmp;
          if (!expectInteger(tmp))
            return false;
          backtrackingDepth = tmp;
          backtracking = backtrackingDepth != 0;
        }
        if (!expect(")"))
          return false;
      } else {
        backtracking = true;
        backtrackingDepth = std::numeric_limits<unsigned int>::max();
      }

    } else if (peek("struct")) {
      if (!parseStruct(valueInfo, type))
        return false;

    } else if (peek("scalar")) {
      if (!parseScalar(valueInfo, type))
        return false;

    } else if (peek("bufferid")) {
      std::string buffId;
      if (!expect("("))
        return false;
      if (!expectString(buffId))
        return false;
      if (!expect(")"))
        return false;
      bufferId = buffId;

    } else {
      error = "Unknown identifier at character index " + std::to_string(stringStream.tellg());
      return false;
    }

    next = skipWhitespace();
    stringStream.putback(next);
  }

  if (valueInfo == nullptr) {
    error = "scalar() or struct() top-level specifiers missing";
    return false;
  }

  valueInfo->target = target;
  valueInfo->bufferId = bufferId;
  return true;
}


bool AnnotationParser::parseScalar(std::shared_ptr<ValueInfo> &thisValueInfo, Type *type) {
  if (!expect("("))
    return false;

  if (thisValueInfo != nullptr) {
    error = "Duplicated content definition in this context";
    return false;
  }
  auto *scalarInfo = new ScalarInfo(type, nullptr, nullptr, nullptr, true);
  thisValueInfo.reset(scalarInfo);

  while (!peek(")")) {
    if (peek("range")) {
      scalarInfo->range = std::make_shared<Range>();
      if (!expect("("))
        return false;
      if (!expectReal(scalarInfo->range->Min))
        return false;
      if (!expect(","))
        return false;
      if (!expectReal(scalarInfo->range->Max))
        return false;
      if (!expect(")"))
        return false;

    } else if (peek("type")) {
      if (!expect("("))
        return false;
      bool isSignd = true;
      int64_t total, frac;
      if (!peek("signed")) {
        if (peek("unsigned")) {
          isSignd = false;
        }
      }
      if (!expectInteger(total))
        return false;
      if (total <= 0) {
        error = "Fixed point data type must have a positive bit size";
        return false;
      }
      if (!expectInteger(frac))
        return false;
      if (!expect(")"))
        return false;
      scalarInfo->numericType = std::make_shared<FixpType>(total, frac, isSignd);

    } else if (peek("error")) {
      scalarInfo->error = std::make_shared<double>(0);
      if (!expect("("))
        return false;
      if (!expectReal(*(scalarInfo->error)))
        return false;
      if (!expect(")"))
        return false;

    } else if (peek("disabled")) {
      scalarInfo->conversionEnabled = false;
    } else if (peek("final")) {
      scalarInfo->final = true;
    } else {
      error = "Unknown identifier at character index " + std::to_string(stringStream.tellg());
      return false;
    }
  }
  return true;
}

bool AnnotationParser::parseStruct(std::shared_ptr<ValueInfo> &thisValueInfo, Type *type) {
  auto *structType = dyn_cast<StructType>(type);
  if (!structType) {
    std::string errStr;
    raw_string_ostream ss(errStr);
    ss << "Typechecking failed: expected an LLVM struct type but got " << *type;
    error = ss.str();
    return false;
  }

  if (!expect("["))
    return false;
  if (thisValueInfo != nullptr) {
    error = "Duplicated content definition in this context";
    return false;
  }
  unsigned int numFields = structType->getNumElements();
  std::vector<std::shared_ptr<ValueInfo>> fields;
  fields.reserve(numFields);

  bool first = true;
  unsigned int currentField = 0;
  while (!peek("]")) {
    if (first) {
      first = false;
    } else {
      if (!expect(","))
        return false;
    }

    if (currentField >= numFields) {
      std::string errStr;
      raw_string_ostream ss(errStr);
      ss << "Typechecking failed: " << (currentField + 1)
         << " fields specified but only " << numFields
         << " fields present in LLVM struct type " << *structType;
      error = ss.str();
      return false;
    }

    if (peek("scalar")) {
      std::shared_ptr<ValueInfo> tmp;
      if (!parseScalar(tmp, structType->getElementType(currentField)))
        return false;
      fields.push_back(tmp);
      currentField++;

    } else if (peek("struct")) {
      std::shared_ptr<ValueInfo> tmp;
      if (!parseStruct(tmp, structType->getElementType(currentField)))
        return false;
      fields.push_back(tmp);
      currentField++;

    } else if (peek("void")) {
      fields.push_back(nullptr);
      currentField++;

    } else {
      error = "Unknown identifier at character index " + std::to_string(stringStream.tellg());
      return false;
    }
  }

  if (currentField < numFields) {
    std::string errStr;
    raw_string_ostream ss(errStr);
    ss << "Typechecking failed: only " << currentField
       << " fields specified but " << numFields
       << " fields are expected in LLVM struct type " << *structType;
    error = ss.str();
    return false;
  }

  if (fields.empty()) {
    error = "Empty structures not allowed";
    return false;
  }
  auto *structInfo = new StructInfo(structType, fields);
  thisValueInfo.reset(structInfo);
  return true;
}

char AnnotationParser::skipWhitespace() {
  char tmp = '\0';
  stringStream >> tmp;
  while (!stringStream.eof() && tmp != '\0' && (isblank(tmp) || iscntrl(tmp)))
    stringStream >> tmp;
  if (stringStream.eof())
    return '\0';
  return tmp;
}

bool AnnotationParser::expect(std::string kw) {
  char next = skipWhitespace();
  error = "Expected " + kw + " at character index " + std::to_string((int)(stringStream.tellg()) - 1);
  if (next == '\0')
    return false;
  size_t i = 0;
  while (i < kw.size() && next != '\0' && next == kw[i]) {
    i++;
    stringStream >> next;
  }
  stringStream.putback(next);
  return i == kw.size();
}

bool AnnotationParser::expectString(std::string &res) {
  char next = skipWhitespace();
  error = "Expected string at character index " + std::to_string((int)(stringStream.tellg()) - 1);
  res = "";
  if (next != '\'')
    return false;
  stringStream >> next;
  while (next != '\'' && next != '\0') {
    if (next == '@') {
      stringStream >> next;
      if (next != '@' && next != '\'')
        return false;
    }
    res.append(&next, 1);
    stringStream >> next;
  }
  if (next == '\'')
    return true;
  return false;
}

bool AnnotationParser::peek(std::string kw) {
  std::streampos pos = stringStream.tellg();
  bool res;
  if (!(res = expect(kw))) {
    stringStream.clear();
    stringStream.seekg(pos);
  }
  return res;
}

bool AnnotationParser::expectInteger(int64_t &res) {
  char next = skipWhitespace();
  error = "Expected integer at character index " + std::to_string((int)(stringStream.tellg()) - 1);
  bool neg = false;
  int base = 10;
  if (next == '+') {
    stringStream >> next;
  } else if (next == '-') {
    neg = true;
    stringStream >> next;
  }
  if (next == '0') {
    base = 8;
    stringStream >> next;
    if (next == 'x') {
      base = 16;
      stringStream >> next;
      if (!isdigit(next))
        return false;
    }
  } else if (!isdigit(next))
    return false;
  res = 0;
  while (true) {
    if (base <= 10 && (next < '0' || '0'+base-1 < next))
      break;
    else if (base == 16 && !isxdigit(next))
      break;
    res *= base;
    if (next > '9')
      res += toupper(next) - 'A' + 10;
    else
      res += next - '0';
    stringStream >> next;
  }
  stringStream.putback(next);
  if (neg)
    res = -res;
  return true;
}

bool AnnotationParser::expectReal(double &res) {
  char next = skipWhitespace();
  error = "Expected real at character index " + std::to_string((int)(stringStream.tellg()) - 1);
  stringStream.putback(next);
  std::streampos pos = stringStream.tellg();
  stringStream >> res;
  if (stringStream.fail()) {
    stringStream.clear();
    stringStream.seekg(pos);
    return false;
  }
  return true;
}

bool AnnotationParser::expectBoolean(bool &res) {
  char next = skipWhitespace();
  error = "Expected boolean at character index " + std::to_string((int)(stringStream.tellg()) - 1);
  stringStream.putback(next);
  if (peek("true") || peek("yes")) {
    res = true;
    return true;
  } else if (peek("false") || peek("no")) {
    res = false;
    return true;
  }
  return false;
}
