#include "Logger.hpp"

#include <llvm/IR/Function.h>

using namespace llvm;
using namespace taffo;

Logger& Logger::getInstance() {
  static Logger instance;
  return instance;
}

Logger& Logger::logValue(const Value* value, bool logParent) {
  if (auto* f = dyn_cast<Function>(value))
    log(f->getName());
  else {
    log(value);
    if (logParent) {
      if (auto* inst = dyn_cast<Instruction>(value)) {
        log(" [in fun] ", Bold);
        log(inst->getFunction()->getName());
      }
      else if (auto* arg = dyn_cast<Argument>(value)) {
        log(" [arg of fun] ", Bold);
        log(arg->getParent()->getName());
      }
    }
  }
  return *this;
}

Logger& Logger::logValueln(const Value* value, bool logParent) {
  logValue(value, logParent);
  log("\n");
  isLineStart = true;
  return *this;
}

Logger& Logger::setContextTag(const std::string& tag) {
  contextTagStack.push_front("[" + tag + "] ");
  return *this;
}

Logger& Logger::setContextTag(const char* tag) {
  setContextTag(std::string(tag));
  return *this;
}

Logger& Logger::restorePrevContextTag() {
  if (!contextTagStack.empty())
    contextTagStack.pop_front();
  return *this;
}

Logger& Logger::setColor(Color color) {
  if (color != Current)
    currentColor = color;
  return *this;
}

Logger& Logger::resetColor() {
  currentColor = Reset;
  return *this;
}

Logger& Logger::increaseIndent(unsigned amount) {
  indent += amount;
  return *this;
}

Logger& Logger::decreaseIndent(unsigned amount) {
  if (indent >= amount)
    indent -= amount;
  else
    indent = 0;
  return *this;
}

raw_fd_ostream& Logger::getOutputStream() {
  static raw_fd_ostream ostream(1, false);
  return ostream;
}

void Logger::logIndent() {
  isLineStart = false;
  ostream << std::string(indent, ' ');
  if (!contextTagStack.empty())
    log(contextTagStack.front(), Bold);
}
