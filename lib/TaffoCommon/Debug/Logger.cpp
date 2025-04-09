#include "Logger.hpp"

#include <llvm/IR/Function.h>

using namespace llvm;
using namespace taffo;

Logger &Logger::getInstance() {
  static Logger instance;
  return instance;
}

void Logger::logValue(const Value *value, bool logParent) {
  if (auto *f = dyn_cast<Function>(value))
    log(f->getName());
  else {
    log(value);
    if (logParent) {
      if (auto *inst = dyn_cast<Instruction>(value)) {
        log(" [in fun] ", raw_ostream::Colors::BLACK);
        log(inst->getFunction()->getName());
      }
      else if (auto *arg = dyn_cast<Argument>(value)) {
        log(" [arg of fun] ", raw_ostream::Colors::BLACK);
        log(arg->getParent()->getName());
      }
    }
  }
}

void Logger::logValueln(const Value *value, bool logParent) {
  logValue(value, logParent);
  log("\n");
  isLineStart = true;
}

void Logger::setContextTag(const std::string &tag) {
  contextTagStack.push_front("[" + tag + "] ");
}

void Logger::setContextTag(const char *tag) {
  setContextTag(std::string(tag));
}

void Logger::restorePrevContextTag() {
  if (!contextTagStack.empty())
    contextTagStack.pop_front();
}

void Logger::setIndent(unsigned indent) {
  this->indent = indent;
}

void Logger::increaseIndent(unsigned amount) {
  indent += amount;
}

void Logger::decreaseIndent(unsigned amount) {
  if (indent >= amount)
    indent -= amount;
  else
    indent = 0;
}

raw_fd_ostream &Logger::getOutputStream() {
  static raw_fd_ostream ostream(1, false);
  return ostream;
}

void Logger::logIndent() {
  isLineStart = false;
  if (!contextTagStack.empty())
    log(contextTagStack.front(), raw_ostream::Colors::BLACK);
  ostream << std::string(indent, ' ');
}
