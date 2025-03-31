#include "Logger.hpp"

using namespace llvm;
using namespace taffo;

Logger &Logger::getInstance() {
  static Logger instance;
  return instance;
}

void Logger::log(const char *message) {
  if (isLineStart) {
    ostream << getIndentString() << message;
    isLineStart = false;
  }
  else
    ostream << message;
}

void Logger::log(const char *message, raw_ostream::Colors color) {
  ostream.changeColor(color, /*Bold=*/true);
  log(message);
  ostream.resetColor();
}

void Logger::log(const std::string &message) {
  log(message.c_str());
}

void Logger::log(const std::string &message, raw_ostream::Colors color) {
  log(message.c_str(), color);
}

void Logger::log(const StringRef message) {
  log(message.str().c_str());
}

void Logger::log(const StringRef message, raw_ostream::Colors color) {
  log(message.str().c_str(), color);
}

void Logger::logln(const char *message) {
  log(message);
  log("\n");
  isLineStart = true;
}

void Logger::logln(const char *message, raw_ostream::Colors color) {
  log(message, color);
  log("\n");
  isLineStart = true;
}

void Logger::logln(const std::string &message) {
  logln(message.c_str());
}

void Logger::logln(const std::string &message, raw_ostream::Colors color) {
  logln(message.c_str(), color);
}

void Logger::logln(const StringRef message) {
  logln(message.str().c_str());
}

void Logger::logln(const StringRef message, raw_ostream::Colors color) {
  logln(message.str().c_str(), color);
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

std::string Logger::getIndentString() const {
  static std::string indentString = "";
  static unsigned indent = 0;
  if (this->indent != indent) {
    indent = this->indent;
    indentString = std::string(indent, ' ');
  }
  return indentString;
}
