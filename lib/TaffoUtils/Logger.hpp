#ifndef TAFFO_LOGGER_HPP
#define TAFFO_LOGGER_HPP

#include <llvm/Support/raw_ostream.h>
#include <string>

namespace taffo {

class Logger {
public:
  static Logger &getInstance();

  void log(const char *message);
  void log(const char *message, llvm::raw_ostream::Colors color);
  void log(const std::string &message);
  void log(const std::string &message, llvm::raw_ostream::Colors color);
  void log(llvm::StringRef message);
  void log(llvm::StringRef message, llvm::raw_ostream::Colors color);
  void logln(const char *message);
  void logln(const char *message, llvm::raw_ostream::Colors color);
  void logln(const std::string &message);
  void logln(const std::string &message, llvm::raw_ostream::Colors color);
  void logln(llvm::StringRef);
  void logln(llvm::StringRef, llvm::raw_ostream::Colors color);

  void setIndent(unsigned indent);
  void increaseIndent(unsigned amount = 1);
  void decreaseIndent(unsigned amount = 1);

private:
  llvm::raw_ostream &ostream;
  unsigned indent;
  bool isLineStart;

  Logger() : ostream(getOutputStream()), indent(0), isLineStart(true) {}
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  static llvm::raw_fd_ostream &getOutputStream();
  std::string getIndentString() const;
};

}

#endif // TAFFO_LOGGER_HPP
