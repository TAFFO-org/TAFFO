#ifndef TAFFO_LOGGER_HPP
#define TAFFO_LOGGER_HPP

#include "SerializationUtils.hpp"

#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Value.h>
#include <string>
#include <list>

namespace taffo {

// Trait to detect iterable types (has begin() and end())
template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                  decltype(std::end(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

// Concept for Printable objects
template <typename T>
concept PrintableConcept = is_printable<T>::value;

// Concept for LLVMPrintable objects
template <typename T>
concept LLVMPrintableConcept = has_llvm_print_v<T> && !PrintableConcept<T>;

// Concept for iterable objects (exclude std::string and llvm::StringRef from being treated as an iterable container)
template <typename T>
concept IterableConcept = is_iterable_v<T> && !std::is_same_v<std::decay_t<T>, std::string> && !std::is_same_v<std::decay_t<T>, llvm::StringRef>;

class Logger {
public:
  static Logger &getInstance();

  Logger &logValue(const llvm::Value *value, bool logParent = true);
  Logger &logValueln(const llvm::Value *value, bool logParent = true);

  Logger &setContextTag(const std::string &tag);
  Logger &setContextTag(const char *tag);
  Logger &restorePrevContextTag();

  Logger &setColor(llvm::raw_ostream::Colors color);
  Logger &resetColor();

  Logger &setIndent(unsigned indent);
  Logger &increaseIndent(unsigned amount = 1);
  Logger &decreaseIndent(unsigned amount = 1);

  // Log for Printable objects
  template <PrintableConcept T>
  Logger &log(const T &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    log(value.toString(), color);
    return *this;
  }

  // Log for shared_ptr to Printable objects.
  template <PrintableConcept T>
  Logger &log(const std::shared_ptr<T> &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    if (value)
      log(value->toString(), color);
    else
      log("null", color);
    return *this;
  }

  // Log for LLVM-printable objects
  template <LLVMPrintableConcept T>
  Logger &log(const T &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    log(toString(value), color);
    return *this;
  }

  // Log for iterable objects
  template <IterableConcept T>
  Logger &log(const T &container, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    log("[", llvm::raw_ostream::Colors::BLACK);
    bool first = true;
    for (const auto &el : container) {
      if (!first)
        log(", ", llvm::raw_ostream::Colors::BLACK);
      else
        first = false;
      log(el, color);
    }
    log("]", llvm::raw_ostream::Colors::BLACK);
    return *this;
  }

  // Generic fallback log (for types that are not printable in any special way)
  template <typename T>
  Logger &log(const T &message, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    // Convert message to string.
    std::string s;
    llvm::raw_string_ostream oss(s);
    oss << message;
    std::string messageString = oss.str();

    llvm::raw_ostream::Colors resColor = (color == llvm::raw_ostream::Colors::RESET) ? currentColor : color;
    bool useColors = ostream.is_displayed() && resColor != llvm::raw_ostream::Colors::RESET;
    // Log the string splitting by '\n'.
    size_t start = 0;
    while (start < messageString.size()) {
      size_t pos = messageString.find('\n', start);
      std::string line = (pos == std::string::npos) ? messageString.substr(start) : messageString.substr(start, pos - start);
      if (isLineStart)
        logIndent();
      if (useColors)
        ostream.changeColor(resColor, /*Bold=*/true);
      ostream << line;
      if (useColors)
        ostream.resetColor();
      // If no newline was found, stop.
      if (pos == std::string::npos) {
        isLineStart = false;
        break;
      }
      ostream << "\n";
      isLineStart = true;
      start = pos + 1;
    }
    return *this;
  }

  template <typename T>
  Logger &logln(const T &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    log(value, color);
    ostream << "\n";
    isLineStart = true;
    return *this;
  }

  template<typename T>
  Logger &operator<<(const T &val) {
    log(val);
    return *this;
  }

  Logger &operator<<(llvm::raw_ostream::Colors color) {
    setColor(color);
    return *this;
  }

private:
  llvm::raw_ostream &ostream;
  std::list<std::string> contextTagStack;
  llvm::raw_ostream::Colors currentColor;
  unsigned indent;
  bool isLineStart;

  Logger() : ostream(getOutputStream()), currentColor(llvm::raw_ostream::Colors::RESET), indent(0), isLineStart(true) {}
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  static llvm::raw_fd_ostream &getOutputStream();
  void logIndent();
};

inline Logger &log() { return Logger::getInstance(); }

}

#endif // TAFFO_LOGGER_HPP
