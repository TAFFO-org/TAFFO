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

  void logValue(const llvm::Value *value, bool logParent = true);
  void logValueln(const llvm::Value *value, bool logParent = true);

  void setContextTag(const std::string &tag);
  void setContextTag(const char *tag);
  void restorePrevContextTag();

  void setIndent(unsigned indent);
  void increaseIndent(unsigned amount = 1);
  void decreaseIndent(unsigned amount = 1);

  // Log for Printable objects
  template <PrintableConcept T>
  void log(const T &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    log(value.toString(), color);
  }

  // Log for shared_ptr to Printable objects.
  template <PrintableConcept T>
  void log(const std::shared_ptr<T> &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    if (value)
      log(value->toString(), color);
    else
      log("null", color);
  }

  // Log for LLVM-printable objects
  template <LLVMPrintableConcept T>
  void log(const T &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    log(toString(value), color);
  }

  // Log for iterable objects
  template <IterableConcept T>
  void log(const T &container, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
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
  }

  // Generic fallback log (for types that are not printable in any special way)
  template <typename T>
  void log(const T &message, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    if (isLineStart)
      logIndent();
    bool useColors = ostream.is_displayed() && color != llvm::raw_ostream::Colors::RESET;
    if (useColors)
      ostream.changeColor(color, /*Bold=*/true);
    ostream << message;
    if (useColors)
      ostream.resetColor();
  }

  template <typename T>
  void logln(const T &value, llvm::raw_ostream::Colors color = llvm::raw_ostream::Colors::RESET) {
    log(value, color);
    log("\n");
    isLineStart = true;
  }

private:
  llvm::raw_ostream &ostream;
  unsigned indent;
  bool isLineStart;
  std::list<std::string> contextTagStack;

  Logger() : ostream(getOutputStream()), indent(0), isLineStart(true) {}
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  static llvm::raw_fd_ostream &getOutputStream();
  void logIndent();
};

}

#endif // TAFFO_LOGGER_HPP
