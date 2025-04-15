#pragma once

#include <nlohmann/json.hpp>
#include <llvm/IR/Type.h>
#include <llvm/Support/Debug.h>
#include <sstream>

using json = nlohmann::ordered_json;

namespace taffo {

/**
 * @brief Interface for objects that can be serialized and deserialized.
 *
 * This abstract class defines the interface required for converting objects to and from JSON.
 */
class Serializable {
public:
  /**
   * @brief Serializes the object to JSON.
   *
   * @return A json object representing the serialized state.
   */
  virtual json serialize() const = 0;

  /**
   * @brief Deserializes the object from JSON.
   *
   * @param j A json object containing the serialized state.
   */
  virtual void deserialize(const json &j) = 0;

  virtual ~Serializable() = default;
};

/**
 * @brief Interface for objects that can be converted to a string.
 *
 * Classes implementing Printable must provide a string representation via toString().
 */
class Printable {
public:
  /**
   * @brief Returns a string representation of the object.
   *
   * @return A std::string representing the object.
   */
  virtual std::string toString() const = 0;

  /**
   * @brief Dumps the string representation of the object to dbgs.
   */
  void dump() const { llvm::dbgs() << toString(); }

  virtual ~Printable() = default;
};

// Trait to check if T is Printable
template <typename T>
using is_printable = std::is_base_of<Printable, std::decay_t<T>>;

// Overload of << to print Printable to std::ostream
inline std::ostream &operator<<(std::ostream &os, const Printable &obj) {
  os << obj.toString();
  return os;
}

// Overload of << to print Printable to llvm::raw_ostream
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Printable &obj) {
  os << obj.toString();
  return os;
}

// Trait to check if T has a print(raw_ostream&) method (supports raw pointers too)
template <typename T, typename = void>
struct has_llvm_print : std::false_type {};

template <typename T>
struct has_llvm_print<T, std::void_t<
  decltype(std::declval<const std::remove_pointer_t<T>&>().print(std::declval<llvm::raw_ostream&>()))>>
  : std::true_type {};

template <typename T>
inline constexpr bool has_llvm_print_v = has_llvm_print<T>::value;

// toString for any T or T* that has a `print(llvm::raw_ostream&)` method
template <typename T>
requires has_llvm_print_v<T>
std::string toString(const T &obj) {
  std::string str;
  llvm::raw_string_ostream os(str);
  if constexpr (std::is_pointer_v<T>)
    obj->print(os);
  else
    obj.print(os);
  std::string result = os.str();
  result.erase(0, result.find_first_not_of(" \t\n\r"));
  return result;
}

inline std::string formatNumber(unsigned int digits, unsigned int number) {
  std::ostringstream oss;
  oss << std::setw(int(digits)) << std::setfill('0') << number;
  return oss.str();
}

} // namespace taffo
