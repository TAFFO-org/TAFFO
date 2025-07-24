#pragma once

#include "TransparentType.hpp"

#include <nlohmann/json.hpp>
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
  virtual void deserialize(const json& j) = 0;

  virtual ~Serializable() = default;
};

inline std::string repeatString(const std::string& str, unsigned n) {
  std::ostringstream oss;
  for (unsigned i = 0; i < n; i++)
    oss << str;
  return oss.str();
}

inline std::string formatUnsigned(unsigned digits, unsigned number) {
  std::ostringstream oss;
  oss << std::setw(int(digits)) << std::setfill('0') << number;
  return oss.str();
}

json serializeDouble(double value);
double deserializeDouble(const json& j);

json serialize(const tda::TransparentType& t);
json serialize(const tda::TransparentArrayType& t);
json serialize(const tda::TransparentStructType& t);
std::shared_ptr<tda::TransparentType> deserialize(const json& j);

} // namespace taffo
