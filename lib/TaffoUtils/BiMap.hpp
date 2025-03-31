#ifndef TAFFO_BIMAP_HPP
#define TAFFO_BIMAP_HPP

#include <map>

/**
 * @brief Comparator for pointers to Value objects.
 *
 * This comparator compares the dereferenced objects using the provided ValueCompare.
 * It supports heterogeneous lookup.
 *
 * @tparam Value The type of the objects pointed to.
 * @tparam ValueCompare The comparator used for comparing Value objects.
 */
template <typename Value, typename ValueCompare = std::less<Value>>
struct ValuePtrCompare {
  using is_transparent = void;
  ValueCompare comp;

  bool operator()(const Value* lhs, const Value* rhs) const {
    return comp(*lhs, *rhs);
  }
  bool operator()(const Value* lhs, const Value &rhs) const {
    return comp(*lhs, rhs);
  }
  bool operator()(const Value &lhs, const Value* rhs) const {
    return comp(lhs, *rhs);
  }
};

/**
 * @brief A bidirectional map with efficient lookup by both key and value.
 *
 * This class behaves like a std::map and adds methods to look up by value and check
 * if a value is contained in the map. Both keys and values must be unique.
 *
 * @tparam Key The type of keys.
 * @tparam Value The type of mapped values.
 * @tparam KeyCompare The comparator used for keys.
 * @tparam ValueCompare The comparator used for values.
 */
template <typename Key, typename Value,
    typename KeyCompare = std::less<Key>,
    typename ValueCompare = std::less<Value>>
class BiMap {
public:
  using value_type = std::pair<const Key, Value>;
  using container_type = std::map<Key, Value, KeyCompare>;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using reverse_container_type = std::map<const Value*, Key, ValuePtrCompare<Value, ValueCompare>>;

  /**
   * @brief Inserts a key-value pair.
   *
   * @param pair The key-value pair to insert.
   * @return A pair consisting of an iterator to the inserted element (or to the element that prevented insertion)
   *         and a bool denoting whether the insertion took place.
   */
  std::pair<iterator, bool> insert(const value_type &pair) {
    auto res = keyValueMap.insert(pair);
    if (!res.second)
      return res;
    auto revRes = valueKeyMap.insert({ &(res.first->second), pair.first });
    if (!revRes.second) {
      // Rollback.
      keyValueMap.erase(res.first);
      return { keyValueMap.end(), false };
    }
    return res;
  }

  /**
   * @brief Erases an element by key.
   *
   * @param key The key of the element to erase.
   * @return The number of elements removed.
   */
  size_t erase(const Key &key) {
    auto it = keyValueMap.find(key);
    if (it == keyValueMap.end())
      return 0;
    valueKeyMap.erase(&(it->second));
    keyValueMap.erase(it);
    return 1;
  }

  /**
   * @brief Erases an element by value.
   *
   * @param value The value of the element to erase.
   * @return The number of elements removed.
   */
  size_t eraseByValue(const Value &value) {
    auto it = findByValue(value);
    if (it == keyValueMap.end())
      return 0;
    valueKeyMap.erase(&(it->second));
    keyValueMap.erase(it);
    return 1;
  }

  /**
   * @brief Finds an element by key.
   *
   * @param key The key to search for.
   * @return An iterator to the element, or keyValueMap.end() if not found.
   */
  iterator find(const Key &key) { return keyValueMap.find(key); }
  const_iterator find(const Key &key) const { return keyValueMap.find(key); }

  /**
   * @brief Finds an element by its mapped value.
   *
   * @param value The value to search for.
   * @return An iterator to the element, or keyValueMap.end() if not found.
   */
  iterator findByValue(const Value &value) {
    auto rit = valueKeyMap.find(value);
    if (rit == valueKeyMap.end())
      return keyValueMap.end();
    return keyValueMap.find(rit->second);
  }
  const_iterator findByValue(const Value &value) const {
    auto rit = valueKeyMap.find(value);
    if (rit == valueKeyMap.end())
      return keyValueMap.end();
    return keyValueMap.find(rit->second);
  }

  /**
   * @brief Checks if a given key is present.
   *
   * @param key The key to check.
   * @return True if the key is present, false otherwise.
   */
  bool contains(const Key &key) const {
    return keyValueMap.contains(key);
  }

  /**
   * @brief Checks if a given value is present.
   *
   * @param value The value to check.
   * @return True if the value is present, false otherwise.
   */
  bool containsValue(const Value &value) const {
    return valueKeyMap.contains(value);
  }

  /**
   * @brief Updates an existing key to a new key.
   *
   * If an element with oldKey exists, it is removed from the map and reinserted
   * with newKey while preserving the associated value.
   *
   * @param oldKey The key to be replaced.
   * @param newKey The new key.
   * @return True if the update succeeded, false if the oldKey was not found or insertion failed.
   */
  bool updateKey(const Key &oldKey, const Key &newKey) {
    auto it = keyValueMap.find(oldKey);
    if (it == keyValueMap.end())
      return false;
    Value value = it->second;
    auto revIt = valueKeyMap.find(&(it->second));
    keyValueMap.erase(it);
    valueKeyMap.erase(revIt);
    auto res = keyValueMap.insert({ newKey, value });
    if (!res.second) {
      // Rollback.
      auto r1 = keyValueMap.insert({ oldKey, value });
      auto r2 = valueKeyMap.insert({ &value, oldKey });
      assert(r1.second && r2.second && "Rollback failure");
      return false;
    }
    if (!updateReverseMapping(newKey, &(res.first->second), value)) {
      // Rollback.
      keyValueMap.erase(res.first);
      auto r1 = keyValueMap.insert({ oldKey, value });
      auto r2 = valueKeyMap.insert({ &value, oldKey });
      assert(r1.second && r2.second && "Rollback failure");
      return false;
    }
    return true;
  }

  /**
   * @brief Updates the key associated with an element that has the specified value.
   *
   * This overload finds the element by its mapped value and then updates its key
   * to newKey.
   *
   * @param value The value whose associated key is to be updated.
   * @param newKey The new key.
   * @return True if the update succeeded, false if the element was not found.
   */
  bool updateKeyByValue(const Value &value, const Key &newKey) {
    auto it = find_by_value(value);
    if (it == keyValueMap.end())
      return false;
    return updateKey(it->first, newKey);
  }

  /**
   * @brief Proxy class for operator[].
   *
   * This class enables assignment via operator[] to update the reverse map.
   */
  class Proxy {
  public:
    /**
     * @brief Constructs a Proxy.
     *
     * @param b Pointer to the parent BiMap.
     * @param k The key.
     * @param v Pointer to the mapped value.
     */
    Proxy(BiMap *b, const Key &k, Value *v)
        : bimap(b), key(k), val_ptr(v) {}

    /**
     * @brief Implicit conversion to Value&.
     *
     * @return Reference to the value.
     */
    operator Value&() { return *val_ptr; }

    /**
     * @brief Assignment operator.
     *
     * Updates the reverse mapping upon assignment.
     *
     * @param newValue The new value.
     * @return Reference to the updated value.
     */
    Value& operator=(const Value &newValue) {
      bimap->updateReverseMapping(key, val_ptr, newValue);
      return *val_ptr;
    }
  private:
    BiMap *bimap;
    Key key;
    Value *val_ptr;
  };

  /**
   * @brief Accesses or inserts an element by key.
   *
   * If the key does not exist, a default-constructed value is inserted.
   * Returns a Proxy that updates the reverse map upon assignment.
   *
   * @param key The key to access.
   * @return A Proxy object for assignment and value access.
   */
  Proxy operator[](const Key &key) {
    auto it = keyValueMap.find(key);
    if (it == keyValueMap.end()) {
      auto pair = std::make_pair(key, Value());
      auto res = keyValueMap.insert(pair);
      it = res.first;
      valueKeyMap.insert({ &(it->second), key });
    }
    return Proxy(this, key, &(it->second));
  }

  /**
   * @brief Returns the number of elements.
   *
   * @return The size of the BiMap.
   */
  size_t size() const { return keyValueMap.size(); }

  /**
   * @brief Checks if the BiMap is empty.
   *
   * @return True if empty, false otherwise.
   */
  bool empty() const { return keyValueMap.empty(); }

  /**
   * @brief Returns an iterator to the beginning.
   *
   * @return An iterator to the first element.
   */
  iterator begin() { return keyValueMap.begin(); }
  const_iterator begin() const { return keyValueMap.begin(); }

  /**
   * @brief Returns an iterator to the end.
   *
   * @return An iterator past the last element.
   */
  iterator end() { return keyValueMap.end(); }
  const_iterator end() const { return keyValueMap.end(); }

  /**
   * @brief Clears all elements.
   */
  void clear() {
    keyValueMap.clear();
    valueKeyMap.clear();
  }

private:
  /**
   * @brief Updates the reverse mapping for a given key and value.
   *
   * This function erases the old reverse mapping for the given value pointer, updates
   * the value, and then reinserts the reverse mapping using the new value.
   *
   * @param key The key associated with the value.
   * @param valPtr Pointer to the value to update.
   * @param newValue The new value.
   * @return True if the reverse mapping was updated successfully, false otherwise.
   */
  bool updateReverseMapping(const Key &key, Value *valPtr, const Value &newValue) {
    Value oldValue = *valPtr;
    auto revIt = valueKeyMap.find(valPtr);
    if (revIt != valueKeyMap.end())
      valueKeyMap.erase(revIt);
    *valPtr = newValue;
    auto revRes = valueKeyMap.insert({ valPtr, key });
    if (!revRes.second) {
      // Rollback.
      *valPtr = oldValue;
      auto r = valueKeyMap.insert({ valPtr, key });
      assert(r.second && "Rollback failure");
      return false;
    }
    return true;
  }

  container_type keyValueMap;
  reverse_container_type valueKeyMap;
};

#endif // TAFFO_BIMAP_HPP
