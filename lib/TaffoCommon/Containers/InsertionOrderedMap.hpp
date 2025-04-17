#pragma once

#include <list>
#include <unordered_map>

/**
 * @brief A map that preserves insertion order.
 *
 * InsertionOrderedMap provides fast lookup by key via an unordered_map
 * while preserving the order in which keys (of type Key) are inserted using a std::list.
 * When an element with an existing key is inserted, the new value (of type Value)
 * replaces the old and the key is moved to the end of the insertion order, or to a specified position.
 *
 * @tparam Key The type of the keys.
 * @tparam Value The type of the mapped values.
 */
template <typename Key, typename Value>
class InsertionOrderedMap {
public:
  /// The pair type stored in the map.
  using pair_type = std::pair<const Key, Value>;

  template <bool IsConst>
  class iterator;

  using iterator_type = iterator<false>;
  using const_iterator_type = iterator<true>;

private:
  /**
   * @brief List storing keys in insertion order.
   *
   * This list maintains the order in which keys were inserted,
   * and allows moving a key from its current position to a new position in O(1) time.
   */
  std::list<Key> order;

  /**
   * @brief Underlying map from key to (value, iterator into the order list).
   *
   * The stored iterator points to the key's location in the order list.
   */
  std::unordered_map<Key, std::pair<Value, typename std::list<Key>::iterator>> map;

public:
  /**
   * @brief Proxy returned by the iterator's operator->
   *
   * This proxy holds references to the current key and its mapped value.
   * The mapped value is optionally const depending on the template parameter.
   *
   * @tparam IsConst If true, the mapped value is returned as a const reference
   */
  template <bool IsConst>
  struct iterator_proxy {
    using value_type = std::conditional_t<IsConst, const Value, Value>;

    const Key& first;
    value_type& second;

    iterator_proxy(const Key& first, value_type& second)
    : first(first), second(second) {}
    iterator_proxy(const iterator_proxy& other)
    : first(other.first), second(other.second) {}

    // Dummy operator to allow the assignment of iterators
    iterator_proxy& operator=(const iterator_proxy&) { return *this; }
  };

  /**
   * @brief Iterator for InsertionOrderedMap that iterates in insertion order
   *
   * This iterator wraps a const_iterator over the order list and retrieves references
   * to the corresponding pair in the underlying unordered_map.
   *
   * @tparam IsConst If true, the iterator yields const references to the elements
   */
  template <bool IsConst>
  class iterator {
  public:
    using list_iterator =
      std::conditional_t<IsConst, typename std::list<Key>::const_iterator, typename std::list<Key>::iterator>;
    using iterator_proxy_type = iterator_proxy<IsConst>;
    using value_type = std::conditional_t<IsConst, const Value, Value>;
    using pair_type = std::pair<const Key, value_type>;
    using container_type = std::conditional_t<IsConst, const InsertionOrderedMap, InsertionOrderedMap>;

    /**
     * @brief Constructs an iterator.
     * @param it Iterator over the order list.
     * @param container Pointer to the parent InsertionOrderedMap.
     */
    iterator(list_iterator it, container_type* container)
    : it(it), container(container) {}

    /// Pre-increment operator.
    iterator& operator++() {
      ++it;
      return *this;
    }

    /// Post-increment operator.
    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    /// Pre-decrement operator.
    iterator& operator--() {
      --it;
      return *this;
    }

    /// Post-decrement operator.
    iterator operator--(int) {
      iterator tmp = *this;
      --(*this);
      return tmp;
    }

    /// Equality comparison.
    template <typename IterType>
    bool operator==(const IterType& other) const {
      return it == other.it;
    }

    /// Inequality comparison.
    template <typename IterType>
    bool operator!=(const IterType& other) const {
      return !(*this == other);
    }

    /**
     * @brief Less-than comparison.
     *
     * Uses std::distance on the underlying list to compare positions.
     * Note: This operation is O(n).
     *
     * @param other The iterator to compare to.
     * @return True if this iterator precedes other.
     */
    template <typename IterType>
    bool operator<(const IterType& other) const {
      return std::distance(container->order.begin(), it) < std::distance(container->order.begin(), other.it);
    }

    /**
     * @brief Greater-than comparison.
     *
     * Uses std::distance on the underlying list to compare positions.
     * Note: This operation is O(n).
     *
     * @param other The iterator to compare to.
     * @return True if this iterator follows other.
     */
    template <typename IterType>
    bool operator>(const IterType& other) const {
      return std::distance(container->order.begin(), it) > std::distance(container->order.begin(), other.it);
    }

    /**
     * @brief Dereferences the iterator
     *
     * Returns a proxy that holds references to the current key and its mapped value
     *
     * @return An iterator_proxy holding references to the current element
     */
    pair_type operator*() const {
      const Key& key = *it;
      value_type& value = container->map.at(*it).first;
      return {key, value};
    }

    /**
     * @brief Provides pointer-like access to the current element
     *
     * Updates an internal mutable proxy with references to the current element and returns its address.
     *
     * @return Pointer to an iterator_proxy representing the current element
     */
    iterator_proxy_type* operator->() const {
      const Key& key = *it;
      value_type& value = container->map.at(*it).first;
      proxy.emplace(key, value);
      return &proxy.value();
    }

  private:
    list_iterator it;
    container_type* container;
    mutable std::optional<iterator_proxy_type> proxy;
    friend class InsertionOrderedMap;
  };

  /**
   * @brief Checks whether the map contains the given key.
   *
   * @param key The key to search for.
   * @return true if the key exists in the map, false otherwise.
   */
  bool contains(const Key& key) const { return map.find(key) != map.end(); }

  /**
   * @brief Inserts a key-value pair into the map at the end.
   *
   * If the key already exists, no insertion is performed.
   *
   * @param key The key to insert.
   * @param value The value to associate with the key.
   * @return A pair consisting of an iterator to the element (newly inserted or already existing)
   *         and a bool that is true if the insertion took place.
   */
  std::pair<iterator_type, bool> insert(const Key& key, const Value& value) {
    auto it = map.find(key);
    if (it != map.end()) {
      // Key exists.
      return {iterator_type(it->second.second, this), false};
    }
    else {
      // New key.
      order.push_back(key);
      auto list_it = std::prev(order.end());
      auto result = map.insert({
        key, {value, list_it}
      });
      return {iterator_type(result.first->second.second, this), true};
    }
  }

  /**
   * @brief Inserts a pair into the map at the end.
   *
   * If the key already exists, no insertion is performed.
   *
   * @param p The pair to insert.
   * @return A pair consisting of an iterator to the element (newly inserted or already existing)
   *         and a bool that is true if the insertion took place.
   */
  std::pair<iterator_type, bool> insert(const pair_type& p) { return insert(p.first, p.second); }

  /**
   * @brief Inserts elements from a range into the map at the end
   *
   * Inserts each element from the range [first, last) into the map.
   * If a key already exists, no insertion is performed for that key.
   *
   * @param first Iterator to the first element in the range
   * @param last Iterator past the last element in the range
   */
  template <typename IterType>
  void insert(IterType first, IterType last) {
    for (; first != last; first++)
      insert(first->first, first->second);
  }

  /**
   * @brief Inserts a key-value pair into the map at the specified position
   *
   * The position is specified by an iterator (which must be in the range [begin(), end()]).
   * If the key already exists, its value is updated and the key is moved to the specified position.
   *
   * @param pos Iterator specifying the position at which to insert
   * @param key The key to insert
   * @param value The value to associate with the key
   * @return An iterator to the inserted (or updated) element
   */
  template <typename IterType>
  iterator_type insertAt(IterType pos, const Key& key, const Value& value) {
    auto pos_list = pos.it; // underlying list iterator from custom iterator
    auto it = map.find(key);
    if (it != map.end()) {
      // Key exists: update value.
      it->second.first = value;
      // Remove key from its current position.
      order.erase(it->second.second);
      // Insert key at specified position.
      auto new_it = order.insert(pos_list, key);
      it->second.second = new_it;
      return iterator_type(new_it, this);
    }
    else {
      // New key: insert at specified position.
      auto new_it = order.insert(pos_list, key);
      auto result = map.insert({
        key, {value, new_it}
      });
      return iterator_type(result.first->second.second, this);
    }
  }

  /**
   * @brief Inserts elements from a range into the map at the specified position
   *
   * Inserts each element from the range [first, last) into the map starting at the position specified by pos.
   * The inserted elements appear in the same order as in the range.
   *
   * @param pos Iterator specifying the insertion position for the first element
   * @param first Iterator to the first element in the range
   * @param last Iterator past the last element in the range
   */
  template <typename IterType>
  void insertAt(IterType pos, IterType first, IterType last) {
    for (; first != last; ++first) {
      pos = insertAt(pos, first->first, first->second);
      ++pos;
    }
  }

  /**
   * @brief Accesses the mapped value corresponding to the given key.
   *
   * If the key is not present throw an error
   *
   * @param key The key to access.
   * @return Reference to the associated Value.
   */
  Value& at(const Key& key) {
    auto it = map.find(key);
    assert(it != map.end() && "Value must be present");
    return it->second.first;
  }

  /**
   * @brief Accesses the mapped value corresponding to the given key.
   *
   * If the key is not present, a new element is created with a default-constructed Value
   * and the key is recorded at the end.
   *
   * @param key The key to access.
   * @return Reference to the associated Value.
   */
  Value& operator[](const Key& key) {
    auto it = map.find(key);
    if (it == map.end()) {
      order.push_back(key);
      auto list_it = std::prev(order.end());
      auto result = map.insert({
        key, {Value(), list_it}
      });
      return result.first->second.first;
    }
    return it->second.first;
  }

  /**
   * @brief Returns the number of elements in the map.
   *
   * @return The size of the map.
   */
  size_t size() const { return map.size(); }

  /**
   * @brief Checks whether the map is empty.
   *
   * @return true if the map contains no elements, false otherwise.
   */
  bool empty() const { return map.empty(); }

  /**
   * @brief Removes all elements from the map.
   */
  void clear() {
    map.clear();
    order.clear();
  }

  /**
   * @brief Finds an element by key in insertion order.
   *
   * @param key The key to search for.
   * @return Iterator to the element if found, otherwise end().
   */
  iterator_type find(const Key& key) {
    auto it = map.find(key);
    if (it == map.end())
      return end();
    return iterator_type(it->second.second, this);
  }

  /**
   * @brief Finds an element by key in insertion order (const version).
   *
   * @param key The key to search for.
   * @return Iterator to the element if found, otherwise end().
   */
  const_iterator_type find(const Key& key) const {
    auto it = map.find(key);
    if (it == map.end())
      return end();
    return const_iterator_type(it->second.second, this);
  }

  /**
   * @brief Erases the element with the specified key.
   *
   * Removes the element identified by key from the map.
   *
   * @param key The key of the element to erase.
   * @return The number of elements removed (0 or 1).
   */
  size_t erase(const Key& key) {
    auto it = map.find(key);
    if (it != map.end()) {
      order.erase(it->second.second);
      map.erase(it);
      return 1;
    }
    return 0;
  }

  /**
   * @brief Erases the element at the given iterator position.
   *
   * The element is removed from both the underlying map and the insertion order list.
   *
   * @param pos Iterator pointing to the element to erase.
   * @return Iterator to the element following the erased element.
   */
  template <typename IterType>
  iterator_type erase(IterType pos) {
    if (pos == end())
      return pos;
    auto listIt = pos.it;
    auto next = listIt;
    ++next;
    const Key& key = *listIt;
    auto mapIt = map.find(key);
    if (mapIt != map.end()) {
      order.erase(mapIt->second.second);
      map.erase(mapIt);
    }
    return iterator_type(next, this);
  }

  /**
   * @brief Returns an iterator to the beginning of the container in insertion order.
   *
   * @return Iterator to the first element.
   */
  iterator_type begin() { return iterator_type(order.begin(), this); }

  /**
   * @brief Returns an iterator to the beginning of the container in insertion order.
   *
   * @return Iterator to the first element.
   */
  const_iterator_type begin() const { return const_iterator_type(order.begin(), this); }

  /**
   * @brief Returns an iterator to the end of the container.
   *
   * @return Iterator past the last element.
   */
  iterator_type end() { return iterator_type(order.end(), this); }

  /**
   * @brief Returns an iterator to the end of the container.
   *
   * @return Iterator past the last element.
   */
  const_iterator_type end() const { return const_iterator_type(order.end(), this); }
};
