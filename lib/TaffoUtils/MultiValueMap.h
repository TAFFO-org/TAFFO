//===- MultiValueMap.h - Ordered Map from Values to Data --------*- C++ -*-===//
//
//   TAFFO - Tuning Assistant for Floating Point to Fixed Point Optimization
//
//===----------------------------------------------------------------------===//
//
// MultiValueMap is intended for mapping small lists of Value* to an arbitrary
// other type.  Any Value* V1 from the list can be used to get the associated
// V2, or the complete list of associated values.
// Every Value* can be contained in at most one list at a time.
// The map itself is ordered, and insertion of new associations can happen
// in any position.
//
// This data structure guarantees that the mapped values will never change
// allocation address during the lifetime of the object, no matter how it is
// mutated.
//
// All operations invalidate all item-wise standard iterators.
//
//===----------------------------------------------------------------------===//

#ifndef MULTI_VALUE_MAP_H
#define MULTI_VALUE_MAP_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include <limits>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>

namespace taffo
{


template <typename KeyT>
struct MultiValueMapConfig {
  // All methods will be called with a first argument of type ExtraData.  The
  // default implementations in this class take a templated first argument so
  // that users' subclasses can use any type they want without having to
  // override all the defaults.
  struct ExtraData {
  };

  template <typename ExtraDataT>
  static void onRAUW(const ExtraDataT & /*Data*/, KeyT /*Old*/, KeyT /*New*/) {}
  template <typename ExtraDataT>
  static void onDelete(const ExtraDataT & /*Data*/, KeyT /*Old*/) {}
};


template <typename KeyT, typename ValueT,
          typename ConfigT = MultiValueMapConfig<KeyT>>
class MultiValueMapBase
{
protected:
  struct KeyListItemT;

  // Type of the list of the groups of values associated to each key
  using KeyListT = std::list<KeyListItemT>;

  struct KeyListItemT {
    std::unique_ptr<ValueT> Value;
    KeyT Key;
    typename KeyListT::iterator TagIt;
    long long OrderIdx = std::numeric_limits<long long>::max();

    bool isTag() const { return bool(Value); };
  };

  struct SingleValueIndexConfig : public llvm::ValueMapConfig<KeyT> {
    using ExtraData = MultiValueMapBase<KeyT, ValueT, ConfigT> *;
    static void onRAUW(const ExtraData &Data, KeyT OldK, KeyT NewK);
    static void onDelete(const ExtraData &Data, KeyT K);
  };

  // Index that maps each value to the list where it is contained
  using SingleValueIndexT = llvm::ValueMap<KeyT, typename KeyListT::iterator,
                                           SingleValueIndexConfig>;

  SingleValueIndexT Index;
  KeyListT KeyList;
  long long OrderIdxSpacing = 0x100000;

  MultiValueMapBase() : Index(this) {}
};


template <typename BaseT, bool isConst = false>
class MultiValueMapIterator : public std::iterator<std::bidirectional_iterator_tag,
                                                   std::pair<typename BaseT::key_type,
                                                             typename BaseT::mapped_type>>
{
  using KeyT = typename BaseT::key_type;
  using MappedT = typename BaseT::mapped_type;
  using KeyListT = typename std::conditional<
      isConst, const typename BaseT::KeyListT,
      typename BaseT::KeyListT>::type;
  using KeyListItT = typename std::conditional<
      isConst, typename BaseT::KeyListT::const_iterator,
      typename BaseT::KeyListT::iterator>::type;
  using ValueT = typename std::conditional<
      isConst, const std::pair<KeyT, MappedT>, std::pair<KeyT, MappedT>>::type;

  template <typename T, typename U, typename C>
  friend class MultiValueMap;

  BaseT *Parent = nullptr;
  mutable KeyListItT IKeyList;

  inline void skipTagForward() const
  {
    if (IKeyList != Parent->KeyList.end() && IKeyList->isTag())
      IKeyList++;
  }
  inline void skipTagBack()
  {
    if (IKeyList != Parent->KeyList.begin() && IKeyList->isTag())
      IKeyList--;
  }
  inline KeyListItT insertionPointerForNewList()
  {
    if (IKeyList == Parent->KeyList.begin() || IKeyList == Parent->KeyList.end())
      return IKeyList;
    auto Next = IKeyList;
    auto Prev = IKeyList;
    --Prev;
    if (Prev->isTag())
      return Prev;
    while (Next != Parent->KeyList.end() && !Next->isTag())
      Next++;
    return Next;
  }
  inline std::pair<KeyListItT, KeyListItT> leftInsertionPointer()
  {
    if (IKeyList == Parent->KeyList.begin())
      return {IKeyList, IKeyList};
    auto Prev = IKeyList;
    auto Next = IKeyList;
    --Prev;
    if (Prev->isTag() && Prev != Parent->KeyList.begin()) {
      --Prev;
      --Next;
    }
    return {Next, Prev->TagIt};
  }
  inline std::pair<KeyListItT, KeyListItT> rightInsertionPointer()
  {
    if (IKeyList == Parent->KeyList.end())
      return {IKeyList, IKeyList};
    skipTagForward();
    return {IKeyList, IKeyList->TagIt};
  }

public:
  MultiValueMapIterator(BaseT &Parent, KeyListItT IKeyList) : Parent(&Parent), IKeyList(IKeyList) {}
  MultiValueMapIterator(BaseT &Parent, KeyT Key) : Parent(&Parent)
  {
    IKeyList = Parent.Index[Key];
  }
  MultiValueMapIterator(MultiValueMapIterator const &Other) = default;
  MultiValueMapIterator(MultiValueMapIterator &&) = default;
  MultiValueMapIterator() = default;

  MultiValueMapIterator &operator=(const MultiValueMapIterator &Other)
  {
    Parent = Other.Parent;
    IKeyList = Other.IKeyList;
    return *this;
  };

  struct ValueTypeProxy {
    const KeyT first;
    MappedT &second;
    ValueTypeProxy *operator->() { return this; }
    operator std::pair<KeyT, MappedT>() const
    {
      return std::make_pair(first, second);
    }
  };

  MultiValueMapIterator &operator++()
  {
    skipTagForward();
    IKeyList++;
    skipTagForward();
    return *this;
  }
  MultiValueMapIterator operator++(int)
  {
    MultiValueMapIterator Res = *this;
    ++(*this);
    return Res;
  }

  MultiValueMapIterator &operator--()
  {
    IKeyList--;
    skipTagBack();
    return *this;
  }
  MultiValueMapIterator operator--(int)
  {
    MultiValueMapIterator Res = *this;
    --(*this);
    return Res;
  }

  MultiValueMapIterator &skip()
  {
    skipTagForward();
    while (IKeyList != Parent->KeyList.end() && !IKeyList->isTag())
      IKeyList++;
    skipTagForward();
    return *this;
  }
  MultiValueMapIterator &reverseSkip()
  {
    skipTagForward();
    IKeyList = IKeyList->TagIt;
    if (IKeyList == Parent->KeyList.begin()) {
      skipTagForward();
      return *this;
    }
    --IKeyList;
    IKeyList = IKeyList->TagIt;
    skipTagForward();
    return *this;
  }

  bool operator==(const MultiValueMapIterator &RHS) const
  {
    if (this == &RHS)
      return true;
    skipTagForward();
    RHS.skipTagForward();
    return RHS.IKeyList == IKeyList;
  }
  bool operator!=(const MultiValueMapIterator &RHS) const
  {
    return !(*this == RHS);
  }
  ValueTypeProxy operator*() const
  {
    skipTagForward();
    MappedT &V = *(IKeyList->TagIt->Value.get());
    return ValueTypeProxy{IKeyList->Key, V};
  }
  ValueTypeProxy operator->() const
  {
    return operator*();
  }

  bool operator<=(const MultiValueMapIterator &RHS) const
  {
    skipTagForward();
    if (RHS.IKeyList == IKeyList)
      return true;
    return IKeyList->OrderIdx <= RHS.IKeyList->OrderIdx;
  }
  bool operator<(const MultiValueMapIterator &RHS) const
  {
    return !(RHS <= *this);
  }
  bool operator>(const MultiValueMapIterator &RHS) const
  {
    return !(*this <= RHS);
  }
  bool operator>=(const MultiValueMapIterator &RHS) const
  {
    return RHS <= *this;
  }
};


template <typename KeyT, typename ValueT,
          typename ConfigT = MultiValueMapConfig<KeyT>>
class MultiValueMap : public MultiValueMapBase<KeyT, ValueT, ConfigT>
{
  template <typename T, bool C>
  friend class MultiValueMapIterator;
  friend struct MultiValueMapBase<KeyT, ValueT, ConfigT>::SingleValueIndexConfig;

  using KeyListT =
      typename MultiValueMapBase<KeyT, ValueT, ConfigT>::KeyListT;
  using KeyListItemT =
      typename MultiValueMapBase<KeyT, ValueT, ConfigT>::KeyListItemT;
  using SingleValueIndexT =
      typename MultiValueMapBase<KeyT, ValueT, ConfigT>::SingleValueIndexT;

protected:
  typename ConfigT::ExtraData Data;

public:
  using key_type = KeyT;
  using mapped_type = ValueT;
  using value_type = std::pair<KeyT, ValueT>;
  using size_type = unsigned;

  MultiValueMap() {}
  MultiValueMap(typename ConfigT::ExtraData &Data) : Data(Data) {}

  // Disable copy & move because they're not implemented yet, and are currently
  // not worth implementing
  MultiValueMap(const MultiValueMap &) = delete;
  MultiValueMap(MultiValueMap &&) = delete;
  MultiValueMap &operator=(const MultiValueMap &) = delete;
  MultiValueMap &operator=(MultiValueMap &&) = delete;

  bool empty() const { return this->KeyList.empty(); }
  unsigned size() const { return this->Index.size(); }

  void clear()
  {
    this->KeyList.clear();
    this->Index.clear();
  }

  size_type count(const KeyT &K) const
  {
    return this->Index.find(K) != this->Index.end();
  }

  using iterator = MultiValueMapIterator<MultiValueMap<KeyT, ValueT, ConfigT>>;
  using const_iterator = MultiValueMapIterator<const MultiValueMap<KeyT, ValueT, ConfigT>, true>;

  inline iterator begin()
  {
    return iterator(*this, this->KeyList.begin());
  }
  inline iterator end()
  {
    return iterator(*this, this->KeyList.end());
  }
  inline const_iterator begin() const
  {
    return const_iterator(*this, this->KeyList.begin());
  }
  inline const_iterator end() const
  {
    return const_iterator(*this, this->KeyList.end());
  }

  iterator find(const KeyT &K)
  {
    auto VListIt = this->Index.find(K);
    if (VListIt == this->Index.end())
      return end();
    return iterator(*this, VListIt->second);
  }
  ValueT lookup(const KeyT &K)
  {
    auto VListIt = this->Index.find(K);
    if (VListIt == this->Index.end())
      return ValueT();
    return *(VListIt->second->TagIt->Value.get());
  }
  ValueT &operator[](const KeyT &K)
  {
    auto VListIt = this->Index.find(K);
    assert(VListIt != this->Index.end());
    return *(VListIt->second->TagIt->Value.get());
  }

  /// Get the list of keys associated to the same value as a given key.
  /// @returns false if the key is not in the map.
  bool getAssociatedValues(KeyT K, llvm::SmallVectorImpl<KeyT> &OutV)
  {
    auto I = this->find(K);
    if (I == this->end())
      return false;
    auto ListI = I.IKeyList->TagIt;
    ++ListI;
    while (ListI != this->KeyList.end() && !ListI->isTag()) {
      OutV.push_back(ListI->Key);
      ListI++;
    }
    return true;
  }

private:
  long long orderIdxForNewElem(typename KeyListT::iterator Pos)
  {
    // TODO: CATCH OVERFLOWS/UNDERFLOWS/BAD IDX ALLOCATIONS!!!!
    auto Next = Pos;
    if (Next == this->KeyList.begin()) {
      if (Next != this->KeyList.end())
        return Next->OrderIdx - this->OrderIdxSpacing;
      else
        return 0;
    }
    auto Prev = --Pos;
    if (Next == this->KeyList.end())
      return Prev->OrderIdx + this->OrderIdxSpacing;
    return (Prev->OrderIdx + Next->OrderIdx) / 2;
  }

public:
  std::pair<iterator, bool> insert(iterator P, const KeyT &K, const ValueT &V)
  {
    iterator Existing = this->find(K);
    if (Existing != this->end())
      return std::make_pair(Existing, false);

    auto FixedP = P.insertionPointerForNewList();
    KeyListItemT NewTag;
    NewTag.Value.reset(new ValueT(V));
    NewTag.OrderIdx = orderIdxForNewElem(FixedP);
    auto TagIt = this->KeyList.insert(FixedP, std::move(NewTag));
    TagIt->TagIt = TagIt;

    KeyListItemT NewItem;
    NewItem.Key = K;
    NewItem.TagIt = TagIt;
    NewItem.OrderIdx = orderIdxForNewElem(FixedP);
    auto ItmIt = this->KeyList.insert(FixedP, std::move(NewItem));

    this->Index[K] = ItmIt;

    iterator ResIt = iterator(*this, ItmIt);
    return std::make_pair(ResIt, true);
  }
  /// Inserts key,value pair into the map if the key isn't already in the map.
  /// Note that this method will always create a new Value* group, regardless
  /// of the mapped value.
  /// @returns pair(iterator pointing to the inserted pair, true) in case of
  ///   success; otherwise -- if the key is already in the map --
  ///   pair(the position of the existing pair, false).
  std::pair<iterator, bool> insert(iterator P,
                                   const std::pair<KeyT, ValueT> &KV)
  {
    return insert(P, KV.first, KV.second);
  }
  std::pair<iterator, bool> insert(iterator P, std::pair<KeyT, ValueT> &&KV)
  {
    return insert(P, KV.first, KV.second);
  }
  std::pair<iterator, bool> push_back(const std::pair<KeyT, ValueT> &KV)
  {
    return insert(end(), KV.first, KV.second);
  }
  std::pair<iterator, bool> push_back(std::pair<KeyT, ValueT> &&KV)
  {
    return insert(end(), KV.first, KV.second);
  }
  std::pair<iterator, bool> push_back(const KeyT &K, const ValueT &V)
  {
    return insert(end(), K, V);
  }

  /// Adds a key to an existing key list / value association.
  /// If P points between two keys associated to the same value, K is associated
  /// to that value. If P points between a left key and a right key associated
  /// to different values, K will be added to the list of the right key.
  /// This method cannot be used at the end of the collection.
  /// @returns pair(iterator pointing to the inserted pair, true) in case of
  ///   success; otherwise -- if the key is already in the map or the iterator
  ///   is invalid -- pair(the given iterator, false).
  std::pair<iterator, bool> insertRight(iterator P, const KeyT &K)
  {
    if (P == end())
      return std::make_pair(P, false);
    if (this->Index.find(K) != this->Index.end())
      return std::make_pair(P, false);
    auto InsertP = P.rightInsertionPointer();
    KeyListItemT NewItem;
    NewItem.Key = K;
    NewItem.TagIt = InsertP.second;
    NewItem.OrderIdx = orderIdxForNewElem(InsertP.first);
    auto ItmIt = this->KeyList.insert(InsertP.first, std::move(NewItem));
    this->Index[K] = ItmIt;
    return std::make_pair(iterator(*this, ItmIt), true);
  }
  /// Adds a key to an existing key list / value association.
  /// If P points between two keys associated to the same value, K is associated
  /// to that value. If P points between a left key and a right key associated
  /// to different values, K will be added to the list of the left key.
  /// This method cannot be used at the beginning of the collection.
  /// @returns pair(iterator pointing to the inserted pair, true) in case of
  ///   success; otherwise -- if the key is already in the map or the iterator
  ///   is invalid -- pair(the given iterator, false).
  std::pair<iterator, bool> insertLeft(iterator P, const KeyT &K)
  {
    if (P == begin())
      return std::make_pair(P, false);
    if (this->Index.find(K) != this->Index.end())
      return std::make_pair(P, false);
    auto InsertP = P.leftInsertionPointer();
    KeyListItemT NewItem;
    NewItem.Key = K;
    NewItem.TagIt = InsertP.second;
    NewItem.OrderIdx = orderIdxForNewElem(InsertP.first);
    auto ItmIt = this->KeyList.insert(InsertP.first, std::move(NewItem));
    this->Index[K] = ItmIt;
    return std::make_pair(iterator(*this, ItmIt), true);
  }

  template <typename InputIt>
  iterator insert(iterator P, InputIt I, InputIt E)
  {
    std::pair<iterator, bool> State{P, true};
    for (; I != E; ++I, ++State.first)
      State = insert(State.first, *I);
    return P;
  }
  template <typename InputIt>
  iterator insert(iterator P, InputIt I, InputIt E, ValueT MV)
  {
    std::pair<iterator, bool> State;
    State = insert(P, std::make_pair(*I, MV));
    for (++I, ++State.first; I != E; ++I, ++State.first)
      State = insertLeft(State.first, *I);
    return P;
  }

  iterator eraseAll(iterator I)
  {
    auto Ptr = I.IKeyList->TagIt;
    Ptr = this->KeyList.erase(Ptr);
    while (Ptr != this->KeyList.end() && !Ptr->isTag()) {
      this->Index.erase(Ptr->Key);
      Ptr = this->KeyList.erase(Ptr);
    }
    return iterator(*this, Ptr);
  }
  bool eraseAll(const KeyT &K)
  {
    iterator KI = find(K);
    if (KI == end())
      return false;
    eraseAll(KI);
    return true;
  }

  iterator erase(iterator I)
  {
    I.skipTagForward();
    auto Itm = I.IKeyList;
    auto Tag = Itm->TagIt;
    this->Index.erase(Itm->Key);
    Itm = this->KeyList.erase(Itm);
    auto Prev = Itm;
    --Prev;
    if (Itm->isTag() && Prev == Tag) {
      this->KeyList.erase(Tag);
    }
    return iterator(*this, Itm);
  }
  bool erase(const KeyT &K)
  {
    iterator KI = find(K);
    if (KI == end())
      return false;
    erase(KI);
    return true;
  }
  iterator erase(iterator B, iterator E)
  {
    while (B != E)
      B = this->erase(B);
    return E;
  }

  void dump()
  {
#define DEBUG_TYPE "MultiValueMap"
    for (auto &V : this->KeyList) {
      if (V.isTag())
        LLVM_DEBUG(llvm::dbgs() << "[TAG] V=" << V.Value.get() << "\n");
      else
        LLVM_DEBUG(llvm::dbgs() << "[ITM] K=" << V.Key << " O=" << V.OrderIdx << "\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "[[INDEX]]\n");
    for (auto I = this->Index.begin(); I != this->Index.end(); ++I) {
      LLVM_DEBUG(llvm::dbgs() << "V=" << I->first << "\n");
    }
#undef DEBUG_TYPE
  }
};

template <typename KeyT, typename ValueT, typename ConfigT>
void MultiValueMapBase<KeyT, ValueT, ConfigT>::SingleValueIndexConfig::onRAUW(
    const MultiValueMapBase<KeyT, ValueT, ConfigT>::
        SingleValueIndexConfig::ExtraData &Data,
    KeyT OldK, KeyT NewK)
{
  MultiValueMap<KeyT, ValueT, ConfigT> &MVM =
      *static_cast<MultiValueMap<KeyT, ValueT, ConfigT> *>(Data);

  ConfigT::onRAUW(MVM.Data, OldK, NewK);

  auto OldKIt = MVM.find(OldK);
  auto NewKIt = MVM.find(NewK);
  if (NewKIt != MVM.end()) {
    /* the new key is in the map; erase it so that we can replace its value
     * with the value of the old key */
    MVM.erase(NewKIt);
  }
  if (OldKIt != MVM.end()) {
    /* associate the new key with same value as the old key */
    MVM.insertRight(OldKIt, NewK);
    MVM.erase(OldKIt);
  }
}

template <typename KeyT, typename ValueT, typename ConfigT>
void MultiValueMapBase<KeyT, ValueT, ConfigT>::SingleValueIndexConfig::onDelete(
    const MultiValueMapBase<KeyT, ValueT, ConfigT>::
        SingleValueIndexConfig::ExtraData &Data,
    KeyT K)
{
  MultiValueMap<KeyT, ValueT, ConfigT> &MVM =
      *static_cast<MultiValueMap<KeyT, ValueT, ConfigT> *>(Data);

  ConfigT::onDelete(MVM.Data, K);

  auto KIt = MVM.find(K);
  if (KIt != MVM.end())
    MVM.erase(KIt);
}


} // namespace taffo

#endif
