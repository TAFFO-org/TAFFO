#pragma once

#include <llvm/Support/Casting.h>

// reimplement dynamic_pointer_cast with LLVM-style RTTI
namespace std {

template <class T, class U>
std::shared_ptr<T> dynamic_ptr_cast(const std::shared_ptr<U>& r) noexcept {
  if (auto p = llvm::dyn_cast<typename std::shared_ptr<T>::element_type>(r.get()))
    return std::shared_ptr<T>(r, p);
  return std::shared_ptr<T>();
}

template <class T, class U>
std::shared_ptr<T> dynamic_ptr_cast_or_null(const std::shared_ptr<U>& r) noexcept {
  if (auto p = llvm::dyn_cast_or_null<typename std::shared_ptr<T>::element_type>(r.get()))
    return std::shared_ptr<T>(r, p);
  return std::shared_ptr<T>();
}

template <class T, class U>
std::shared_ptr<T> static_ptr_cast(const std::shared_ptr<U>& r) noexcept {
  auto p = llvm::cast<typename std::shared_ptr<T>::element_type>(r.get());
  return std::shared_ptr<T>(r, p);
}

template <class T, class U>
bool isa_ptr(const std::shared_ptr<U>& r) noexcept {
  return llvm::isa<T>(r.get());
}

} // namespace std
