#include <concepts>
#include <type_traits>

// std::copyable<int[]> == false
// std::is_trivially_copyable_v<int[]> == true
template <typename T>
concept Trivially_copyable = std::copyable<T>&& std::is_trivially_copyable_v<T>;

template <typename T>
concept Buffer_safe = Trivially_copyable<T> ||
                      (std::is_array_v<T> &&
                       Trivially_copyable<std::remove_all_extents_t<T>>);
