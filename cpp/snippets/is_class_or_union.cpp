#include <string>
#include <type_traits>

struct A {};
union B {};

template <typename T> constexpr bool is_class_or_union(int T::*) {
  return true;
}

template <typename T> constexpr bool is_class_or_union(...) { return false; }

static_assert(is_class_or_union<std::string>(nullptr));
static_assert(is_class_or_union<A>(nullptr));
static_assert(is_class_or_union<B>(nullptr));
static_assert(!is_class_or_union<int>(nullptr));

template <typename T, typename = void>
struct is_class_or_union0 : std::false_type {};

template <typename T>
struct is_class_or_union0<T, std::void_t<int T::*>> : std::true_type {};

int main() {}
