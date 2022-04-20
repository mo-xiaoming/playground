#include <array>
#include <vector>
#include <type_traits>

template <typename Container>
struct Value_type_of_impl { using type = typename Container::value_type; };

template <typename T, std::size_t N>
struct Value_type_of_impl<T[N]> { using type = T; };

template <typename Container>
using Value_type_of = typename Value_type_of_impl<Container>::type;


static_assert(std::is_same_v<Value_type_of<std::vector<int>>, int>);
static_assert(std::is_same_v<Value_type_of<int[20]>, int>);
static_assert(std::is_same_v<Value_type_of<std::array<int, 20>>, int>);
