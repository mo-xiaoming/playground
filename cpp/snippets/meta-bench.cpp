// https://www.youtube.com/watch?v=9bSG1aHXU60

#include <iostream>
#include <tuple>
#include <type_traits>

namespace v1 {
template <typename, typename> struct tuple_sum;

template <typename... Ts, typename... Us>
struct tuple_sum<std::tuple<Ts...>, std::tuple<Us...>> {
  using type = std::tuple<Ts..., Us...>;
};

template <typename T1, typename T2>
using tuple_sum_t = typename tuple_sum<T1, T2>::type;
} // namespace v1

namespace v2 {
template <typename... Ts, typename... Us>
std::tuple<Ts..., Us...> tuple_sum(std::tuple<Ts...>, std::tuple<Us...>) {
  return {};
}

template <typename Tuple1, typename Tuple2>
using tuple_sum_t = decltype(tuple_sum(Tuple1{}, Tuple2{}));
} // namespace v2

namespace v3 {
template <typename... Tuples>
using tuple_sum_t = decltype(std::tuple_cat(Tuples{}...));
}

using tuple_1_t = std::tuple<int, double>;
using tuple_2_t = std::tuple<double, int[3], char>;
using tuple_1_2_t = std::tuple<int, double, double, int[3], char>;
static_assert(
    std::is_same_v<v1::tuple_sum_t<tuple_1_t, tuple_2_t>, tuple_1_2_t>);
static_assert(
    std::is_same_v<v2::tuple_sum_t<tuple_1_t, tuple_2_t>, tuple_1_2_t>);
static_assert(
    std::is_same_v<v3::tuple_sum_t<tuple_1_t, tuple_2_t>, tuple_1_2_t>);

template <typename Sequence, typename... Args> struct type_sequence_maker;

template <std::size_t... I, class... Args>
struct type_sequence_maker<std::index_sequence<I...>, Args...> {
  using type = std::tuple<
      std::tuple_element_t<I % sizeof...(Args), std::tuple<Args...>>...>;
};

template <std::size_t N, typename... Args>
using make_type_sequence =
    typename type_sequence_maker<std::make_index_sequence<N>, Args...>::type;

static_assert(std::is_same_v<make_type_sequence<1, int>, std::tuple<int>>);
static_assert(std::is_same_v<make_type_sequence<2, int>, std::tuple<int, int>>);
static_assert(std::is_same_v<make_type_sequence<2, int, char, double>,
                             std::tuple<int, char>>);
static_assert(
    std::is_same_v<make_type_sequence<1, int, char>, std::tuple<int>>);
static_assert(
    std::is_same_v<make_type_sequence<2, int, char>, std::tuple<int, char>>);
static_assert(std::is_same_v<make_type_sequence<3, int, char>,
                             std::tuple<int, char, int>>);
static_assert(std::is_same_v<make_type_sequence<5, int, char>,
                             std::tuple<int, char, int, char, int>>);

template <typename... Types, typename... Args> void meta_print(Args...) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

template <typename T> struct pack_size;

template <typename T>
inline constexpr std::size_t pack_size_v = pack_size<T>::size;

template <std::size_t I, typename T> struct pack_element;

template <std::size_t I, typename T>
using pack_element_t = typename pack_element<I, T>::type;

template <typename T, typename... Args> struct pack_rebind;

template <typename T, typename... Args>
using pack_rebind_t = typename pack_rebind<T, Args...>::type;

template <typename T, typename = void> struct is_pack : std::false_type {};

template <typename T>
struct is_pack<T, pack_element_t<pack_size_v<pack_rebind_t<T, void>> - 1,
                                 pack_rebind_t<T, void>>> : std::true_type {};

template <typename T> inline constexpr bool is_pack_v = is_pack<T>::value;

template <typename T, typename... Types> struct if_pack;

template <typename T> struct if_pack<T> : std::enable_if<is_pack_v<T>> {};

template <typename T, typename True>
struct if_pack<T, True> : std::enable_if<is_pack_v<T>, True> {};

template <typename T, typename True, typename False>
struct if_pack<T, True, False> : std::conditional<is_pack_v<T>, True, False> {};

template <typename T, typename... Types>
using if_pack_t = typename if_pack<T, Types...>::type;

template <std::size_t I>
struct index : std::integral_constant<std::size_t, I> {};

template <std::size_t I> using make_index = index<I>;

template <std::size_t... I> struct indexer : std::index_sequence<I...> {};

template <typename Seq> struct to_indexer;

template <std::size_t... I> struct to_indexer<std::index_sequence<I...>> {
  using type = indexer<I...>;
};

template <typename Seq> using to_indexer_t = typename to_indexer<Seq>::type;

template <std::size_t N>
using make_indexer = to_indexer_t<std::make_index_sequence<N>>;

template <typename... Args>
using indexer_for = to_indexer_t<std::index_sequence_for<Args...>>;

template <typename... Args>
struct pack_size<std::tuple<Args...>>
    : std::integral_constant<std::size_t, sizeof...(Args)> {};

template <std::size_t I, typename... Args>
struct pack_element<I, std::tuple<Args...>> {
  using type = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <typename... Types, typename... Args>
struct pack_rebind<std::tuple<Types...>, Args...> {
  using type = std::tuple<Args...>;
};

template <std::size_t I, typename Pack, typename Default = void,
          typename = void>
struct pack_element_or;

template <std::size_t I, typename Pack, typename Default>
struct pack_element_or<I, Pack, Default,
                       std::enable_if_t<(I < pack_size_v<Pack>)>> {
  using type = pack_element_t<I, Pack>;
};

template <std::size_t I, typename Pack, typename Default>
struct pack_element_or<I, Pack, Default,
                       std::enable_if_t<(I >= pack_size_v<Pack>)>> {
  using type = Default;
};

// declaration
template <typename Pack, std::size_t L = 1, typename = void> struct pack_front;

// recursion
template <typename Pack, std::size_t L>
struct pack_front<Pack, L, std::enable_if_t<(pack_size_v<Pack> > 0 && L > 1)>> {
  using type = typename pack_front<pack_element_t<0, Pack>, L - 1>::type;
};

// termination
template <typename Pack>
struct pack_front<Pack, 1, std::enable_if<(pack_size_v<Pack> > 0)>> {
  using type = pack_element_or<0, Pack>;
};

template <typename Pack, std::size_t L = 1>
using pack_front_t = typename pack_front<Pack, L>::type;

// declaration
template <typename Pack, std::size_t L = 1, typename = void>
struct pack_back;

// recursion
template <typename Pack, std::size_t L>
struct pack_back<Pack, L, std::enable_if_t<(pack_size_v<Pack> > 0 && L > 1) >> {
  using type = typename pack_back<pack_element_t<pack_size_v<Pack>-1, Pack>, L-1>::type;
};

// termination
template <typename Pack>
struct pack_back<Pack, 1, std::enable_if_t<(pack_size_v<Pack> > 0)>> {
  using type = pack_element_t<pack_size_v<Pack>-1, Pack>;
};

template <typename Pack, std::size_t N, typename = make_index<N>>
struct pack_truncate;

template <typename Pack, std::size_t N, std::size_t ... I>
struct pack_truncate<Pack, N, indexer<I...>> {
  using type = pack_rebind_t<Pack, pack_element_t<I, Pack>...>;
};

template <typename Pack, std::size_t N>
using pack_truncate_ = typename pack_truncate<Pack, N>:: type;

int main() {
  using type = make_type_sequence<7, bool, char, int, double>;
  type x = {};
  meta_print(x);
}

// https://marzer.github.io/md_blog_2021_05_31_compilation_speed_humps_std_tuple.html
