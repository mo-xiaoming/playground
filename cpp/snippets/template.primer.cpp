#include <cstddef>
#include <functional>
#include <type_traits>

template <typename T>
struct rank { static inline constexpr int value = 0; };

template <typename T, auto N>
struct rank<T[N]> { static inline constexpr int value = 1 + rank<T>::value; };

template <typename T>
struct rank<T[]> { static inline constexpr int value = 1 + rank<T>::value; };

template <typename T>
constexpr int rank_v = rank<T>::value;

static_assert(rank_v<int> == 0);
static_assert(rank_v<int[3]> == 1);
static_assert(rank_v<int[3][4]> == 2);
static_assert(rank_v<int[3][4][5]> == 3);
static_assert(rank_v<int[]> == 1);
static_assert(rank_v<int[][1][2][3]> == 4);

template <typename T, T v>
struct integral_constant {
    static inline constexpr T value = v;
    constexpr operator T() const noexcept { return value; }
    constexpr T operator()() const noexcept { return value; }
};

template <typename T, T v>
constexpr T integral_constant_v = integral_constant<T, v>::value;

static_assert(integral_constant_v<int, 3> == 3);

template <typename T>
struct rank0 : integral_constant<unsigned, 0> {};

template <typename T, auto N>
struct rank0<T[N]> : integral_constant<unsigned, 1 + rank0<T>::value> {};

template <typename T>
struct rank0<T[]> : integral_constant<unsigned, 1 + rank0<T>::value> {};

template <typename T>
constexpr int rank0_v = rank0<T>::value;

static_assert(rank0_v<int> == 0);
static_assert(rank0_v<int[3]> == 1);
static_assert(rank0_v<int[3][4]> == 2);
static_assert(rank0_v<int[3][4][5]> == 3);
static_assert(rank0_v<int[]> == 1);
static_assert(rank0_v<int[][1][2][3]> == 4);

template <bool b>
using bool_constant = integral_constant<bool, b>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <typename T, typename U>
struct is_same : false_type {};

template <typename T>
struct is_same<T, T> : true_type {};

template <typename T, typename U>
constexpr bool is_same_v = is_same<T, U>::value;

static_assert(is_same_v<int, int>);
static_assert(is_same_v<int, double> == false);

template <typename T>
struct type_is { using type = T; };

template <typename T>
struct remove_const : type_is<T> {};

template <typename T>
struct remove_const<T const> : type_is<T> {};

template <typename T>
using remove_const_t = typename remove_const<T>::type;

static_assert(is_same_v<remove_const_t<int>, int>);
static_assert(is_same_v<remove_const_t<int const>, int>);

template <bool, typename T, typename F>
struct conditional : type_is<T> {};

template <typename T, typename F>
struct conditional<false, T, F> : type_is<F> {};

template <bool b, typename T, typename F>
using conditional_t = typename conditional<b, T, F>::type;

static_assert(is_same_v<conditional_t<sizeof(int) < sizeof(double), int, double>, int>);
static_assert(is_same_v<conditional_t<std::greater<unsigned>()(sizeof(int), sizeof(double)), int, double>, double>);
struct X { void operator()(){} };
struct Y { void operator()(){} };
void foo() {
    auto t = conditional_t<1 < 0, X, Y>();
    t();
}
struct Z : conditional_t<1 < 0, X, Y> {};

template <bool, typename T = void>
struct enable_if : type_is<T> {};

template <typename T>
struct enable_if<false, T> {};

template <bool b, typename T>
using enable_if_t = typename enable_if<b, T>::type;

template <typename T>
struct is_void : is_same<void, typename remove_const<T>::type> {};

template <typename T>
constexpr bool is_void_v = is_void<T>::value;

static_assert(is_void_v<void>);
static_assert(is_void_v<void const>);

// declaration
template <typename T, typename ... P0toN>
struct is_one_of : false_type {};

// termination
template <typename T>
struct is_one_of<T> : false_type {};

// recursion
template <typename T, typename U, typename ... P1toN>
struct is_one_of<T, U, P1toN...> : conditional_t<is_same_v<T, U>, true_type, is_one_of<T, P1toN...>> {};

template <typename T, typename ... P0toN>
constexpr bool is_one_of_v = is_one_of<T, P0toN...>::value;

static_assert(is_one_of_v<int> == false);
static_assert(is_one_of_v<int, double> == false);
static_assert(is_one_of_v<int, int> == true);
static_assert(is_one_of_v<int, double, void, int> == true);
static_assert(is_one_of_v<int, double, void, float> == false);

template <typename T>
constexpr bool is_void0_v = is_one_of_v<T, void, void const, void volatile, void const volatile>;
static_assert(is_void0_v<void>);
static_assert(is_void0_v<void const>);
static_assert(is_void0_v<void const volatile>);
static_assert(is_void0_v<int> == false);

template <typename T>
struct is_copy_assignable {
private:
    template <typename U, typename = decltype(std::declval<T&>() = std::declval<U const&>())>
    static true_type try_assign(U);

    static false_type try_assign(...);
public:
    static inline constexpr bool value = decltype(try_assign(std::declval<T>()))::value;
};

template <typename T>
constexpr bool is_copy_assignable_v = is_copy_assignable<T>::value;

static_assert(is_copy_assignable_v<X>);
struct A {
    A& operator=(A const&) = delete;
};
static_assert(is_copy_assignable_v<A> == false);

template <typename ...>
using void_t = void;

template <typename, typename = void>
struct has_type_member : false_type {};

template <typename T>
struct has_type_member<T, void_t<typename T::foo>> : true_type {};

struct B {
    using foo = int;
};
static_assert(has_type_member<B>::value);
static_assert(has_type_member<X>::value == false);

template <typename T>
using copy_assignment_t = decltype(std::declval<T&>() = std::declval<T const&>());

template <typename T, typename = void>
struct is_copy_assignable0 : false_type {};

template <typename T>
//struct is_copy_assignable0<T, void_t<copy_assignment_t<T>>> : true_type {};
struct is_copy_assignable0<T, void_t<copy_assignment_t<T>>> : is_same<copy_assignment_t<T>, T&> {};

static_assert(is_copy_assignable0<X>::value);
static_assert(is_copy_assignable0<A>::value == false);
struct C {
    C(C const&) {};
    C operator=(C const&) { return *this; }; // wrong return type
};
static_assert(is_copy_assignable0<C>::value == false);

// declaration
template <int N, typename ... Ts>
struct at;

// termination
template <typename T, typename ... Ts>
struct at<0, T, Ts...> { using type = T; };

// recursion
template <int N, typename T, typename ... Ts>
struct at<N, T, Ts...> : at<N-1, Ts...> {};

template <int N, typename ... Ts>
using at_t = typename at<N, Ts...>::type;

static_assert(is_same_v<at_t<0, int, bool, double>, int>);
static_assert(is_same_v<at_t<1, int, bool, double>, bool>);
static_assert(is_same_v<at_t<2, int, bool, double>, double>);

template <typename T, typename ... Ts>
using front_t = T;

static_assert(is_same_v<front_t<int, bool, double>, int>);
static_assert(is_same_v<front_t<int>, int>);

template <typename ... Ts>
struct tail : at<sizeof...(Ts) - 1U, Ts...> {};

template <typename ... Ts>
using tail_t = typename tail<Ts...>::type;

static_assert(is_same_v<tail_t<int, bool, double>, double>);
static_assert(is_same_v<tail_t<int>, int>);

namespace detail {
    std::true_type is_nullptr(std::nullptr_t);
    std::false_type is_nullptr(...);
}

template <typename T>
using is_nullptr_v = decltype(detail::is_nullptr(std::declval<T>()));

namespace detail {
    template <typename T>
        std::true_type is_const(type_is<T const>);
    template <typename T>
        std::false_type is_const(type_is<T>);
}

template <typename T>
using is_const_v = typename decltype(detail::is_const<T>())::value;

template <typename T, typename ...Ts>
struct hasType {
      static constexpr bool value = false;
};

template <typename T, typename H, typename ...Ts>
struct hasType<T, H, Ts...> {
      static constexpr bool value = std::is_same_v<T, H> || hasType<T, Ts...>::value;
};

template <typename ...Ts>
struct hasDuplication {
      static constexpr bool value = false;
};

template <typename T, typename ...Ts>
struct hasDuplication<T, Ts...> {
      static constexpr bool value = hasType<T, Ts...>::value || hasDuplication<Ts...>::value;
};

template <typename T, typename ...Ts>
struct CountType {
    static constexpr std::size_t value = 0;
};

template <typename T, typename U, typename ...Ts>
struct CountType<T, U, Ts...> {
    static constexpr std::size_t value =
        (std::is_same<T, U>::value ? 1 : 0) + CountType<T, Ts...>::value;
};
