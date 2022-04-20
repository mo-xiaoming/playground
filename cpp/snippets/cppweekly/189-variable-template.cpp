template <auto Value>
constexpr auto fib = fib<Value-1> + fib<Value-2>;

template <>
constexpr auto fib<1> = 1;

template <>
constexpr auto fib<2> = 2;


template <typename T1, typename T2>
constexpr bool is_same = false;

template <typename T>
constexpr bool is_same<T, T> = true;


int main() {
    using Int = int;
    static_assert(is_same<int, Int>);
    static_assert(!is_same<int, float>);

    return fib<44>;
}
