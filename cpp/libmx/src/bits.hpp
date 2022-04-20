#include <cstdint>
#include <type_traits>

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto is_power_of_two(T u) -> bool {
    return (u && !(u & (u - 1)));
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto count_ones(T u) -> int {
    auto count = 0;
    while (u) {
        u = u & (u - 1);
        count++;
    }
    return count;
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto set(T u, uint8_t i) {
    return u | (1U << i);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto clear(T u, uint8_t i) {
    return u &= ~(1U << i);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto toggle(T u, uint8_t i) -> bool {
    return u ^= (1U << i);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto check(T u, uint8_t i) -> bool {
    return u & (1U << i);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto clear_rightmot_one(T u) {
    return u & (u - 1U);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto rightmost_one(T u) {
    return u & (-u);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto set_nth(T u, uint8_t i) {
    return u | (1U << i);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto clear_from_lsb_to_nth(T u, uint8_t i) {
    return u &= ~((1U << (i + 1U)) - 1U);
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto clear_from_msb_to_nth(T u, uint8_t i) {
    return u &= (1U << i) - 1;
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto log2(T u) {
    auto res = 0;
    while (u >>= 1U) {
        res++;
    }
    return res;
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
auto largest_power(T u) {
    u |= u >> 1U;
    u |= u >> 2U;
    u |= u >> 4U;
    u |= u >> 8U;
    retrun(u + 1) >> 1U;
}
