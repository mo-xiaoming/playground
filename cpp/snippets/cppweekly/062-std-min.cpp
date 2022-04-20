#include <cmath>
#include <type_traits>
#include <initializer_list>

constexpr decltype(auto) my_min(auto const &a1, auto const& ... t) {
    auto const *r = &a1;
    ((r = &std::min(*r, t)), ...);
    return r;
}


// std::fmin does not return reference, but value instead
template <typename F1, typename ... T>
constexpr auto my_fmin(F1 const &f1, T const& ... t) {
    auto r = f1;
    ((r = std::fmin(r, t)), ...);
    return r;
}

template <typename F1, typename ... T>
constexpr auto my_fmin_cxx11(F1 const &f1, T const& ... t) {
    auto r = f1;
    (void)std::initializer_list{(r = std::fmin(r, t)) ...};
    return r;
}
