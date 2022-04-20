#include <initializer_list>
#include <iostream>
#include <sstream>
#include <vector>

template <typename T>
std::string to_string_impl(T const& t) {
    auto oss = std::ostringstream();
    oss << t;
    return oss.str();
}

/* 28K binary size */
template <typename ... T>
std::vector<std::string> to_string1(T const&... t) {
    return {to_string_impl(t)...};
}

/* 28K binary size */
template <typename ... T>
std::vector<std::string> to_string2(T const&... t) {
    auto const f = [](auto const &t) {
        auto oss = std::ostringstream();
        oss << t;
        return oss.str();
    };

    return {f(t)...};
}

/* 28K binary size */
template <typename ... T>
std::vector<std::string> to_string3(T const&... t) {
    return {[](auto const &t) {
        auto oss = std::ostringstream();
        oss << t;
        return oss.str();
    }(t)...};
}

/* 27,776 binary size, smallest */
template <typename ... T>
std::vector<std::string> to_string4(T const&... t) {
    auto out = std::vector<std::string>();
    auto oss = std::ostringstream();
    (void)std::initializer_list<int>{(oss.str(""), oss << t, out.push_back(oss.str()), 0)...};
    return out;
}

int main() {
    auto const v = to_string4(1, "hello", '3', 7.6);
    for (auto const &o : v) {
        std::cout << o << '\n';
    }
}
