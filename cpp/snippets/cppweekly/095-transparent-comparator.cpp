#include <string>
#include <set>

struct S {
    std::string key;
    std::string value;
};

struct S_comparator {
    bool operator()(S const& lhs, S const& rhs) const {
        return lhs.key < rhs.key;
    }
    template <typename T>
    bool operator()(S const& lhs, T const& t) const {
        return lhs.key < t;
    }
    template <typename T>
    bool operator()(T const& t, S const& rhs) const {
        return t < rhs.key;
    }
    using is_transparent = void;
};

int main() {
    auto s = std::set<S, S_comparator>();
    return s.count("hello"); // doesn't have to be type S
}
