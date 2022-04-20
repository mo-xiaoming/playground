#include <set>
#include <string>
#include <type_traits>
#include <utility>

struct S {
    std::string key;
    std::string value;
};

template <typename Type, typename... Comparator>
auto make_set(Comparator&&... comparator)
{
    struct Compare : std::decay_t<Comparator>... {
        using std::decay_t<Comparator>::operator()...;
        using is_transparent = void;
    };

    return std::set<Type, Compare> { Compare { std::forward<Comparator>(comparator)... } };
}

int main()
{
    auto s = make_set<S>(
        [](S const& lhs, S const& rhs) { return lhs.key < rhs.key; },
        [](S const& lhs, auto const& k) { return lhs.key < k; },
        [](auto const& k, S const& rhs) { return k < rhs.key; });
    return s.count("hello"); // doesn't have to be type S
}
