# The Nightmare of Move Semantics for Trivial Classes

How many expensive string calls?
- Potential memory allocations
  * mallocs if no SSO is used
- i.e., copy constructors or copy assignments for std::string

```cpp
class Cust {
    std::string first;
    std::string last;
    int id;
public:
    Cust(const std::string &f, const std::string &l="", int i = 0) : first(f), last(l), id(i) {}
};

// 4 mallocs (2 create + 2 copy)
Cust c{"Joe", "Fix", 42};
```

```cpp
    Cust(const char *f, const char *l = "", int i = 0) : first(f), last(l), id(i) {}

// 2 mallocs (2 create)
Cust c{"Joe", "Fix", 42};

std::string s = "Joe";
// 3 mallocs (2 create + 1 copy)
Cust d{s, "Fix", 42};

// Or we can overload for all combinations, then back to 2 mallocs(1 create + 1 copy)
// That's what we do in C++98/03

    Cust(const std::string &f, const std::string &l="", int i = 0) : first(f), last(l), id(i) {}
    Cust(const std::string &f, const char *l = "", int i = 0) : first(f), last(l), id(i) {}
    Cust(const char *f, const std::string &l = "", int i = 0) : first(f), last(l), id(i) {}
    Cust(const char *f, const char *l = "", int i = 0) : first(f), last(l), id(i) {}

// 2 mallocs (1 create + 1 copy), not perfect
Cust e{std::move(s), "Fix", 42};
```

```cpp
    Cust(const std::string &f, const std::string &l="", int i = 0) : first(f), last(l), id(i) {}
    Cust(std::string &&f, std::string &&l="", int i=0): first(std::move(f)), last(std::move(l)), id(i) {}

// 2 mallocs (2 create + 2 move)
Cust c{"Joe", "Fix", 42};

    // then add all combinations
    Cust(const std::string &f, std::string &&l="", int i = 0) : first(f), last(std::move(l)), id(i) {}
    Cust(std::string &&f, const std::string &l="", int i=0): first(std::move(f)), last(l), id(i) {}

std::string s = "Joe"
// 2 mallocs (1 copy + 1 create + 1 move)
Cust d{s, "Fix", 42};

// 1 mallocs (1 move + 1 create + 1 move)
Cust e{std::move(s), "Fix", 42}

// However
Cust f{"nico"};     // ERROR: ambigious
                    // can be solved by remove all const std::string &'s default value
//struct S {
//    S(std::string s);
//};
//S x = "hi"; // ERROR
Cust g = "nico";    // ERROR: implicit user-defined conversions.
                    // only const char * to std::string,
                    // not const char[5] to std::string

// const std::string &, std::string &&, const char *, 9 combinations
```

```cpp
    Cust(std::string f, std::string l = "", int i = 0) : first(std::move(f)), last(std::move(l)), id(i) {}
    Cust(const char *f): first(f), last(""), id(0) {}   // solve Cust g = "nico" problem

// 2 mallocs (1 create + 1 move + 1 create + 1 move)
Cust c{"Joe", "Fix", 42};

std::string s = "Joe";
// 2 mallocs (1 copy + 1 move + 1 create + 1 move)
Cust d{s, "Fix", 42};

// 1 mallocs (1 move + 1 move + 1 create + 1 move)
Cust e{std::move(s), "Fix", 42};
```

```cpp
    template <typename S1, typename S2> // covers all 9 combinations, even more
    Cust(S1 &&f, S2 &&l = "", int i = 0) : first(std::forward<S1>(f)), last(std::forward<S2>(l)), id(i) {}

// 2 malloc (2 create)
Cust c{"Joe", "Fix", 42};

std::string s = "Joe";
// 2 malloc (1 copy + 1 create)
Cust d{s, "Fix", 42};

// 1 malloc (1 move + 1 create)
Cust e{std::move{s}, "Fix", 42};

Cust f{"Nico"}; // ERROR, couldn't infer template argument 'S2'
```

> template parameters for call arguments of default values are not deduced

```cpp
    template <typename S1, typename S2 = const char *>
    Cust(S1 &&f, S2 &&l="", int i = 0): first(std::forward<S1>(f)), last(std::forward<S2>(l)), id(i) {}

// Nico found this out one day before he gave this speech
Cust f{"Nico"}; // gcc: OK
                // clang: OK
                // msvc 19.22: error C2440: "default argument": can't convert "const char [1]" to "S2 &&"
                //           : you cannot bind an lvalue to an rvalue reference
```

> we wrote a new template about 850 pages, and I don't know which compiler is right

```cpp
    // change to std::string, every compiler works
    template <typename S1, typename S2 = std::string>

Cust g{f};  // ERROR, f is not const, so template matches better

    // ERROR, because Cust is an lvalue, so S1 is Cust &
    // Rule in $14.8.2.1 [temp.deduct.call]
    // If the parameter type is an rvalue reference to a cv-unqualified template parameter and the
    // argument is an lvalue, the type "lvalue reference to T" is used in plae of T for type deduction
    template <typename S1, typename S2 = std::string, typname = std::enable_if_t<!std::is_same_v<Cust, S1>>>

    template <typename S1, typename S2 = std::string, typname = std::enable_if_t<!std::is_same_v<Cust&, S1>>>

class VIP: public Cust {
    using Cust::Cust;
};
VIP v{"nico"};
VIP v2{v};  // OK, because using Cust::Cust doesn't include template functions
C c{v}; // ERROR

    // use to be an ERROR, nico sent a bug report, not sure about present
    // ERROR: the value of std::is_convertible_v<const Cust&, Cust> is not usable in a contant expression
    // Because a logical error: whether S1 is convertible to Cust depends on the constructor we define here
    template <typename S1, typename S2 = std::string, typname = std::enable_if_t<!std::is_convertible_v<S1, Cust>>>

    template <typename S1, typename S2 = std::string, typname = std::enable_if_t<std::is_construtible_v<std::string, S1>>>
    template <typename S1, typename S2 = std::string, typname = std::enable_if_t<std::is_convertible<S1, std::string>>>
```

The winner is

```cpp
Cust c{"Joe", "Fix", 42};
Cust d{s, "Fix", 42};
Cust e{std::move(s), "Fix", 42};

    // 11 mallocs (4create + 7copy + 1move)
    // needs const char * overload, otherwise = "hello" failes
    Cust(std::string f, std::string l = "", int i = 0): first(f), last(l), id{i} {}

    // 5 mallocs (4create + 1copy + 5move)
    // needs const char * overload, otherwise = "hello" failes
    Cust(std::string f, std::string l = "", int i = 0): first(std::move(f)), last(std::move(l)), id{i} {}

    // 5 mallocs (4create + 1copy + 1move)
    // the 9 combination version

    // 5 mallocs (4create + 1copy + 1move)
    // template version
```

C++ is tricky, you can do everything, you can even make every mistake

Pass-By-Value is good, but not perfect