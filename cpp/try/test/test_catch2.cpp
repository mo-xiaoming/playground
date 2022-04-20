#include <cstdint>
#include <exception>
#include <list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include <fmt/core.h>

#pragma GCC diagnostic ignored "-Wdisabled-macro-expansion"

namespace {
auto Factorial(int number) -> int {
    return number <= 1 ? 1 : Factorial(number - 1) * number;
}

struct A : public std::exception {
    explicit A(std::string msg) : msg_(std::move(msg)) {}
    [[nodiscard]] auto what() const noexcept -> char const* override {
        return msg_.c_str();
    }

private:
    std::string const msg_;
};
} // namespace

using namespace std::literals;

TEST_CASE("Factorials are computed") {
    REQUIRE(Factorial(0) == 1);        // NOLINT
    REQUIRE(Factorial(1) == 1);        // NOLINT
    REQUIRE(Factorial(2) == 2);        // NOLINT
    REQUIRE(Factorial(3) == 6);        // NOLINT
    REQUIRE(Factorial(10) == 3628800); // NOLINT
}

TEST_CASE("vectors can be sized and resized", "[vector]") {

    std::vector<int> v(5); // NOLINT

    REQUIRE(v.size() == 5);     // NOLINT
    REQUIRE(v.capacity() >= 5); // NOLINT

    SECTION("resizing bigger changes size and capacity") {
        v.resize(10); // NOLINT

        REQUIRE(v.size() == 10);     // NOLINT
        REQUIRE(v.capacity() >= 10); // NOLINT
    }
    SECTION("resizing smaller changes size but not capacity") {
        v.resize(0);

        REQUIRE(v.size() == 0);     // NOLINT
        REQUIRE(v.capacity() >= 5); // NOLINT
    }
    SECTION("reserving bigger changes capacity but not size") {
        v.reserve(10); // NOLINT

        REQUIRE(v.size() == 5);      // NOLINT
        REQUIRE(v.capacity() >= 10); // NOLINT
    }
    SECTION("reserving smaller does not change size or capacity") {
        v.reserve(0);

        REQUIRE(v.size() == 5);     // NOLINT
        REQUIRE(v.capacity() >= 5); // NOLINT
    }
}

SCENARIO("vectors can be sized and resized", "[vector]") {

    GIVEN("A vector with some items") {
        std::vector<int> v(5); // NOLINT

        REQUIRE(v.size() == 5);     // NOLINT
        REQUIRE(v.capacity() >= 5); // NOLINT

        AND_GIVEN("Another empty vector") {
            auto a = std::vector<int>();
            WHEN("First vector copied into the second") {
                a = v;
                THEN("The size of first vector doesn't change") {
                    REQUIRE(v.size() == 5); // NOLINT
                    AND_THEN("The second vector have the same size as the "
                             "first one") {
                        REQUIRE(a.size() == 5); // NOLINT
                    }
                }
            }
        }
        WHEN("the size is increased") {
            v.resize(10); // NOLINT

            THEN("the size and capacity change") {
                REQUIRE(v.size() == 10);     // NOLINT
                REQUIRE(v.capacity() >= 10); // NOLINT
            }
        }
        WHEN("the size is reduced") {
            v.resize(0);

            THEN("the size changes but not capacity") {
                REQUIRE(v.size() == 0);     // NOLINT
                REQUIRE(v.capacity() >= 5); // NOLINT
            }
        }
        WHEN("more capacity is reserved") {
            v.reserve(10); // NOLINT

            THEN("the capacity changes but not the size") {
                REQUIRE(v.size() == 5);      // NOLINT
                REQUIRE(v.capacity() >= 10); // NOLINT
            }
        }
        WHEN("less capacity is reserved") {
            v.reserve(0);

            THEN("neither size nor capacity are changed") {
                REQUIRE(v.size() == 5);     // NOLINT
                REQUIRE(v.capacity() >= 5); // NOLINT
            }
        }
    }
}

TEST_CASE("decomposable expressions") {
    auto const a = 1;
    auto const b = 2;
    // paranthesis must be added
    REQUIRE((a == 1 && b == 2)); // NOLINT
}

TEST_CASE("floating point comparasion") {
    using namespace Catch::literals;
    auto const a = 3.5;
    REQUIRE(a == 3.5_a);       // NOLINT
    REQUIRE(a == Approx(3.5)); // NOLINT
}

TEST_CASE("exceptions") {
    auto const a = [] {};
    auto const b = [] { throw A("hello"); };
    REQUIRE_NOTHROW(a());
    REQUIRE_THROWS(b());
    REQUIRE_THROWS_AS(b(), A);
    REQUIRE_THROWS_WITH(b(), "hello");
    REQUIRE_THROWS_WITH(b(), Catch::Contains("ell"));
    REQUIRE_THROWS_MATCHES(b(), A,
                           Catch::Matchers::Predicate<A>(
                               [](A const& e) { return e.what() == "hello"s; },
                               "should be hello"));
    REQUIRE_THROWS_MATCHES(b(), A, Catch::Matchers::Message("hello"));
}

TEST_CASE("epxression with comma") {
    // based on document, this shouldn't work
    REQUIRE(std::pair<int, int>(1, 2) == std::pair<int, int>(1, 2));
}

TEST_CASE("matchers") {
    using namespace Catch::Matchers;

    SECTION("for string") {
        auto const v = "Hello world"s;
        REQUIRE_THAT(v, StartsWith("hello", Catch::CaseSensitive::No) &&
                            !Contains("haha") && Equals("Hello world") &&
                            Matches("^Hello *world$"));
        REQUIRE_THAT(v, !Predicate<std::string>(
                            [](auto const& s) { return s.front() == s.back(); },
                            "First and last character should not be equal"));
    }
    SECTION("for vectors") {
        auto const v = std::vector{1, 2, 3};
        REQUIRE_THAT(v, Contains<int>({1, 3}));
        REQUIRE_THAT(v, VectorContains<int>(3));
        REQUIRE_THAT(v, Equals<int>({1, 2, 3}));
        REQUIRE_THAT(v, UnorderedEquals<int>({3, 2, 1}));
        auto const d = std::vector{1.0, 2.0, 3.0};
        CHECK_THAT(
            d, Catch::Matchers::Approx<double>({1.00001, 2.00001, 3.00001}));
    }
}

TEST_CASE("custom matchers") {
    struct IntRange : public Catch::MatcherBase<int> {
        IntRange(int begin, int end) : begin_(begin), end_(end) {}
        bool match(int const& i) const override {
            return i >= begin_ && i <= end_;
        }
        std::string describe() const override {
            return fmt::format("is between {} and {}", begin_, end_);
        }

    private:
        int const begin_;
        int const end_;
    };

    auto const isBetween = [](int begin, int end) {
        return IntRange(begin, end);
    };

    CHECK_THAT(3, isBetween(1, 10));
}

TEST_CASE("tags hidden", "[.integration]") {
    // hidden from normal run, but can be selected with [integration]
    REQUIRE(true);
}

TEST_CASE("tags work-in-progress", "[!mayfail]") {
    // still report failed
    REQUIRE(false);
}

TEMPLATE_TEST_CASE("build-in types are trivially copyable", "", int, double,
                   char) {
    REQUIRE(std::is_trivially_copyable_v<TestType>);
}

using MyTypes = std::tuple<int, char, float>;
TEMPLATE_LIST_TEST_CASE("like above example, types are specified in tuple",
                        "[template][list]", MyTypes) {
    REQUIRE(sizeof(TestType) > 0);
}

TEMPLATE_PRODUCT_TEST_CASE("template product", "[template][product]",
                           (std::vector, std::list), (int, char)) {
    auto x = TestType();
    REQUIRE(x.size() == 0);
}

// different number of type parameters
TEMPLATE_PRODUCT_TEST_CASE("Product with differing arities",
                           "[template][product]", std::tuple,
                           (int, (int, double), (int, double, float))) {
    REQUIRE(std::tuple_size<TestType>::value >= 1);
}

TEMPLATE_TEST_CASE_SIG(
    "TemplateTestSig: arrays can be created from NTTP arguments",
    "[vector][template][nttp]", ((typename T, int V), T, V), (int, 5),
    (float, 4), (std::string, 15), ((std::tuple<int, float>), 6)) {

    std::array<T, V> v;
    REQUIRE(v.size() > 1);
}

template <typename T, size_t S> struct Bar {
    size_t size() { return S; }
};

TEMPLATE_PRODUCT_TEST_CASE_SIG(
    "A Template product test case with array signature",
    "[template][product][nttp]", ((typename T, size_t S), T, S),
    (std::array, Bar), ((int, 9), (float, 42))) {
    auto x = TestType();
    REQUIRE(x.size() > 0);
}

TEST_CASE("Generators") {
    SECTION("simple") {
        auto i = GENERATE(1, 3, 5);
        auto j = GENERATE(2, 4);
        // 3*2 times
        REQUIRE([](int m, int n) { return m * n <= 20; }(i, j));
    }
    SECTION("nested") {
        auto i = GENERATE(1, 2);
        SECTION("one") {
            auto j = GENERATE(-3, -2);
            REQUIRE(j < i); // 4 times
        }
        SECTION("two") {
            auto k = GENERATE(4, 5, 6);
            REQUIRE(i != k); // 6 times
        }
    }
}

TEST_CASE("other macros") {
    SECTION("if else") {
        auto a = 3;
        auto b = 4;
        CHECKED_IF(a < b) { REQUIRE(a < b); }
        CHECKED_ELSE(a < b) { REQUIRE(a >= b); }
    }
    SECTION("no fail") { CHECK_NOFAIL(1 == 2); }
    SECTION("static require") {
        STATIC_REQUIRE(std::is_void_v<void>);
        STATIC_REQUIRE_FALSE(std::is_void_v<int>);
    }
}

static auto Fibonacci(std::uint64_t number) -> std::uint64_t {
    return number < 2 ? 1 : Fibonacci(number - 1) + Fibonacci(number - 2);
}

TEST_CASE("Fibonacci") {
    // BE AWARE OF the TRAILING COMMA
    BENCHMARK("Fibonacci 20") { return Fibonacci(20); };

    BENCHMARK("Fibonacci 25") { return Fibonacci(25); };

    BENCHMARK("Fibonacci 30") { return Fibonacci(30); };

    BENCHMARK("Fibonacci 35") { return Fibonacci(35); };
}

TEST_CASE("benchmarks need setup") {
    BENCHMARK("simple") { return std::string(); };

    BENCHMARK_ADVANCED("advance")(Catch::Benchmark::Chronometer meter) {
        auto s = std::string();
        meter.measure([&s] {
            auto b = s;
            return b;
        });
    };

    BENCHMARK_ADVANCED("advanced mutable")
    (Catch::Benchmark::Chronometer meter) {
        auto v =
            std::vector<std::string>(static_cast<std::uint32_t>(meter.runs()));
        std::fill(v.begin(), v.end(), "abc");
        meter.measure(
            [&v](std::vector<std::string>::size_type i) { v[i] = "def"; });
    };

    const auto f = [](std::uint64_t i) { return std::vector<char>(i); };
    BENCHMARK("indexed, same as above", i) {
        return f(static_cast<std::uint64_t>(i));
    };
}

TEST_CASE("ctor and dtor") {
    BENCHMARK_ADVANCED("construct")(Catch::Benchmark::Chronometer meter) {
        std::vector<Catch::Benchmark::storage_for<std::string>> storage(
            static_cast<std::uint64_t>(meter.runs()));
        meter.measure([&](std::uint64_t i) { storage[i].construct("thing"); });
    };

    BENCHMARK_ADVANCED("destroy")(Catch::Benchmark::Chronometer meter) {
        std::vector<Catch::Benchmark::destructable_object<std::string>> storage(
            static_cast<std::uint64_t>(meter.runs()));
        for (auto&& o : storage)
            o.construct("thing");
        meter.measure([&](std::uint64_t i) { storage[i].destruct(); });
    };
}
