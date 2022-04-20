#include <algorithm>
#include <vector>

#include <catch2/catch.hpp>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

using namespace std::literals;
using fmt::literals::operator""_a;
using fmt::literals::operator""_format;

enum class Log_level : std::uint8_t { debug, info, error, off };
namespace fmt {
template <> struct [[maybe_unused]] formatter<Log_level> : formatter<std::string_view> {
    template <typename Format_context> auto format(Log_level level, Format_context& ctx) {
        static constexpr auto levels = std::array{"debug", "info", "error", "off"};
        auto const s = levels[std::underlying_type_t<Log_level>(level)]; // NOLINT
        return formatter<std::string_view>::format(s, ctx);
    }
};
} // namespace fmt

struct A {
    A() = default;
    A(A const&) = default;
    A(A&&) = default;
    A& operator=(A const&) = default;
    A& operator=(A&&) = default;
    virtual ~A();
    [[nodiscard]] virtual std::string name() const { return "A"; }
};

A::~A() = default;

struct B : A {
    [[nodiscard]] std::string name() const final;
};

std::string B::name() const { return "B"; }

namespace fmt {
template <typename T> requires std::is_base_of_v<A, T> struct formatter<T> : formatter<std::string> {
    template <typename Format_context> auto format(A const& a, Format_context& ctx) {
        return formatter<std::string>::format(a.name(), ctx);
    }
};
} // namespace fmt

TEST_CASE("format variants") { // NOLINT
    constexpr auto answer_to_universe = 42;
    constexpr auto answer_string = "The answer is 42"sv;

    SECTION("basic format") {
        auto const r = "The answer is {}"_format(answer_to_universe);
        REQUIRE(r == answer_string);
    }

    SECTION("format to memory buffer, avoid std::string construction", "[!mayfail]") {
        auto out = fmt::memory_buffer();
        fmt::format_to(out, "The answer is {}", answer_to_universe);
        auto* r = out.data(); // should not use r directly, there is no trailing \0
        static_assert(std::is_same_v<decltype(r), char*>);
        REQUIRE(std::string_view(r, out.size()) == answer_string);
        REQUIRE(fmt::to_string(out) == answer_string);
    }

    SECTION("positional arguments") {
        auto const r = "I'd rather be {1} than {0}."_format("right", "happy");
        REQUIRE(r == "I'd rather be happy than right.");
    }

    SECTION("named arguments") {
        auto const r =
            fmt::format("Hello, {name}! The answer is {number}. Goodbye, {name}", "name"_a = "World", "number"_a = 42);
        REQUIRE(r == "Hello, World! The answer is 42. Goodbye, World");
    }

    SECTION("compilation error") {
        // fmt::format("Cyrillic letter{}", L'\x42e');
        // fmt::format(FMT_STRING("The answer is {:d}"), "forty-two");

        // fmt::format("The answer is {:d}", "forty-two"); // run-time error
    }

    SECTION("user defined type") {
        REQUIRE("{}"_format(Log_level::debug) == "debug");
        REQUIRE("{}"_format(Log_level::info) == "info");
        REQUIRE("{}"_format(Log_level::error) == "error");
        REQUIRE("{}"_format(Log_level::off) == "off");

        SECTION("inheritance") {
            REQUIRE("{}"_format(A()) == "A");
            REQUIRE("{}"_format(B()) == "B");
            A const& a = B{};
            REQUIRE("{}"_format(a) == "B");
        }
    }

    SECTION("to string") {
        auto const s = fmt::to_string(42);
        REQUIRE(std::is_same_v<decltype(s), std::string const>);
        REQUIRE(s == "42");
    }

    SECTION("format to container") {
        auto out = std::vector<char>();
        fmt::format_to(std::back_inserter(out), "{}", answer_to_universe);
        REQUIRE(std::equal(out.cbegin(), out.cend(), "42"));
        REQUIRE_THAT(out, Catch::Matchers::Equals<char>({'4', '2'}));
    }
}

TEST_CASE("system error") {
    constexpr auto filename = "made_up"sv;
    SECTION("create exception") {
        auto const f = [&filename] {
            if (auto* const file = std::fopen(filename.data(), "r"); file == nullptr) { // NOLINT
                throw fmt::system_error(errno, "cannot open file '{}'", filename);
            }
        };

        REQUIRE_THROWS_MATCHES(f(), std::runtime_error,
                               Catch::Matchers::Predicate<std::runtime_error>([&filename](auto const& e) noexcept {
                                   return std::string_view(e.what()).starts_with(
                                       "cannot open file '{}': "_format(filename));
                               }));
    }
    SECTION("format error message") {
        auto* const file = std::fopen(filename.data(), "r"); // NOLINT
        int const stored_errno = errno;
        CHECK(file == nullptr);
        auto out = fmt::memory_buffer();
        fmt::format_system_error(out, stored_errno, "system error");
        REQUIRE_THAT(fmt::to_string(out), Catch::Matchers::StartsWith("system error: "));
    }
}

TEST_CASE("format collections") {
    SECTION("tuple") {
        auto const t = std::tuple{'a', 1, 2.0F};
        REQUIRE("{}"_format(t) == "('a', 1, 2.0)");
        REQUIRE("{}"_format(fmt::join(t, ", ")) == "a, 1, 2.0");
    }

    SECTION("formats vector") {
        auto const v = std::array{1, 2, 3};
        REQUIRE("{}"_format(fmt::join(v, ", ")) == "1, 2, 3");
        REQUIRE("{:02}"_format(fmt::join(v, ", ")) == "01, 02, 03");
    }
}

TEST_CASE("datetime") {
    auto t = std::time(nullptr);
    REQUIRE_THAT(fmt::format("{:%Y-%m-%d}", fmt::localtime(t)), Catch::Matches(R"(\d{4}-\d{2}-\d{2})"));
}

enum class Code_quality : std::uint8_t { good, bad };
static std::ostream& operator<<(std::ostream& os, Code_quality quality) {
    static constexpr auto qualities = std::array{"good", "bad"};
    return os << qualities[std::underlying_type_t<Log_level>(quality)]; // NOLINT
}

TEST_CASE("ostream") {
    REQUIRE("{}"_format(Code_quality::good) == "good");
    REQUIRE("{}"_format(Code_quality::bad) == "bad");
}

TEST_CASE("format specs") {
    // {:spec} ::= [[fill]align][sign]["#"]["0"][width]["." precision][type]

    SECTION("alignment with width specified") {
        REQUIRE("{:<30}"_format("left aligned") == "left aligned                  ");
        REQUIRE("{:>30}"_format("right aligned") == "                 right aligned");
        REQUIRE("{:^30}"_format("centered") == "           centered           ");
        REQUIRE("{:*^30}"_format("centered") == "***********centered***********");
    }

    SECTION("dynamic width") { REQUIRE("{:<{}}"_format("left aligned", 30) == "left aligned                  "); }

    constexpr double pi = 3.14;
    SECTION("dynamic precision") { REQUIRE(fmt::format(FMT_STRING("{:.{}f}"), pi, 1) == "3.1"); }

    SECTION("sign") {
        REQUIRE(fmt::format("{:+f}; {:+f}", pi, -pi) == "+3.140000; -3.140000");
        REQUIRE(fmt::format("{: f}; {: f}", pi, -pi) == " 3.140000; -3.140000");
        REQUIRE(fmt::format("{:-f}; {:-f}", pi, -pi) == "3.140000; -3.140000");
    }

    SECTION("different bases") {
        REQUIRE(fmt::format("int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42) ==
                "int: 42;  hex: 2a;  oct: 52; bin: 101010");
        REQUIRE(fmt::format("int: {0:d};  hex: {0:#x};  oct: {0:#o};  bin: {0:#b}", 42) ==
                "int: 42;  hex: 0x2a;  oct: 052;  bin: 0b101010");
    }

    SECTION("with prefix and padding") { REQUIRE(fmt::format("{:#04x}", 0) == "0x00"); }

    SECTION("box") {
        REQUIRE(fmt::format(FMT_STRING("┌{0:─^{2}}┐\n"
                                       "│{1: ^{2}}│\n"
                                       "└{0:─^{2}}┘\n"),
                            "", "Hello, world!", 20) == "┌────────────────────┐\n"
                                                        "│   Hello, world!    │\n"
                                                        "└────────────────────┘\n");
    }

    SECTION("locale") { REQUIRE(fmt::format(std::locale("en_US.UTF-8"), "{:L}", 1234567890) == "1,234,567,890"); }
}
