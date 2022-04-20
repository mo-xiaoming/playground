#include <catch2/catch.hpp>

#include "fib.hpp"

TEST_CASE("number test") {
    auto const new_foo = [](auto i) { return foo<MockLogger>(i); };
    CHECK(new_foo(0) == 0);    // NOLINT
    CHECK(new_foo(-3) == 0);   // NOLINT
    CHECK(new_foo(3) == 0);    // NOLINT
    CHECK(new_foo(10) == 0);   // NOLINT
    CHECK(new_foo(-10) == 0);  // NOLINT
    CHECK(new_foo(11) == 1);   // NOLINT
    CHECK(new_foo(-11) == -1); // NOLINT
}
