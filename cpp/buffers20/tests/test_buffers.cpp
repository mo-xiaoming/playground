#include <string>

#include <catch2/catch.hpp>

#include <buffers.hpp>

namespace {
struct S {};
} // namespace

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("not trivally copyable objects", "[copyable][false]", int,
                   S) {
    REQUIRE(Trivially_copyable<TestType>); // NOLINT
}

// NOLINTNEXTLINE
TEMPLATE_TEST_CASE("trivally copyable objects", "[copyable][true]", int[], S[],
                   std::string) {
    REQUIRE(!Trivially_copyable<TestType>); // NOLINT
}
