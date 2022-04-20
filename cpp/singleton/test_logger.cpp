#include <catch2/catch.hpp>

#include "logger.hpp"

namespace {
struct MockOutStream : OutStream {
    void output(std::string_view msg) override {
        ++count_;
        message_ = msg;
    }
    [[maybe_unused]] void reset() {
        count_ = 0;
        message_.clear();
    }
    [[nodiscard]] constexpr auto counter() const noexcept -> int {
        return count_;
    }
    [[nodiscard]] auto message() const& noexcept -> std::string const& {
        return message_;
    }
    [[maybe_unused, nodiscard]] auto message() && noexcept -> std::string {
        return std::move(message_);
    }

private:
    std::string message_;
    int count_ = 0;
};
} // namespace

using namespace logging;

using MockLogger = Logger<MockOutStream>;

TEST_CASE("initial logger") {
    // NOLINTNEXTLINE
    REQUIRE(MockLogger::get_level() == LogLevel::Info); // NOLINT
}

TEST_CASE("set and get log level") {
    for (auto const v :
         {LogLevel::Debug, LogLevel::Info, LogLevel::Error, LogLevel::Off}) {
        MockLogger::set_level(v);
        DYNAMIC_SECTION("log level is: "
                        << static_cast<std::underlying_type_t<LogLevel>>(v)) {
            CHECK(MockLogger::get_level() == v); // NOLINT
        }
    }
}

SCENARIO("filter logs based on log level", "[logger]") {
    std::string_view const msg = "msg";
    SECTION("log level set to Debug") {
        MockLogger::set_level(LogLevel::Debug);
        MockLogger::info(msg);
        REQUIRE(MockLogger::get_out_stream().counter() == 1);   // NOLINT
        REQUIRE(MockLogger::get_out_stream().message() == msg); // NOLINT
    }
}
