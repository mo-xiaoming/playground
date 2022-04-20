#pragma once

#include <fmt/core.h>

#include "logger.hpp"

struct MockLogger {
    static void set_level(logging::LogLevel /*unused*/) {}
    template <typename T> static void info(T&& /*unused*/) {}
};

template <typename T = logging::Logger<DefaultOutStream>>
auto foo(int n) -> int {
    constexpr auto UPPER = 10;
    constexpr auto LOWER = -10;
    if (n > UPPER) {
        T::info([] { return "large number"; });
        return 1;
    }
    if (n < LOWER) {
        T::info("small number");
        return -1;
    }
    T::set_level(logging::LogLevel::Info);
    T::info("normal number");
    return 0;
}
