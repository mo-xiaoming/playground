#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string_view>
#include <type_traits>

struct OutStream {
    OutStream() = default;
    OutStream(OutStream const&) = delete;
    OutStream(OutStream&&) = delete;
    auto operator=(OutStream const&) -> OutStream& = delete;
    auto operator=(OutStream &&) -> OutStream& = delete;
    virtual ~OutStream();
    virtual void output(std::string_view msg) = 0;
};

struct DefaultOutStream : OutStream {
    void output(std::string_view msg) override { std::puts(msg.data()); }
};

namespace logging {
enum class LogLevel : std::uint8_t { Debug, Info, Error, Off };

template <typename OutStreamType = DefaultOutStream> struct Logger {
    static auto get_out_stream() -> OutStreamType& {
        return impl.get_out_stream();
    }
    static void set_level(LogLevel level) { impl.do_set_level(level); }
    static auto get_level() -> LogLevel { return impl.do_get_level(); }

    static void info(std::string_view msg) { return impl.do_info(msg); }

    template <typename F>
    requires std::is_invocable_v<F> static void info(F&& f) {
        impl.do_info(std::forward<F>(f)());
    }

private:
    static inline struct Impl {
        void do_info(std::string_view msg) {
            if (level_ <= LogLevel::Info) {
                out_.output(msg);
            }
        }

        void do_set_level(LogLevel level) { level_ = level; }
        [[nodiscard]] auto do_get_level() const -> LogLevel { return level_; }

        [[nodiscard]] auto get_out_stream() -> OutStreamType& { return out_; }

    private:
        std::atomic<LogLevel> level_{LogLevel::Info};
        OutStreamType out_;
    } impl;
};

#if 0
struct Logger {
    static void set_out_stream(std::shared_ptr<OutStream> const& out) {
        Logger::instance().do_set_out_stream(out);
    }
    static void set_level(LogLevel level) {
        Logger::instance().do_set_level(level);
    }
    static auto get_level() -> LogLevel {
        return Logger::instance().do_get_level();
    }

    static void info(std::string_view msg) {
        return Logger::instance().do_info(msg);
    }

    template <typename F>
    requires std::is_invocable_v<F> static void info(F&& f) {
        Logger::instance().do_info(std::forward<F>(f)());
    }

private:
    static auto instance() -> Logger& {
        static Logger logger;
        return logger;
    }

    void do_info(std::string_view msg) {
        if (level_ <= logging::LogLevel::Info) {
            out_->output(msg);
        }
    }

    void do_set_out_stream(std::shared_ptr<OutStream> const& out) {
        out_ = out;
    }

    void do_set_level(logging::LogLevel level) { level_ = level; }
    [[nodiscard]] auto do_get_level() const -> logging::LogLevel {
        return level_;
    }

    std::atomic<logging::LogLevel> level_{logging::LogLevel::Info};
    std::shared_ptr<OutStream> out_{std::make_shared<DefaultOutStream>()};
};
#endif
} // namespace logging
