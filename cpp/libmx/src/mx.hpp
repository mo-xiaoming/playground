#include <chrono>
#include <cstring>
#include <exception>
#include <fmt/core.h>
#include <random>
#include <ratio>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace mx {

// repeatedly release/grow vectors
template <typename T, std::size_t InitCapacity> class CyclingVector {
public:
    CyclingVector();
    CyclingVector(const CyclingVector& rhs) : CyclingVector{} { v_ = rhs.v_; }
    auto operator=(const CyclingVector&) & -> CyclingVector& = delete;
    CyclingVector(CyclingVector&&) noexcept = delete;
    auto operator=(CyclingVector&&) & noexcept -> CyclingVector& = delete;
    ~CyclingVector();

private:
    static std::vector<std::vector<T>> recycling_;
    std::vector<T> v_;
};

template <typename T, std::size_t InitCapacity>
CyclingVector<T, InitCapacity>::CyclingVector() {
    if (recycling_.empty()) {
        v_.reserve(InitCapacity);
    } else {
        v_ = std::move(recycling_.back());
        recycling_.pop_back();
    }
}

template <typename T, std::size_t InitCapacity>
CyclingVector<T, InitCapacity>::~CyclingVector() {
    v_.clear();
    recycling_.push_back(std::move(v_));
}

template <typename T, std::size_t InitCapacity>
std::vector<std::vector<T>> CyclingVector<T, InitCapacity>::recycling_{};

// number randomizer
template <auto Min, decltype(Min) Max> class RandomGen {
public:
    [[nodiscard]] auto next() { return dis_(rng_); }

private:
    std::random_device rd_{};
    std::mt19937 rng_{rd_()};
    std::uniform_int_distribution<decltype(Min)> dis_{Min, Max};
};

// uuid
class UuidGen {
public:
    using uuidType = boost::uuids::uuid;
    [[nodiscard]] static auto next() -> uuidType { return gen_(); }
    [[nodiscard]] static auto to_string(const uuidType& uuid) -> std::string {
        return boost::uuids::to_string(uuid);
    }

private:
    static boost::uuids::random_generator gen_;
};

// helper for std::variant, lambda overload
template <class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// for_each_argument([](auto i) { std::cout << i << '\n';}, 1, 3.7, "hello");
template <typename F, typename... Ts>
void for_each_argument(F&& f, Ts&&... ts) {
    // (void) is here just incase `,` is overloaded
    ((void)std::forward<F>(f)(std::forward<Ts>(ts)), ...);
}

// SCOPE_EXIT {}; SCOPE_FAIL {};
namespace internal {
enum class ScopeGuardOnExit {};
template <typename F> class ScopeGuard {
    F f_;

public:
    ~ScopeGuard() { f_(); }
    ScopeGuard(ScopeGuard const&) = default;
    auto operator=(ScopeGuard const&) & -> ScopeGuard& = default;
    ScopeGuard(ScopeGuard&&) noexcept = default;
    auto operator=(ScopeGuard&&) & noexcept -> ScopeGuard& = default;
    explicit ScopeGuard(F&& f) : f_{std::forward<F>(f)} {}
};
template <typename F>
auto operator+(ScopeGuardOnExit /*unused*/, F&& f) -> ScopeGuard<F> {
    return ScopeGuard<F>(std::forward<F>(f));
}
} // namespace internal

#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)

#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)

// NOLINT
#define SCOPE_EXIT                                                             \
    auto ANONYMOUS_VARIABLE(SCOPE_EXIT_STATE) =                                \
        ::mx::internal::ScopeGuardOnExit{} + [&]()

namespace internal {
enum class ScopeGuardOnFail {};

class UncaughtExceptionCounter {
    int exceptionCount_ = std::uncaught_exceptions();

public:
    UncaughtExceptionCounter() = default;
    [[nodiscard]] auto newUncaughtException() const noexcept {
        return std::uncaught_exceptions() > exceptionCount_;
    }
};

template <typename FunctionType, bool executeOnException>
class ScopeGuardForNewException {
    FunctionType function_;
    UncaughtExceptionCounter ec_;

public:
    explicit ScopeGuardForNewException(FunctionType const& fn)
        : function_{fn} {}
    explicit ScopeGuardForNewException(FunctionType&& fn)
        : function_{std::move(fn)} {}
    ScopeGuardForNewException(ScopeGuardForNewException const&) = default;
    auto operator=(
        ScopeGuardForNewException const&) & -> ScopeGuardForNewException& =
        default;
    ScopeGuardForNewException(ScopeGuardForNewException&&) noexcept = default;
    auto operator=(ScopeGuardForNewException&&) & noexcept
        -> ScopeGuardForNewException& = default;
    ~ScopeGuardForNewException() noexcept(executeOnException) {
        if (executeOnException == ec_.newUncaughtException()) {
            function_();
        }
    }
};

template <typename FunctionType>
auto operator+(ScopeGuardOnFail /*unused*/, FunctionType&& fn) {
    return ScopeGuardForNewException<std::decay_t<FunctionType>, true>(
        std::forward<FunctionType>(fn));
}
} // namespace internal

// NOLINT
#define SCOPE_FAIL                                                             \
    auto ANONYMOUS_VARIABLE(SCOPE_FAIL_STATE) =                                \
        ::mx::internal::ScopeGuardOnFail{} + [&]() noexcept

// functor helper for overloaded functions
// CppCon 2018 - Simon Brand “Overloading - The Bane of All Higher-Order
// Functions”-L_QKlAx31Pw
#define FWD(...) std::forward<decltype(__VAR_ARGS__)>(__VAR_ARGS__)

#define LIFE(X)                                                                \
    [](auto&&... args) noexcept(                                               \
        noexcept(X(FWD(args)...))) -> decltype(X(FWD(args)...)) {              \
        return X(FWD(args)...);                                                \
    }

// safe cast
template <typename To, typename From,
          typename = std::enable_if_t<(sizeof(To) == sizeof(From)) &&
                                      std::is_trivially_copyable_v<From> &&
                                      std::is_trivially_copyable_v<To>>>
[[nodiscard]] auto bit_cast(From const& f) noexcept -> To {
    To t;
    std::memcpy(&t, &f, sizeof(f));
    return t;
}

// fast (last-first)/2
template <typename T> auto half(T first, T last) {
    if constexpr (std::is_unsigned<T>) {
        return (last - first) / 2;
    } else {
        return static_cast<T>(std::make_unsigned<T>(last - first) / 2);
    }
}

// scoped exit
// auto cleanup = make_scope_exit([&i] { delete i; i = nullptr; });
template <typename T> struct scope_exit {
    explicit scope_exit(T&& t) : t_{std::move(t)} {}
    ~scope_exit() { t_(); }
    T t_;
};

template <typename T> scope_exit<T> make_scope_exit(T&& t) {
    return scope_exit<T>{std::move(t)};
}

// scoped timer
class ScopedTimer {
public:
    ScopedTimer() : start_{std::chrono::high_resolution_clock::now()} {}
    ~ScopedTimer() {
        fmt::printf("{}",
                    std::chrono<double, std::nano>(
                        std::chrono::high_resolution_clock::now() - start_));
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template <typename T> struct InstanceCounter {
    InstanceCounter() { ++counter_.numDefaultConstructor; }
    InstanceCounter(InstanceCounter const&) noexcept {
        ++counter_.numCopyConstructor;
    }
    InstanceCounter& operator=(InstanceCounter const&) & noexcept {
        ++counter_.numCopyAssignment;
        return *this;
    }
    InstanceCounter(InstanceCounter&&) noexcept {
        ++counter_.numMoveConstructor;
    }
    InstanceCounter& operator=(InstanceCounter&&) & noexcept {
        ++counter_.numMoveAssignment;
        return *this;
    }
    ~InstanceCounter() { ++counter_.numDestructor; }

private:
    static inline struct Counter {
        unsigned char numDefaultConstructor : 4 = 0;
        unsigned char numCopyConstructor : 4 = 0;
        unsigned char numCopyAssignment : 4 = 0;
        unsigned char numMoveConstructor : 4 = 0;
        unsigned char numMoveAssignment : 4 = 0;
        unsigned char numDestructor : 4 = 0;
        Counter& operator=(Counter&&) = delete;
        ~Counter() {
            if (numDefaultConstructor)
                std::printf("ctor  %d\n", numDefaultConstructor);
            if (numCopyConstructor)
                std::printf("copy  %d\n", numCopyConstructor);
            if (numCopyAssignment)
                std::printf("copy= %d\n", numCopyAssignment);
            if (numMoveConstructor)
                std::printf("move  %d\n", numMoveConstructor);
            if (numMoveAssignment)
                std::printf("move= %d\n", numMoveAssignment);
            if (numDestructor)
                std::printf("dtor  %d\n", numDestructor);
        }

    } counter_;
    static_assert(sizeof(counter_) == 3);
};

template <typename T> struct Counted : T, private InstanceCounter<T> {
    using T::T;
};

using String = Counted<std::string>;

static_assert(sizeof(std::string) == sizeof(String));

// noexcept C cast
template <typename Fnc> struct noexcept_cast_helper;

template <typename Ret, typename... Args>
struct noexcept_cast_helper<Ret (*)(Args...)> {
    using type = Ret (*)(Args...) noexcept;
};

template <typename T> auto noexcept_cast(const T obj) noexcept {
    return reinterpret_cast<typename noexcept_cast_helper<T>::type>(obj);
};

// the C function is not marked noexcept, but you know it won't throw
// extern "C" int foo();
// int fnc() noexcept {
//    return noexcept_cast(foo)();
//}

} // namespace mx
