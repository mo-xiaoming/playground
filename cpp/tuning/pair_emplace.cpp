#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

using namespace std::string_literals;

auto const A = "hellllllllllllllllllllllllllllllllllo"s;
auto const B = "worllllllllllllllllllllllllllllllllld"s;
auto const C = "worllllllllllllllllllllllllllllllllld"s;
auto const D = "worllllllllllllllllllllllllllllllllld"s;
auto const E = "worllllllllllllllllllllllllllllllllld"s;
auto const K = "keyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"s;
constexpr auto SZ = 10000U;

struct S {
    explicit S(std::string a) : a{std::move(a)} {}

private:
    std::string a;
};

static void bench_copy(benchmark::State& state) {
    auto v = std::vector<std::pair<std::string, S>>{};
    v.reserve(SZ);

    for (auto _ : state) {
        for (auto i = 0U; i < SZ; ++i) {
            v.emplace_back(K, S{"helooooooooooooooooooooooooooooooooooooooo"s});
        }
        benchmark::DoNotOptimize(v.back());
        v.clear();
    }
}
BENCHMARK(bench_copy);

static void bench_inplace(benchmark::State& state) {
    auto v = std::vector<std::pair<std::string, S>>{};
    v.reserve(SZ);

    for (auto _ : state) {
        for (auto i = 0U; i < SZ; ++i) {
            v.emplace_back(std::piecewise_construct, std::forward_as_tuple(K),
                           std::forward_as_tuple(
                               "helooooooooooooooooooooooooooooooooooooooo"s));
        }
        benchmark::DoNotOptimize(v.back());
        v.clear();
    }
}
BENCHMARK(bench_inplace);

BENCHMARK_MAIN();
