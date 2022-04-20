#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "mx.hpp"

namespace {
struct B {
  virtual ~B() = default;
  [[nodiscard]] virtual auto foo() const -> int { return 0; }
};

struct D1 : B {
  [[nodiscard]] auto foo() const -> int override { return 1; }
};

struct D2 : B {
  [[nodiscard]] auto foo() const -> int override { return 2; }
};
} // namespace

static void bench_shuffle(benchmark::State& state) {
  auto v = std::vector<B*>{};
  std::fill_n(back_inserter(v), 10'000, new B{});
  std::fill_n(back_inserter(v), 10'000, new D1{});
  std::fill_n(back_inserter(v), 10'000, new D2{});

  std::shuffle(begin(v), end(v), std::mt19937(std::random_device()()));

  for (auto _ : state) {
    auto s = 0;
    for (auto a : v) {
      s += a->foo();
    }
    benchmark::DoNotOptimize(&s);
  }
}
BENCHMARK(bench_shuffle);

static void bench(benchmark::State& state) {
  auto v = std::vector<B*>{};
  std::fill_n(back_inserter(v), 10'000, new B{});
  std::fill_n(back_inserter(v), 10'000, new D1{});
  std::fill_n(back_inserter(v), 10'000, new D2{});

  for (auto _ : state) {
    auto s = 0;
    for (auto a : v) {
      s += a->foo();
    }
    benchmark::DoNotOptimize(&s);
  }
}
BENCHMARK(bench);

BENCHMARK_MAIN();
