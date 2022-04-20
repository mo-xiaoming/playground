/*

`bench_col/100000`:
     4,869,231,336      L1-dcache-loads           #  616.193 M/sec                    (38.48%)
        24,753,645      L1-dcache-load-misses     #    0.51% of all L1-dcache hits    (38.48%)

`bench_row/100000`:
     2,093,002,518      L1-dcache-loads           #  340.101 M/sec                    (38.43%)
       317,286,927      L1-dcache-load-misses     #   15.16% of all L1-dcache hits    (38.51%)

*/
#include <algorithm>

#include <benchmark/benchmark.h>

#include "mx.hpp"

static void bench_col(benchmark::State& state) {
  auto v = std::vector<unsigned char>{};
  auto constexpr LENGTH = 100'000'000;
  v.resize(LENGTH);
  auto g = mx::RandomGen<0U, 100U>{};
  std::generate(begin(v), end(v), [&g] { return g.next(); });

  auto const COLS = state.range(0);
  for (auto _ : state) {
    auto sum = uint64_t{};
    benchmark::DoNotOptimize(&sum);
    for (int r = 0; r < LENGTH / COLS; ++r) {
      for (int c = 0; c < COLS; ++c) {
        sum += v[c + r * COLS];
        benchmark::DoNotOptimize(&sum);
      }
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_col)->Range(10, 100'000'000);
// BENCHMARK(bench_col)->Arg(10)->Arg(1'000)->Arg(100'000)->Arg(10'000'000);

static void bench_row(benchmark::State& state) {
  auto v = std::vector<unsigned char>{};
  auto constexpr LENGTH = 100'000'000;
  v.resize(LENGTH);
  auto g = mx::RandomGen<0U, 100U>{};
  std::generate(begin(v), end(v), [&g] { return g.next(); });

  auto const COLS = state.range(0);
  for (auto _ : state) {
    auto sum = uint64_t{};
    benchmark::DoNotOptimize(&sum);
    for (int c = 0; c < COLS; ++c) {
      for (int r = 0; r < LENGTH / COLS; ++r) {
        sum += v[c + r * COLS];
        benchmark::DoNotOptimize(&sum);
      }
    }
    benchmark::DoNotOptimize(&sum);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(bench_row)->Range(10, 100'000'000);
// BENCHMARK(bench_row)->Arg(10)->Arg(1'000)->Arg(100'000)->Arg(10'000'000);

BENCHMARK_MAIN();
