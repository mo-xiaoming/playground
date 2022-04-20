#include <iostream>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "mx.hpp"

static void bench_00(benchmark::State& state) {
  int a[2]{17, 87};
  for (auto _ : state) {
    for (int i = 0; i < 64 * 1024 * 1024; ++i) {
      a[0] *= 14;
      benchmark::DoNotOptimize(a);
      a[0] += 31;
      benchmark::DoNotOptimize(a);
    }
  }
}
BENCHMARK(bench_00)->Unit(benchmark::kMillisecond);

static void bench_01(benchmark::State& state) {
  int a[2]{17, 87};
  for (auto _ : state) {
    for (int i = 0; i < 64 * 1024 * 1024; ++i) {
      a[0] *= 14;
      benchmark::DoNotOptimize(a);
      a[1] += 31;
      benchmark::DoNotOptimize(a + 1);
    }
  }
}
BENCHMARK(bench_01)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
