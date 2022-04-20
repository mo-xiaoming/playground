#include <algorithm>

#include <benchmark/benchmark.h>

/*
 * probably not
 */
void multiplyAdd(float* b1, float* b2, float f, size_t s) {
  for (auto i = 0U; i < s; ++i)
    b1[i] += b2[i] * f;
}

static void bench_align(benchmark::State& state) {
  const int size = 32 * 1024;
  auto b1 = new float[size];
  auto b2 = new float[size];
  for (auto _ : state) {
    multiplyAdd(b1, b2, 0.0001f, size);
  }
}
BENCHMARK(bench_align);

static void bench(benchmark::State& state) {
  const int size = 32 * 1024;
  auto b1 = new float[size];
  auto b2 = new float[size];
  for (auto _ : state) {
    multiplyAdd(b1 + 1, b2 + 2, 0.0001f, size - 2);
  }
}
BENCHMARK(bench);

BENCHMARK_MAIN();
