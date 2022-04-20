#include <vector>

#include <benchmark/benchmark.h>

static void bench(benchmark::State &state) {
  auto const size = 1 << state.range(0);
  auto a = std::vector<unsigned char>(static_cast<std::size_t>(size), 0);
  constexpr auto const step = 64 * 1024 * 1024;
  auto const mask = static_cast<unsigned>(size - 1);
  for (auto _ : state) {
    for (auto i = 0U; i < step; ++i) {
      a[(64U * i) & mask] += static_cast<unsigned char>(3U);
    }
  }
}
BENCHMARK(bench)->DenseRange(10, 24);

BENCHMARK_MAIN();
