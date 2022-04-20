#include <exception>
#include <optional>

#include <benchmark/benchmark.h>
#include <stdexcept>

namespace {
void throw_int() { throw 3; }
void throw_runtime_short() { throw std::runtime_error("short"); }
void throw_runtime_long() { throw std::runtime_error("really loooooooooooooooooooooooooooooong error message"); }
void throw_runtime_long_no_throw(int64_t s) {
  if (s == 7) {
    throw std::runtime_error("really loooooooooooooooooooooooooooooong error message");
  }
}
auto by_return() -> int { return 3; }
auto by_optional() -> std::optional<int> { return {}; }
} // namespace

static void bench_throw_int(benchmark::State& state) {
  for (auto _ : state) {
    try {
      throw_int();
    } catch (int i) {
      benchmark::DoNotOptimize(&i);
    }
  }
}
BENCHMARK(bench_throw_int);

static void bench_runtime_short(benchmark::State& state) {
  for (auto _ : state) {
    try {
      throw_runtime_short();
    } catch (std::runtime_error const& i) {
      benchmark::DoNotOptimize(&i);
    }
  }
}
BENCHMARK(bench_runtime_short);

static void bench_runtime_long(benchmark::State& state) {
  for (auto _ : state) {
    try {
      throw_runtime_long();
    } catch (std::runtime_error const& i) {
      benchmark::DoNotOptimize(&i);
    }
  }
}
BENCHMARK(bench_runtime_long);

static void bench_runtime_long_no_throw(benchmark::State& state) {
  for (auto _ : state) {
    try {
      throw_runtime_long_no_throw(state.range(0));
    } catch (std::runtime_error const& i) {
      benchmark::DoNotOptimize(&i);
    }
  }
}
BENCHMARK(bench_runtime_long_no_throw)->Arg(3);

static void bench_by_return(benchmark::State& state) {
  for (auto _ : state) {
    auto const r = by_return();
    benchmark::DoNotOptimize(&r);
  }
}
BENCHMARK(bench_by_return);

static void bench_by_optional(benchmark::State& state) {
  for (auto _ : state) {
    auto const r = by_optional();
    benchmark::DoNotOptimize(&r);
  }
}
BENCHMARK(bench_by_optional);

BENCHMARK_MAIN();
