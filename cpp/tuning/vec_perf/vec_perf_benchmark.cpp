#include <deque>
#include <list>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <fmt/core.h>

template <auto Size> struct S {
  int a[Size] = {};
};

#define VECTOR(n)                                                                                                      \
  do {                                                                                                                 \
    auto v = std::vector<S<(n)>>{};                                                                                      \
    v.reserve(state.range(0));                                                                                         \
    for (auto _ : state) {                                                                                             \
      v.insert(v.cbegin(), S<n>{});                                                                                    \
      benchmark::DoNotOptimize(v.data());                                                                              \
      v.erase(v.cbegin());                                                                                             \
      benchmark::DoNotOptimize(v.data());                                                                              \
      benchmark::ClobberMemory();                                                                                      \
    }                                                                                                                  \
  } while (0)

static void vector_1(benchmark::State& state) { VECTOR(1); }
BENCHMARK(vector_1)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void vector_10(benchmark::State& state) { VECTOR(10); }
BENCHMARK(vector_10)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void vector_50(benchmark::State& state) { VECTOR(50); }
BENCHMARK(vector_50)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void vector_100(benchmark::State& state) { VECTOR(100); }
BENCHMARK(vector_100)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

#define DEQUE(n)                                                                                                       \
  do {                                                                                                                 \
    auto v = std::deque<S<1>>{};                                                                                       \
    for (auto _ : state) {                                                                                             \
      v.push_front(S<1>{});                                                                                            \
      v.pop_front();                                                                                                   \
    }                                                                                                                  \
  } while (0)

static void deque_1(benchmark::State& state) { DEQUE(1); }
BENCHMARK(deque_1)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void deque_10(benchmark::State& state) { DEQUE(10); }
BENCHMARK(deque_10)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void deque_50(benchmark::State& state) { DEQUE(50); }
BENCHMARK(deque_50)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void deque_100(benchmark::State& state) { DEQUE(100); }
BENCHMARK(deque_100)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

#define LIST(n)                                                                                                        \
  do {                                                                                                                 \
    auto v = std::list<S<1>>{};                                                                                        \
    for (auto _ : state) {                                                                                             \
      v.push_front(S<1>{});                                                                                            \
      v.pop_front();                                                                                                   \
    }                                                                                                                  \
  } while (0)

static void list_1(benchmark::State& state) { LIST(1); }
BENCHMARK(list_1)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void list_10(benchmark::State& state) { LIST(10); }
BENCHMARK(list_10)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void list_50(benchmark::State& state) { LIST(50); }
BENCHMARK(list_50)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

static void list_100(benchmark::State& state) { LIST(100); }
BENCHMARK(list_100)->Arg(100'000)->Arg(1'000'000)->Arg(10'000'000);

BENCHMARK_MAIN();
